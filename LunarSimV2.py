import os
import sys

import numpy as np
import pandas as pd
import spiceypy as spice
import rasterio
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from scipy.integrate import solve_ivp
from datetime import datetime, timedelta
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.ndimage import map_coordinates
from pathlib import Path
from functools import lru_cache

# -----------------------------
# Paths
# -----------------------------

KERNEL_DIR = Path("kernels")
DIVINER_DIR = Path("data/diviner_level4")


# -----------------------------
# Load SPICE kernels
# -----------------------------

spice.furnsh(str(KERNEL_DIR / "naif0012.tls"))
spice.furnsh(str(KERNEL_DIR / "pck00011.tpc"))
spice.furnsh(str(KERNEL_DIR / "de440.bsp"))


# -----------------------------
# Constants
# -----------------------------

SIGMA = 5.670374419e-8      # W/m^2/K^4
AU_KM = 149_597_870.7       # km

SOLAR_FLUX_1_AU = 1361.0    # W/m^2
LUNAR_ALBEDO = 0.12
LUNAR_EMISSIVITY = 0.95

TORR_TO_PA = 133.322

# -----------------------------
# Basic geometry helpers
# -----------------------------
 
def latlon_to_unit_vector(lat_deg, lon_deg):
    """
    Convert latitude/longitude to a Moon-fixed unit vector.
    Latitude: -90 to +90 deg
    Longitude: can be -180 to 180 or 0 to 360 deg
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
 
    return np.array([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat)
    ])
 
 
def get_sun_direction_moon_fixed(utc_time):
    """
    Get apparent Sun direction as seen from the Moon,
    expressed in the Moon-fixed IAU_MOON frame.
    """
    et = spice.utc2et(utc_time)
 
    sun_pos_km, light_time = spice.spkpos(
        "SUN",
        et,
        "IAU_MOON",
        "LT+S",
        "MOON"
    )
 
    sun_pos_km = np.array(sun_pos_km)
    sun_distance_km = np.linalg.norm(sun_pos_km)
    sun_direction = sun_pos_km / sun_distance_km
 
    return sun_direction, sun_distance_km
 
 
def get_subsolar_latlon(utc_time):
    """
    Get approximate subsolar latitude/longitude on the Moon.
    """
    sun_direction, sun_distance_km = get_sun_direction_moon_fixed(utc_time)
 
    x, y, z = sun_direction
 
    subsolar_lat = np.degrees(np.arcsin(z))
    subsolar_lon = np.degrees(np.arctan2(y, x)) % 360
 
    return subsolar_lat, subsolar_lon
 
 
def get_illumination(lat_deg, lon_deg, utc_time):
    """
    Determine whether the point is sunlit and compute solar zenith angle.
    """
    utc_string = utc_time.strftime("%Y-%m-%dT%H:%M:%S")
    surface_normal = latlon_to_unit_vector(lat_deg, lon_deg)
    sun_direction, sun_distance_km = get_sun_direction_moon_fixed(utc_string)
 
    cos_zenith = np.dot(surface_normal, sun_direction)
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
 
    sunlit = cos_zenith > 0
    solar_zenith_angle_deg = np.degrees(np.arccos(cos_zenith))
 
    return sunlit, cos_zenith, solar_zenith_angle_deg, sun_distance_km

# -----------------------------
# Solar Flux and Absorption
# -----------------------------
 
def solar_flux_at_moon(sun_distance_km):
    """
    Solar flux at the Moon using inverse-square scaling.
    """
    return SOLAR_FLUX_1_AU * (AU_KM / sun_distance_km) ** 2
 
 
def absorbed_solar_flux(cos_zenith, solar_flux):
    """
    Absorbed solar flux by the lunar surface.
    Includes solar zenith angle dependence.
    """
    illumination_factor = max(cos_zenith, 0.0)
 
    return (1.0 - LUNAR_ALBEDO) * solar_flux * illumination_factor
 
 
def vacuum_pressure(sunlit):
    """
    Approximate lunar surface/exosphere pressure in Pa.
    Dayside: ~1e-6 Pa, nightside: ~1e-10 Pa.
    Note: not currently used in the thermal ODE, but available for reference.
    """
    return 1e-6 if sunlit else 1e-10

# -----------------------------
# Getting Temperature Data from Diviner Maps
# -----------------------------
 
def rgb_image_to_temperature(
    rgb,
    cmap_name="nipy_spectral",
    temp_min_K=0.0,
    temp_max_K=450.0,
    n_colors=4096,
    white_threshold=245
):
    """
    Convert RGB image to approximate temperature using inverse colormap matching.
 
    rgb shape must be:
        rows x cols x 3
 
    Returns:
        2D temperature map in K
    """
 
    rgb = rgb.astype(np.uint8)
 
    white_mask = (
        (rgb[:, :, 0] > white_threshold) &
        (rgb[:, :, 1] > white_threshold) &
        (rgb[:, :, 2] > white_threshold)
    )
 
    rgb_norm = rgb.astype(float) / 255.0
 
    cmap = cm.get_cmap(cmap_name, n_colors)
    color_values = np.linspace(0, 1, n_colors)
    cmap_rgb = cmap(color_values)[:, :3]
 
    tree = cKDTree(cmap_rgb)
 
    rows, cols, _ = rgb_norm.shape
    flat_rgb = rgb_norm.reshape(-1, 3)
 
    distances, indices = tree.query(flat_rgb)
 
    normalized_temperature = color_values[indices]
 
    temp_flat = (
        temp_min_K
        + normalized_temperature * (temp_max_K - temp_min_K)
    )
 
    temperature_K = temp_flat.reshape(rows, cols)
 
    temperature_K[white_mask] = np.nan
 
    # Remove obviously nonphysical values
    temperature_K[temperature_K < 0] = np.nan
    temperature_K[temperature_K > 500] = np.nan
 
    return temperature_K
 
def get_snapshot_file(snapshot_lon_deg):
    """
    Build Diviner snapshot filename.
    """
    return DIVINER_DIR / f"diviner_tbol_snapshot_{snapshot_lon_deg:03d}E.tif"
 
 
@lru_cache(maxsize=32)
def load_diviner_snapshot(snapshot_lon_deg):
    """
    Load one RGB Diviner snapshot and convert it to approximate temperature.
    Cached for speed.
    """
    tif_path = get_snapshot_file(snapshot_lon_deg)
 
    if not tif_path.exists():
        raise FileNotFoundError(f"Could not find Diviner file: {tif_path}")
 
    with rasterio.open(tif_path) as src:
        rgb = src.read([1, 2, 3])
        bounds = src.bounds
        profile = src.profile
 
    # Convert from bands, rows, cols to rows, cols, bands
    rgb = np.transpose(rgb, (1, 2, 0))
 
    temperature_K = rgb_image_to_temperature(
        rgb,
        cmap_name="nipy_spectral",
        temp_min_K=0.0,
        temp_max_K=450.0
    )
 
    return temperature_K, bounds, profile
 
def get_temperature_at_point(lat_deg, lon_deg, temperature_K, bounds):
    """
    Get temperature from a Diviner temperature map at a given lat/lon.
 
    Handles both:
        -180 to 180 longitude maps
        0 to 360 longitude maps
    """
 
    n_rows, n_cols = temperature_K.shape
 
    # Handle longitude convention
    if bounds.left >= 0:
        lon_query = lon_deg % 360
    else:
        lon_query = ((lon_deg + 180) % 360) - 180
 
    lat_query = lat_deg
 
    if not (bounds.left <= lon_query <= bounds.right):
        return np.nan
 
    if not (bounds.bottom <= lat_query <= bounds.top):
        return np.nan
 
    col = int(
        (lon_query - bounds.left)
        / (bounds.right - bounds.left)
        * n_cols
    )
 
    row = int(
        (bounds.top - lat_query)
        / (bounds.top - bounds.bottom)
        * n_rows
    )
 
    row = np.clip(row, 0, n_rows - 1)
    col = np.clip(col, 0, n_cols - 1)
 
    return float(temperature_K[row, col])

# -----------------------------
# Diviner Temperature Interpolation for given lat/lon using SciPy's PCHIP interpolation
# Used this to keep the curve C1 smooth without the extra oscillation of cubic splines
#
# NOTE: The Diviner snapshots are keyed by subsolar longitude. The interpolation
# therefore assumes that surface temperature tracks primarily with subsolar longitude
# (i.e. local solar time). This is a good approximation at equatorial latitudes but
# becomes less accurate at high latitudes where insolation geometry differs significantly
# from the equatorial case.
# -----------------------------
 
DIVINER_SNAPSHOT_LONGITUDES = np.arange(0, 375, 15)  # 0, 15, ..., 360 (360 repeats 0 for wrap)
 
@lru_cache(maxsize=512)
def get_diviner_temperature_interp(
    lat_deg,
    lon_deg,
    points,
):
    """
    Build a PCHIP interpolator mapping subsolar longitude (0–360 deg) to
    surface temperature at the given lat/lon.
 
    The `points` argument is included so that different callers with different
    time-axis resolutions get separate cache entries. The returned time_vals
    array spans [0, 360] with `points` samples and is provided for convenience
    only — the interpolator itself is the primary return value.
    """
    temps = []
    for i in range(24):
        temperature_K, bounds, profile = load_diviner_snapshot(DIVINER_SNAPSHOT_LONGITUDES[i])
        T1 = get_temperature_at_point(lat_deg, lon_deg, temperature_K, bounds)
        temps.append(T1)
 
    # Append first value to close the periodic cycle
    temps.append(temps[0])
    temps = np.array(temps)
 
    time_vals = np.linspace(0, 360, points)
    temps_interp = PchipInterpolator(DIVINER_SNAPSHOT_LONGITUDES, temps)
 
    return temps_interp, time_vals
 
def get_diviner_temperature_evolution(
    lat_deg,
    lon_deg,
    start_time,
    duration_days,
    dt_hours=1
):
    temp_func, lat_vals = get_diviner_temperature_interp(lat_deg, lon_deg, points=360)
    total_hours = duration_days * 24
    n_steps = int(total_hours / dt_hours) + 1
 
    datetimes = []
    time_days = []
    subsolar_lons = []
    temperatures_K = []
 
    for i in range(n_steps):
        current_time = start_time + timedelta(hours=i * dt_hours)
        utc_string = current_time.strftime("%Y-%m-%dT%H:%M:%S")
 
        subsolar_lat, subsolar_lon = get_subsolar_latlon(utc_string)
 
        T = temp_func(subsolar_lon % 360)
 
        datetimes.append(current_time)
        time_days.append(i * dt_hours / 24)
        subsolar_lons.append(subsolar_lon)
        temperatures_K.append(float(T))
 
    temperatures_K = np.array(temperatures_K)
 
    return {
        "datetime": datetimes,
        "time_days": np.array(time_days),
        "subsolar_longitude_deg": np.array(subsolar_lons),
        "temperature_K": temperatures_K
    }, temperatures_K, time_days
    
    # -----------------------------
# Lunar surface conditions
# -----------------------------
 
def lunar_surface_conditions(lat_deg, lon_deg, utc_time, duration):
    """
    Return lunar surface environmental conditions at a given location and UTC time.
 
    Inputs:
        lat_deg: lunar latitude in degrees
        lon_deg: lunar longitude in degrees
        utc_time: datetime object
 
    Returns:
        dictionary of environmental conditions
    """
 
    sunlit, cos_zenith, sza_deg, sun_distance_km = get_illumination(
        lat_deg,
        lon_deg,
        utc_time
    )
 
    solar_flux = solar_flux_at_moon(sun_distance_km)
 
    illumination_factor = max(cos_zenith, 0.0)
 
    incident_flux = solar_flux * cos_zenith if sunlit else 0.0
    absorbed_flux = absorbed_solar_flux(cos_zenith, solar_flux)
 
    pressure_Pa = vacuum_pressure(sunlit)
 
 
    diviner_result, temps, days = get_diviner_temperature_evolution(
        lat_deg,
        lon_deg,
        utc_time,
        duration_days=duration,
        dt_hours=1
    )
 
    result = {
        "utc_time": utc_time,
        "latitude_deg": lat_deg,
        "longitude_deg": lon_deg,
 
        "sunlit": sunlit,
        "cos_solar_zenith": cos_zenith,
        "solar_zenith_angle_deg": sza_deg,
        "illumination_factor": illumination_factor,
 
        "sun_moon_distance_km": sun_distance_km,
        "solar_flux_at_moon_W_m2": solar_flux,
        "incident_solar_flux_W_m2": incident_flux,
        "absorbed_solar_flux_W_m2": absorbed_flux,
 
        "pressure_Pa": pressure_Pa,
    }
 
    result.update(diviner_result)
 
    return result

# -----------------------------
# Illumination simulation
# -----------------------------
 
def simulate_illumination_over_time(
    lat_deg,
    lon_deg,
    start_time,
    duration_days=30,
    dt_hours=1
):
    """
    Simulate lunar illumination factor over time.
 
    illumination_factor = max(cos(solar zenith angle), 0)
 
    1.0 = Sun directly overhead
    0.0 = Sun on horizon or below horizon
    """
 
    rows = []
 
    total_steps = int(duration_days * 24 / dt_hours)
 
    for step in range(total_steps + 1):
        current_time = start_time + timedelta(hours=step * dt_hours)
        utc_string = current_time.strftime("%Y-%m-%dT%H:%M:%S")
 
        sunlit, cos_zenith, sza_deg, sun_distance_km = get_illumination(
            lat_deg,
            lon_deg,
            utc_string
        )
 
        illumination_factor = max(cos_zenith, 0.0)
 
        rows.append({
            "utc_time": utc_string,
            "latitude_deg": lat_deg,
            "longitude_deg": lon_deg,
            "sunlit": sunlit,
            "cos_solar_zenith": cos_zenith,
            "solar_zenith_angle_deg": sza_deg,
            "illumination_factor": illumination_factor
        })
 
    return pd.DataFrame(rows)

# -----------------------------
# Multi-node cube temperature simulation
# -----------------------------
 
def simulate_multinode_cube_temperature(
    lat_deg,
    lon_deg,
    start_time,
    T0,
    power_gen,
    duration_days=30,
    dt_minutes=10,
    L_cube=1.00,
    alpha_solar=0.20,
    epsilon_IR=0.30,
    rho=2700.0,
    cp=900.0,
    k_body=205.0,          # W/(m K), aluminum ~205
    k_internal=5.0,        # effective coupling from electronics to cube core
    m_internal=1.0,        # kg, electronics mass
    cp_internal=900.0,     # J/(kg K)
    h_contact=0.0,
    include_lunar_albedo=True,
    method="RK45",
    surface_mass_fraction=0.20,
    bottom_ground_view=0.20,
    T_space=3.0,           # FIX #7: default is CMB ~3 K, not 30 K
    G_face_core_override=None
):
    """
    Multi-node thermal model for a cube on the lunar surface.
 
    Nodes:
        y[0] = T_top
        y[1] = T_bottom
        y[2] = T_north
        y[3] = T_south
        y[4] = T_east
        y[5] = T_west
        y[6] = T_core
        y[7] = T_internal
 
    Key physical assumptions:
    - Cube is fixed (no attitude dynamics).
    - Top face normal = local vertical = lunar surface normal.
    - View factors to space and ground sum to 1 for each face (no self-view
      between neighbouring faces). For a convex object this is exact; for a
      recessed or concave geometry it would require a full radiosity calculation.
    - Lunar albedo reflected flux uses a Lambertian scattering assumption.
      The Moon is non-Lambertian (Hapke model), which would increase backscatter
      at low phase angles. This is a standard first-order approximation.
    - Ground radiation exchange uses the small-object-over-large-surface limit:
        Q = epsilon_obj * A * F * sigma * (T_ground^4 - T_face^4)
      LUNAR_EMISSIVITY is not multiplied in because for eps_lunar ~ 0.95 the
      combined emissivity is dominated by epsilon_obj (see FIX #5).
    """
 
    # -------------------------
    # Geometry and thermal mass
    # -------------------------
 
    A_face = L_cube**2
    volume = L_cube**3
    mass_total = rho * volume
 
    if not (0.0 < surface_mass_fraction < 1.0):
        raise ValueError("surface_mass_fraction must be between 0 and 1.")
 
    if not (0.0 <= bottom_ground_view <= 1.0):
        raise ValueError("bottom_ground_view must be between 0 and 1.")
 
    m_surface_total = surface_mass_fraction * mass_total
    m_face = m_surface_total / 6.0
    m_core = mass_total - m_surface_total
 
    dx_face_core = L_cube / 2.0
 
    # Conductance from each face node to the core.
    # For solid aluminum this is very large, making the cube nearly isothermal.
    # Use G_face_core_override to represent internal insulation / contact resistance.
    if G_face_core_override is None:
        G_face_core = k_body * A_face / dx_face_core
    else:
        G_face_core = float(G_face_core_override)
 
    # Conductance from internal electronics node to cube core.
    G_internal = k_internal * A_face / dx_face_core
 
    A_contact = A_face
 
    # -------------------------
    # Time setup
    # -------------------------
 
    t_start = 0.0
    t_end = duration_days * 24.0 * 3600.0
    dt_seconds = dt_minutes * 60.0
 
    t_eval = np.arange(t_start, t_end + 0.5 * dt_seconds, dt_seconds)
    t_eval = t_eval[t_eval <= t_end]
 
    if t_eval[-1] < t_end:
        t_eval = np.append(t_eval, t_end)
 
    # -------------------------
    # Build Diviner temperature interpolation once
    # -------------------------
 
    surface_temp_interp, _ = get_diviner_temperature_interp(
        lat_deg,
        lon_deg,
        points=360
    )
 
    # -------------------------
    # Cube face normals
    # -------------------------
 
    face_names = [
        "top",
        "bottom",
        "north",
        "south",
        "east",
        "west"
    ]
 
    r_hat = latlon_to_unit_vector(lat_deg, lon_deg)
 
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
 
    east_hat = np.array([
        -np.sin(lon),
        np.cos(lon),
        0.0
    ])
 
    north_hat = np.array([
        -np.sin(lat) * np.cos(lon),
        -np.sin(lat) * np.sin(lon),
        np.cos(lat)
    ])
 
    up_hat = r_hat
 
    face_normals = {
        "top": up_hat,
        "bottom": -up_hat,
        "north": north_hat,
        "south": -north_hat,
        "east": east_hat,
        "west": -east_hat
    }
 
    # -------------------------
    # View factors
    # -------------------------
    # view_space[face] + view_ground[face] = 1.0 for all faces.
    # This assumes the cube is convex — no face-to-face self-view.
    # For a cube, the self-view factor between adjacent faces is non-zero
    # (~0.2 each), but for an isolated object in open terrain the sky/ground
    # split is the dominant effect. This is a standard simplification.
 
    bottom_space_view = 1.0 - bottom_ground_view
 
    view_space = {
        "top":    1.0,
        "bottom": bottom_space_view,
        "north":  0.5,
        "south":  0.5,
        "east":   0.5,
        "west":   0.5
    }
 
    view_ground = {
        "top":    0.0,
        "bottom": bottom_ground_view,
        "north":  0.5,
        "south":  0.5,
        "east":   0.5,
        "west":   0.5
    }
 
    # -------------------------
    # Environment function
    # -------------------------
 
    def environment_at_time(t):
        current_time = start_time + timedelta(seconds=float(t))
        utc_string = current_time.strftime("%Y-%m-%dT%H:%M:%S")
 
        sunlit, cos_zenith, sza_deg, sun_distance_km = get_illumination(
            lat_deg,
            lon_deg,
            current_time
        )
 
        solar_flux = solar_flux_at_moon(sun_distance_km)
 
        subsolar_lat, subsolar_lon = get_subsolar_latlon(utc_string)
 
        T_surface = float(surface_temp_interp(subsolar_lon % 360.0))
 
        sun_direction, _ = get_sun_direction_moon_fixed(utc_string)
 
        return {
            "datetime": current_time,
            "utc_time": utc_string,
            "sunlit": bool(sunlit),
            "cos_zenith": float(cos_zenith),
            "solar_zenith_angle_deg": float(sza_deg),
            "sun_distance_km": float(sun_distance_km),
            "solar_flux": float(solar_flux),
            "subsolar_longitude_deg": float(subsolar_lon),
            "surface_temperature_K": T_surface,
            "sun_direction": sun_direction
        }
 
    # -------------------------
    # Heat terms
    # -------------------------
 
    def heat_terms(t, temps):
        """
        Compute heat terms for all face nodes, core node, and internal node.
 
        temps keys:
            top, bottom, north, south, east, west, core, internal
        """
 
        env = environment_at_time(t)
 
        solar_flux = env["solar_flux"]
        T_surface = max(float(env["surface_temperature_K"]), 1.0)
        sun_direction = env["sun_direction"]
        cos_zenith = max(float(env["cos_zenith"]), 0.0)
 
        T_core = max(float(temps["core"]), 1.0)
        T_internal = max(float(temps["internal"]), 1.0)
 
        terms = {}
 
        # -------------------------
        # Face heat balances
        # -------------------------
 
        for face in face_names:
            T_face = max(float(temps[face]), 1.0)
 
            normal = face_normals[face]
 
            # Direct solar heating
            projection = max(float(np.dot(normal, sun_direction)), 0.0)
 
            Q_solar = alpha_solar * A_face * solar_flux * projection
 
            # Reflected sunlight from lunar surface
            # Lambertian approximation — see docstring note on Hapke scattering.
            if include_lunar_albedo:
                Q_albedo = (
                    alpha_solar
                    * A_face
                    * view_ground[face]
                    * LUNAR_ALBEDO
                    * solar_flux
                    * cos_zenith
                )
            else:
                Q_albedo = 0.0
 
            # Net radiation exchange with deep space.
            # Positive Q_rad_space means heat leaves the face.
            Q_rad_space = (
                epsilon_IR
                * A_face
                * view_space[face]
                * SIGMA
                * (T_face**4 - T_space**4)
            )
 
            # Small convex object over large flat surface limit:
            #   Q_net = epsilon_obj * A * F * sigma * (T_ground^4 - T_face^4)
            # LUNAR_EMISSIVITY is NOT multiplied in — for eps_lunar ~ 0.95 the
            # combined emissivity 1/(1/eps_obj + 1/eps_lunar - 1) ≈ eps_obj.
            # Multiplying both emissivities would underestimate the exchange.
            # Positive Q_rad_ground_net means heat enters the face from the ground.
            Q_rad_ground_net = (
                epsilon_IR
                * A_face
                * view_ground[face]
                * SIGMA
                * (T_surface**4 - T_face**4)
            )
 
            # Conduction between this face and cube core.
            # Positive means heat enters the face from the core.
            Q_cond_core = G_face_core * (T_core - T_face)
 
            # Contact conduction only for bottom face.
            # Positive means heat enters bottom face from regolith.
            if face == "bottom":
                Q_contact = h_contact * A_contact * (T_surface - T_face)
            else:
                Q_contact = 0.0
 
            # Net heat into this face.
            Q_net = (
                Q_solar
                + Q_albedo
                - Q_rad_space
                + Q_rad_ground_net
                + Q_cond_core
                + Q_contact
            )
 
            terms[face] = {
                "Q_solar_W": Q_solar,
                "Q_albedo_W": Q_albedo,
                "Q_rad_space_W": Q_rad_space,
                "Q_rad_ground_net_W": Q_rad_ground_net,
                "Q_cond_core_W": Q_cond_core,
                "Q_contact_W": Q_contact,
                "Q_net_W": Q_net,
                "projection": projection,
                "view_space": view_space[face],
                "view_ground": view_ground[face]
            }
 
        # -------------------------
        # Core heat balance
        # -------------------------
        # The core receives the opposite of the face conduction terms.
        # If heat flows from core to face (positive Q_cond_core for face),
        # the core loses the same amount.
 
        Q_core_from_faces = 0.0
 
        for face in face_names:
            T_face = max(float(temps[face]), 1.0)
            Q_core_from_faces += G_face_core * (T_face - T_core)
 
        # Positive means heat enters core from internal electronics.
        Q_core_from_internal = G_internal * (T_internal - T_core)
 
        Q_core_net = Q_core_from_faces + Q_core_from_internal
 
        terms["core"] = {
            "Q_from_faces_W": Q_core_from_faces,
            "Q_from_internal_W": Q_core_from_internal,
            "Q_net_W": Q_core_net
        }
 
        # -------------------------
        # Internal electronics heat balance
        # -------------------------
        # Power generation heats the internal node.
        # Q_internal_to_core > 0 means heat flows from internal node to core
        # (i.e. internal is hotter than core, which is the normal operating case).
 
        Q_internal_to_core = G_internal * (T_core - T_internal)
 
        Q_internal_net = power_gen + Q_internal_to_core
 
        terms["internal"] = {
            "Q_gen_W": power_gen,
            "Q_internal_to_core_W": Q_internal_to_core,
            "Q_net_W": Q_internal_net
        }
 
        terms["environment"] = env
        terms["T_core"] = T_core
 
        return terms
 
    # -------------------------
    # ODE system
    # -------------------------
 
    def ode(t, y):
        T_top      = y[0]
        T_bottom   = y[1]
        T_north    = y[2]
        T_south    = y[3]
        T_east     = y[4]
        T_west     = y[5]
        T_core     = y[6]
        T_internal = y[7]
 
        temps = {
            "top":      T_top,
            "bottom":   T_bottom,
            "north":    T_north,
            "south":    T_south,
            "east":     T_east,
            "west":     T_west,
            "core":     T_core,
            "internal": T_internal
        }
 
        terms = heat_terms(t, temps)
 
        dT_top_dt      = terms["top"]["Q_net_W"]      / (m_face * cp)
        dT_bottom_dt   = terms["bottom"]["Q_net_W"]   / (m_face * cp)
        dT_north_dt    = terms["north"]["Q_net_W"]    / (m_face * cp)
        dT_south_dt    = terms["south"]["Q_net_W"]    / (m_face * cp)
        dT_east_dt     = terms["east"]["Q_net_W"]     / (m_face * cp)
        dT_west_dt     = terms["west"]["Q_net_W"]     / (m_face * cp)
 
        dT_core_dt     = terms["core"]["Q_net_W"]     / (m_core * cp)
 
        dT_internal_dt = terms["internal"]["Q_net_W"] / (m_internal * cp_internal)
 
        return [
            dT_top_dt,
            dT_bottom_dt,
            dT_north_dt,
            dT_south_dt,
            dT_east_dt,
            dT_west_dt,
            dT_core_dt,
            dT_internal_dt
        ]
 
    # -------------------------
    # Solve ODE
    # -------------------------
 
    y0 = [
        T0,  # top
        T0,  # bottom
        T0,  # north
        T0,  # south
        T0,  # east
        T0,  # west
        T0,  # core
        T0   # internal
    ]
 
    sol = solve_ivp(
        fun=ode,
        t_span=(t_start, t_end),
        y0=y0,
        t_eval=t_eval,
        method=method,
        rtol=1e-6,
        atol=1e-8
    )
 
    # FIX #14: solver success check
    if not sol.success:
        print("Warning: solve_ivp did not finish successfully.")
        print(sol.message)
 
    # -------------------------
    # Build dataframe
    # Note: heat_terms is re-evaluated at each t_eval point to populate the
    # DataFrame. This is a reconstruction pass — values are post-hoc evaluations,
    # not the exact intermediate values the adaptive solver used internally.
    # -------------------------
 
    rows = []
 
    for i, t in enumerate(sol.t):
        temps = {
            "top":      sol.y[0, i],
            "bottom":   sol.y[1, i],
            "north":    sol.y[2, i],
            "south":    sol.y[3, i],
            "east":     sol.y[4, i],
            "west":     sol.y[5, i],
            "core":     sol.y[6, i],
            "internal": sol.y[7, i]
        }
 
        terms = heat_terms(t, temps)
        env = terms["environment"]
 
        row = {
            "time_s":    t,
            "time_hours": t / 3600.0,
            "time_days":  t / (24.0 * 3600.0),
            "datetime":  env["datetime"],
            "utc_time":  env["utc_time"],
 
            "T_top_K":      temps["top"],
            "T_bottom_K":   temps["bottom"],
            "T_north_K":    temps["north"],
            "T_south_K":    temps["south"],
            "T_east_K":     temps["east"],
            "T_west_K":     temps["west"],
            "T_core_K":     temps["core"],
            "T_internal_K": temps["internal"],
 
            "surface_temperature_K": env["surface_temperature_K"],
 
            "sunlit":                  env["sunlit"],
            "cos_solar_zenith":        env["cos_zenith"],
            "solar_zenith_angle_deg":  env["solar_zenith_angle_deg"],
            "solar_flux_W_m2":         env["solar_flux"],
 
            "mass_total_kg": mass_total,
            "m_face_kg":     m_face,
            "m_core_kg":     m_core,
            "m_internal_kg": m_internal,
 
            "L_cube_m":    L_cube,
            "A_face_m2":   A_face,
 
            "alpha_solar":  alpha_solar,
            "epsilon_IR":   epsilon_IR,
 
            "k_body_W_mK":     k_body,
            "G_face_core_W_K": G_face_core,
            "G_internal_W_K":  G_internal,
 
            "h_contact_W_m2K": h_contact,
            "power_gen_W":     power_gen,
 
            "surface_mass_fraction": surface_mass_fraction,
            "bottom_ground_view":    bottom_ground_view,
            "bottom_space_view":     bottom_space_view,
            "T_space_K":             T_space
        }
 
        for face in face_names:
            row[f"{face}_Q_solar_W"]          = terms[face]["Q_solar_W"]
            row[f"{face}_Q_albedo_W"]         = terms[face]["Q_albedo_W"]
            row[f"{face}_Q_rad_space_W"]      = terms[face]["Q_rad_space_W"]
            row[f"{face}_Q_rad_ground_net_W"] = terms[face]["Q_rad_ground_net_W"]
            row[f"{face}_Q_cond_core_W"]      = terms[face]["Q_cond_core_W"]
            row[f"{face}_Q_contact_W"]        = terms[face]["Q_contact_W"]
            row[f"{face}_Q_net_W"]            = terms[face]["Q_net_W"]
            row[f"{face}_solar_projection"]   = terms[face]["projection"]
            row[f"{face}_view_space"]         = terms[face]["view_space"]
            row[f"{face}_view_ground"]        = terms[face]["view_ground"]
 
        row["core_Q_from_faces_W"]    = terms["core"]["Q_from_faces_W"]
        row["core_Q_from_internal_W"] = terms["core"]["Q_from_internal_W"]
        row["core_Q_net_W"]           = terms["core"]["Q_net_W"]
 
        row["internal_Q_gen_W"]    = terms["internal"]["Q_gen_W"]
        row["internal_Q_to_core_W"] = terms["internal"]["Q_internal_to_core_W"]
        row["internal_Q_net_W"]    = terms["internal"]["Q_net_W"]
 
        rows.append(row)
 
    df = pd.DataFrame(rows)
 
    return sol, df

def get_float(prompt, default=None):
    while True:
        user_input = input(prompt)

        if user_input == "" and default is not None:
            return default

        try:
            return float(user_input)
        except ValueError:
            print("Please enter a valid number.")


def get_int(prompt, default=None):
    while True:
        user_input = input(prompt)

        if user_input == "" and default is not None:
            return default

        try:
            return int(user_input)
        except ValueError:
            print("Please enter a valid integer.")


def get_datetime(prompt):
    while True:
        user_input = input(prompt)

        try:
            return datetime.fromisoformat(user_input)
        except ValueError:
            print("Please enter the time in this format: YYYY-MM-DDTHH:MM:SS")


def main():
    print("Lunar Surface Object Temperature Simulation")
    print("------------------------------------------")

    # Location inputs
    lat = get_float("Enter latitude in degrees: ")
    lon = get_float("Enter longitude in degrees: ")

    # Time inputs
    start_time = get_datetime("Enter start time UTC, example 2026-06-01T12:00:00: ")
    duration_days = get_float("Enter simulation duration in days: ")
    dt_minutes = get_float("Enter time step in minutes [default 1]: ", default=1)

    # Material / object properties
    L_cube = get_float("Enter cube side length in meters: ")
    absorptivity = get_float("Enter absorptivity [0 to 1]: ")
    emissivity = get_float("Enter emissivity [0 to 1]: ")
    density_kg_m3 = get_float("Enter object density in kg/m^3: ")
    heat_capacity = get_float("Enter specific heat capacity in J/(kg K): ")
    # area_solar = get_float("Enter sun-facing area in m^2: ")
    # area_radiating = get_float("Enter radiating surface area in m^2: ")
    initial_temp_K = get_float("Enter initial object temperature in K: ")
    power_gen = get_float("Enter internal power generation in W (if any): ", default=0.0)
    thermal_conductivity = get_float("Enter thermal conductivity in W/(m K): ", default=205.0)
    

    print("\nRunning simulation...\n")

    # Example call to your existing simulation function
    sol, df_cube = simulate_multinode_cube_temperature(
        lat_deg=lat,
        lon_deg=lon,
        start_time=start_time,
        duration_days=duration_days,
        dt_minutes=dt_minutes,
        L_cube=L_cube,
        T0=initial_temp_K,
        power_gen=power_gen,
        alpha_solar=absorptivity,
        epsilon_IR=emissivity,
        rho=density_kg_m3,
        cp=heat_capacity,
        k_body=thermal_conductivity,
        k_internal=5.0,
        m_internal=1.0,
        cp_internal=heat_capacity,
        h_contact=0.001,
        surface_mass_fraction=0.2,
        bottom_ground_view=0.2,
        T_space=3.0,
        G_face_core_override=5.0
    )

    print("Simulation complete.")
    
    df_cube.head()
    
    # -----------------------------
    # Plot: face temperatures
    # -----------------------------
 
    plt.figure(figsize=(10, 5))
    
    plt.plot(df_cube["time_days"], df_cube["T_top_K"],      label="Top face")
    plt.plot(df_cube["time_days"], df_cube["T_bottom_K"],   label="Bottom face")
    plt.plot(df_cube["time_days"], df_cube["T_north_K"],    label="North face")
    plt.plot(df_cube["time_days"], df_cube["T_south_K"],    label="South face")
    plt.plot(df_cube["time_days"], df_cube["T_east_K"],     label="East face")
    plt.plot(df_cube["time_days"], df_cube["T_west_K"],     label="West face")
    plt.plot(df_cube["time_days"], df_cube["T_internal_K"], label="Internal electronics", linewidth=2)
    
    plt.plot(
        df_cube["time_days"],
        df_cube["surface_temperature_K"],
        label="Lunar surface",
        alpha=0.7,
        linestyle="--"
    )

    
if __name__ == "__main__":
    main()