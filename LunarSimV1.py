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

# Solar Flux and Absorption

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
    """
    pressure_torr = 1e-10 if sunlit else 1e-12
    return pressure_torr * TORR_TO_PA

# Getting Temperature Data from Diviner Maps

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

# Diviner Temperature Interpolation for given lat/lon using SciPy's PCHIP interpolation
# Used this to keep the curve C1 smooth without the extra oscillation of cubic splines

DIVINER_SNAPSHOT_LONGITUDES = np.arange(0, 375, 15)  # 0, 15, ..., 345

@lru_cache(maxsize=512)
def get_diviner_temperature_interp(
    lat_deg,
    lon_deg,
    points,
):
    temps = []
    for i in range(24):
        temperature_K, bounds, profile = load_diviner_snapshot(DIVINER_SNAPSHOT_LONGITUDES[i])
        T1 = get_temperature_at_point(lat_deg, lon_deg, temperature_K, bounds)
        temps.append(T1)
    
    temps.append(temps[0])
    temps = np.array(temps)
    
    time_vals = np.linspace(0, 360, points)
    # temps_interp = np.interp(time_vals, DIVINER_SNAPSHOT_LONGITUDES, temps)
    # temps_interp = CubicSpline(DIVINER_SNAPSHOT_LONGITUDES, temps)
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
    
def lunar_surface_conditions(lat_deg, lon_deg, utc_time, duration):
    """
    Return lunar surface environmental conditions at a given location and UTC time.

    Inputs:
        lat_deg: lunar latitude in degrees
        lon_deg: lunar longitude in degrees
        utc_time: UTC time string, e.g. "2026-05-28T12:00:00"

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

def simulate_object_temperature(
    lat_deg,
    lon_deg,
    start_time,
    T0,
    duration_days=30,
    dt_minutes=10,
    L_cube=1.00,
    alpha_solar=0.20,
    epsilon_IR=0.10,
    rho=2700.0,
    cp=900.0,
    h_contact=0.0,
    include_lunar_albedo=True,
    method="RK45"
):
    A_face = L_cube**2
    A_total = 6.0 * A_face

    # Assumption: one face is effectively projected toward the Sun
    A_projected = A_face

    # Bottom face sees / contacts lunar ground
    A_ground = A_face
    A_contact = A_face

    volume = L_cube**3
    mass = rho * volume
    
    t_start = 0.0
    t_end = duration_days * 24.0 * 3600.0
    dt_seconds = dt_minutes * 60.0

    t_eval = np.arange(t_start, t_end + dt_seconds, dt_seconds)

    # -------------------------
    # Build Diviner temperature interpolation once
    # -------------------------
    surface_temp_interp, _ = get_diviner_temperature_interp(
        lat_deg,
        lon_deg,
        points=360
    )
    
    def environment_at_time(t):
        """
        Compute lunar environment at elapsed time t in seconds.
        """

        current_time = start_time + timedelta(seconds=float(t))
        utc_string = current_time.strftime("%Y-%m-%dT%H:%M:%S")

        sunlit, cos_zenith, sza_deg, sun_distance_km = get_illumination(
            lat_deg,
            lon_deg,
            current_time
        )

        solar_flux = solar_flux_at_moon(sun_distance_km)

        illumination_factor = max(float(cos_zenith), 0.0)

        subsolar_lat, subsolar_lon = get_subsolar_latlon(utc_string)

        T_surface = float(surface_temp_interp(subsolar_lon % 360.0))

        return {
            "datetime": current_time,
            "utc_time": utc_string,
            "sunlit": bool(sunlit),
            "cos_zenith": float(cos_zenith),
            "illumination_factor": illumination_factor,
            "solar_zenith_angle_deg": float(sza_deg),
            "sun_distance_km": float(sun_distance_km),
            "solar_flux": float(solar_flux),
            "subsolar_longitude_deg": float(subsolar_lon),
            "surface_temperature_K": T_surface,
        }

    def heat_terms(t, T_cube):
        """
        Compute all heat terms in W.
        Positive terms heat the cube.
        Negative terms cool the cube.
        """

        env = environment_at_time(t)

        T_surface = env["surface_temperature_K"]
        solar_flux = env["solar_flux"]
        illumination_factor = env["illumination_factor"]

        # 1. Direct solar heating
        Q_solar = (
            alpha_solar
            * A_projected
            * solar_flux
            * illumination_factor
        )

        # 2. Reflected solar radiation from lunar surface
        # This is a rough approximation using the bottom/ground-facing area.
        if include_lunar_albedo:
            Q_albedo = (
                alpha_solar
                * A_ground
                * LUNAR_ALBEDO
                * solar_flux
                * illumination_factor
            )
        else:
            Q_albedo = 0.0

        # 3. Lunar infrared radiation absorbed from warm lunar surface
        Q_lunar_IR = epsilon_IR* A_ground * LUNAR_EMISSIVITY * SIGMA * T_surface**4

        # 4. Thermal radiation emitted by cube
        # Use max(T_cube, 1.0) to avoid numerical issues if solver tries unphysical values.
        T_safe = max(float(T_cube), 1.0)

        Q_emit = epsilon_IR * SIGMA * A_total * T_safe**4

        # 5. Contact conduction with lunar surface
        Q_contact = h_contact * A_contact * (T_surface - T_safe)

        Q_net = Q_solar + Q_albedo + Q_lunar_IR + Q_contact - Q_emit

        return {
            "Q_solar_W": Q_solar,
            "Q_albedo_W": Q_albedo,
            "Q_lunar_IR_W": Q_lunar_IR,
            "Q_contact_W": Q_contact,
            "Q_emit_W": Q_emit,
            "Q_net_W": Q_net,
            **env
        }

    def ode(t, y):
        """
        solve_ivp right-hand side.

        y[0] = cube temperature in K
        """

        T_cube = y[0]

        terms = heat_terms(t, T_cube)

        dTdt = terms["Q_net_W"] / (mass * cp)

        return [dTdt]

    # -------------------------
    # Solve ODE
    # -------------------------
    sol = solve_ivp(
        fun=ode,
        t_span=(t_start, t_end),
        y0=[T0],
        t_eval=t_eval,
        method=method,
        rtol=1e-6,
        atol=1e-8
    )

    # -------------------------
    # Build output dataframe
    # -------------------------
    rows = []

    for t, T_cube in zip(sol.t, sol.y[0]):
        terms = heat_terms(t, T_cube)

        rows.append({
            "time_s": t,
            "time_hours": t / 3600.0,
            "time_days": t / (24.0 * 3600.0),
            "datetime": terms["datetime"],
            "utc_time": terms["utc_time"],

            "cube_temperature_K": T_cube,
            "cube_temperature_C": T_cube - 273.15,

            "surface_temperature_K": terms["surface_temperature_K"],
            "surface_temperature_C": terms["surface_temperature_K"] - 273.15,

            "sunlit": terms["sunlit"],
            "cos_solar_zenith": terms["cos_zenith"],
            "illumination_factor": terms["illumination_factor"],
            "solar_zenith_angle_deg": terms["solar_zenith_angle_deg"],
            "solar_flux_W_m2": terms["solar_flux"],

            "Q_solar_W": terms["Q_solar_W"],
            "Q_albedo_W": terms["Q_albedo_W"],
            "Q_lunar_IR_W": terms["Q_lunar_IR_W"],
            "Q_contact_W": terms["Q_contact_W"],
            "Q_emit_W": terms["Q_emit_W"],
            "Q_net_W": terms["Q_net_W"],

            "mass_kg": mass,
            "L_cube_m": L_cube,
            "A_total_m2": A_total,
            "A_projected_m2": A_projected,
            "A_ground_m2": A_ground,
            "alpha_solar": alpha_solar,
            "epsilon_IR": epsilon_IR,
            "h_contact_W_m2K": h_contact,
        })

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

    print("\nRunning simulation...\n")

    # Example call to your existing simulation function
    sol, df_cube = simulate_object_temperature(
        lat_deg=lat,
        lon_deg=lon,
        start_time=start_time,
        duration_days=duration_days,
        dt_minutes=dt_minutes,
        L_cube=L_cube,
        T0=initial_temp_K,
        alpha_solar=absorptivity,
        epsilon_IR=emissivity,
        rho=density_kg_m3,
        cp=heat_capacity,
        h_contact=0.0
    )

    print("Simulation complete.")
    
    df_cube.head()
    
    plt.figure(figsize=(10, 5))
    plt.plot(df_cube["time_days"], df_cube["cube_temperature_K"], label="Aluminum cube")
    plt.plot(df_cube["time_days"], df_cube["surface_temperature_K"], label="Lunar surface", alpha=0.7)
    plt.xlabel("Time since start [days]")
    plt.ylabel("Temperature [K]")
    plt.title("Aluminum Cube Temperature Evolution on Lunar Surface")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()