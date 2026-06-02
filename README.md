# Lunar Surface Temperature Simulation
Simulates how the temperature of an object evolves over time at a specific latitude and longitude on the surface of the moon.
Users provide location, start time, duration, and material properties to simulate how temperature will change throughout the experiment.
This project uses data from the Diviner Lunar Radiometer Experiment to determine surface temperature at any given time.
Solar illumination is also determined based on the solar zenith angle at each given time.
This information, along with the material properties, is used to write an ODE, which is then solved to determine the temperature at each given time.
