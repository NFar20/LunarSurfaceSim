# Lunar Surface Temperature Simulation
Simulates how the temperature of an object evolves over time at a specific latitude and longitude on the surface of the moon.
Users provide location, start time, duration, and material properties to simulate how temperature will change throughout the experiment.
This project uses data from the Diviner Lunar Radiometer Experiment to determine surface temperature at any given time.
Solar illumination is also determined based on the solar zenith angle at each given time.
This information, along with the material properties, is used to write an ODE, which is then solved to determine the temperature at each given time.

# Usage
Using this program requires the additional download of SPICE kernels and the Diviner temperature data (files were too large to upload to the repository). These files can be found in this [Google Drive folder](https://drive.google.com/drive/folders/1FCagEN8Y4_OmcxCDl7Fshm0vRy1-ySBz?usp=sharing).
When installing, make sure the "data" and "kernels" folders are in the same directory as "LunarSimV1.py" and that all required Python libraries are installed.
