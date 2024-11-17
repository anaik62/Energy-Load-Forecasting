import xarray as xr
import matplotlib.pyplot as plt

nc_file = xr.open_dataset('weather_data_0.nc')
print(nc_file)
print(nc_file["2m_temperature"].isel(time=0))


temps = nc_file["2m_temperature"].isel(time=0)

temps.plot(cmap = 'RdBu_r')
plt.title("2m Temperature at First Time Step")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.xticks(range(0, 360, 40)) 
plt.yticks(range(-90, 91, 20)) 


plt.show() 