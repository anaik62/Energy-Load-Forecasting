import xarray as xr
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import numpy as np
import pandas as pd 
import geopandas as gpd
from shapely.geometry import Point



# Open the NetCDF file
nc_file = xr.open_dataset('texas_all_timesteps.nc')
print(nc_file)

lon_values = nc_file.lon.values
lat_values = nc_file.lat.values

# Create an array of tuples containing each pair of latitude and longitude points
coordinate_tuples = [(lat, lon) for lon in lon_values for lat in lat_values]



# Choose a specific time step (adjust the index as needed)
time_index = 100
time_step_data = nc_file.isel(time=time_index)


# Access the latitude, longitude, and latent_embedding values
lon_values = time_step_data['lon'].values
lat_values = time_step_data['lat'].values
latent_embedding_values = time_step_data['latent_embedding'].values


vectors = time_step_data['latent_embedding'].values.reshape(-1, 512)

# print("First few vectors:")
# print(vectors[:5, :])  # Print the first 5 rows (or adjust the number as needed)


# Perform hierarchical clustering
linkage_matrix = linkage(vectors, method='ward')
num_clusters = 8  # Set the desired number of clusters
clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
print(time_step_data['latent_embedding'].values)
print(clusters)

# Create a dictionary with (lon, lat) tuples as keys and corresponding cluster labels as values
coordinate_cluster_tuples = zip(coordinate_tuples, clusters)
cluster_dict = dict(coordinate_cluster_tuples)

# Display the dictionary
print(cluster_dict)

length = min(len(lon_values.ravel()), len(lat_values.ravel()), len(clusters))
lon_values = lon_values.ravel()[:length]
lat_values = lat_values.ravel()[:length]
clusters = clusters[:length]


# Flatten lon and lat arrays
lon_flat = time_step_data['lon'].values.ravel()
lat_flat = time_step_data['lat'].values.ravel()
# print("Length of lon_flat:", len(lon_flat))
# print("Length of lat_flat:", len(lat_flat))
# print("Length of clusters:", len(clusters))

# Assuming 'cluster_dict' is available
# You can adjust the names accordingly based on your actual variables

# Convert the cluster_dict to a list of (lon, lat, cluster) tuples
cluster_data = [(lon, lat, cluster) for (lon, lat), cluster in cluster_dict.items()]

# Create a DataFrame from the list
cluster_df = pd.DataFrame(cluster_data, columns=['Longitude', 'Latitude', 'Cluster'])

# Convert the DataFrame to a GeoDataFrame
geometry = [Point(lon, lat) for lon, lat in zip(cluster_df['Longitude'], cluster_df['Latitude'])]
geo_df = gpd.GeoDataFrame(cluster_df, geometry=geometry, crs="EPSG:4326")

# Plot the GeoDataFrame
fig, ax = plt.subplots(figsize=(10, 8))

# Specify aspect='equal' directly in the plot function
geo_df.plot(ax=ax, column='Cluster', legend=True, markersize=30, cmap='hsv', legend_kwds={'label': "Cluster"}, aspect='equal')

plt.title('Clustered Points in Texas')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()




# # Plot the dendrogram with distinct cluster colors
# plt.figure(figsize=(10, 6))
# dendrogram(linkage_matrix, labels=np.arange(1, len(vectors)+1), above_threshold_color='grey')
# plt.title('Dendrogram')
# plt.xlabel('Data Points')
# plt.ylabel('Distance')
# plt.show ()


# print("Lon_flat Size:", lon_flat.size)
# print("Lat_flat Size:", lat_flat.size)
# print("Clusters Size:", clusters.size)


# print("Lon Shape:", time_step_data['lon'].shape)        
# print("Lat Shape:", time_step_data['lat'].shape)

