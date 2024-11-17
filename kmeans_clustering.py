import xarray as xr
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# Open the NetCDF file
nc_file = xr.open_dataset('texas_all_timesteps.nc')

# Select the data at the first time step
time_step = nc_file.time[100]
data_at_time_step = nc_file.sel(time=time_step)

# Extract the 512-dimensional vectors
vectors = data_at_time_step['latent_embedding'].values.reshape(-1, 512)

# Perform k-means clustering
num_clusters = 8  # Set the desired number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(vectors)

# Add the cluster labels to the DataFrame
data_at_time_step['cluster'] = xr.DataArray(clusters, dims=['point'])

# Plotting
plt.figure(figsize=(10, 8))

# Scatter plot with corrected 'c' parameter
scatter = plt.scatter(data_at_time_step.longitude, data_at_time_step.latitude, c=clusters, cmap='viridis')

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Cluster')

# Set axis labels
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Add a title
plt.title('K-Means Clustering of Latent Embeddings at Time Step {}'.format(time_step))

# Show the plot
plt.show()
