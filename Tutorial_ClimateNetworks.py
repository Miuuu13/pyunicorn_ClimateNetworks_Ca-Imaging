

"""
https://www.pik-potsdam.de/~donges/pyunicorn/
"""


#Try analysis based on Climate Networks from pyunicorn

# Use luminescence traces 

# folder data > folder Batch_A > folder for individual

# e.g. Batch_A_2022_936_CFC_GPIO


""" Tutorial: Climate Networks """
#%%
# [1]
import numpy as np
from matplotlib import pyplot as plt
from pyunicorn import climate

#%%
# [2]
DATA_NAME = "air.mon.mean.nc"
DATA_URL = f"https://psl.noaa.gov/repository/entry/get/{DATA_NAME}?entryid=synth%3Ae570c8f9-ec09-4e89-93b4-babd5651e7a9%3AL25jZXAucmVhbmFseXNpcy5kZXJpdmVkL3N1cmZhY2UvYWlyLm1vbi5tZWFuLm5j"
DATA_FILE = f"./data/{DATA_NAME}"
# ![ -f {DATA_FILE} ] || wget -O {DATA_FILE} -nv --show-progress "{DATA_URL}"

# %%
# [3]
DATA_FILENAME = "./data/air.mon.mean.nc"
#  Indicate data source (optional)
DATA_SOURCE = "ncep_ncar_reanalysis"
#  Type of data file ("NetCDF" indicates a NetCDF file with data on a regular
#  lat-lon grid, "iNetCDF" allows for arbitrary grids - > see documentation).
FILE_TYPE = "NetCDF"
#  Name of observable in NetCDF file ("air" indicates surface air temperature
#  in NCEP/NCAR reanalysis data)
OBSERVABLE_NAME = "air"
#  Select a region in time and space from the data (here the whole dataset)
WINDOW = {"time_min": 0., "time_max": 0., "lat_min": 0, "lon_min": 0,
          "lat_max": 30, "lon_max": 0}
#  Indicate the length of the annual cycle in the data (e.g., 12 for monthly
#  data). This is used for calculating climatological anomaly values.
TIME_CYCLE = 12

#%%
# [4]
from pyunicorn import climate ###needed! not in tutorial


#  For setting fixed threshold
THRESHOLD = 0.5
#  For setting fixed link density
LINK_DENSITY = 0.005
#  Indicates whether to use only data from winter months (DJF) for calculating
#  correlations
WINTER_ONLY = False

data = climate.ClimateData.Load(
    file_name=DATA_FILENAME, observable_name=OBSERVABLE_NAME,
    data_source=DATA_SOURCE, file_type=FILE_TYPE,
    window=WINDOW, time_cycle=TIME_CYCLE)
print(data)


#%%
# [5]

net = climate.TsonisClimateNetwork(
    data, threshold=THRESHOLD, winter_only=WINTER_ONLY)


#%%
# [6]

#  Create a climate network based on Pearson correlation without lag and with
#  fixed link density
# net = climate.TsonisClimateNetwork(
#     data, link_density=LINK_DENSITY, winter_only=WINTER_ONLY)


#%%
# [7]

#  Create a climate network based on Spearman's rank order correlation without
#  lag and with fixed threshold
# net = climate.SpearmanClimateNetwork(
#     data, threshold=THRESHOLD, winter_only=WINTER_ONLY)

#%%
# [8]

#  Create a climate network based on mutual information without lag and with
#  fixed threshold
# net = climate.MutualInfoClimateNetwork(
#     data, threshold=THRESHOLD, winter_only=WINTER_ONLY)



#%%
# [9]

print("Link density:", net.link_density)

degree = net.degree()
closeness = net.closeness()
betweenness = net.betweenness()
clustering = net.local_clustering()
ald = net.average_link_distance()
mld = net.max_link_distance()

#  Save the grid (mainly vertex coordinates) to text files
#data.grid.save_txt(filename="grid.txt")
#  Save the degree sequence. Other measures may be saved similarly.
#np.savetxt("degree.txt", degree)


#%%
# [10]


import cartopy.crs as ccrs
### not in tutorial


# create a Cartopy plot instance called map_plot
# from the data with title DATA_SOURCE
map_plot = climate.MapPlot(data.grid, DATA_SOURCE)


#%%
# [11]

# plot degree
map_plot.plot(degree, "Degree")

# add matplotlib.pyplot or cartopy commands to customize figure
plt.set_cmap('plasma')
# optionally save figure
#plt.savefig('degree.png')

#%%
# [12]

# plot betwenness
map_plot.plot(np.log10(betweenness + 1), "Betweenness (log10)")

# add matplotlib.pyplot or cartopy commands to customize figure
plt.set_cmap('plasma')
# optionally save figure
#plt.savefig('degree.png')
#%%
