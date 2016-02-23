import csv
import sys
from matplotlib import pyplot
from mpl_toolkits.basemap import Basemap


COLORS = ['yo', 'go', 'ro', 'wo', 'ko', 'yo', 'mo', 'ro', 'yo', 'ko']
class GeoMap:
	
	def plot_points(self, data_points, color_provider, coord_mapper):
		"""
		Plots the list of data point("data_points") on geo map. 
		"color_provider" is the mapper function to map a data row to the corresponding color of the data point.
		"coord_mapper" is the mapper function to map a data row to the [latitude, langitude] of the data point.
		"""
		base_map = Basemap(projection='robin', lat_0=0, lon_0=0, resolution='l', area_thresh=1000.0)
		base_map.drawcoastlines()
		base_map.drawcountries()
		base_map.fillcontinents()
		for row in data_points:
			latitude, longitude = coord_mapper(row)
			x, y = base_map(longitude, latitude)
			base_map.plot(x, y, color_provider(row), markersize=4)
		pyplot.show()
		
if __name__ == "__main__":
	data_file = sys.argv[1]
	file_reader = csv.reader(open(data_file, "r"))
	GeoMap().plot_points(file_reader, lambda row: COLORS[int(row[1])], lambda row:[row[2], row[3]])
