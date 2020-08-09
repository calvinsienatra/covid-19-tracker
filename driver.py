import csv
import requests
import random
import math
import matplotlib.pyplot as plt
import copy
import numpy as np
from numpy.linalg import norm

# Download the latest COVID-19 dataset
def download_timeseries():
	r = requests.get("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv")

	f = open("dataset.csv", "w")
	f.write(r.text)
	f.close()

# Process and return the COVID-19 dataset
def request_timeseries():
	dataset = []

	with open("sample-dataset.csv", newline='') as csvfile: # Open the file
		reader = csv.DictReader(csvfile)

		for row in reader: # Clean and append the data
			del row["iso2"]
			del row["iso3"]
			del row["FIPS"]
			del row["Country_Region"]
			del row["Lat"]
			del row["Long_"]
			del row["code3"]

			dataset.append(row)

	return dataset

# Calculate the x, y coordinate for the particular time series dependant on the ten-hundred plot algorithm
def calculate_x_y(time_series):
	series = time_series.copy()

	del series["UID"]
	del series["Admin2"]
	del series["Province_State"]
	del series["Combined_Key"]

	series = list(series.values()) # Turn series into a list
	
	t = len(series) - 1 # Get the n

	if(int(series[t]) <= 0): # If n = 0, return NaN
		return [float("NaN"), float("NaN")]

	threshold10 = int(series[t])/10 # Get n/10
	x = float("NaN")
	xIndex = -1
	for i in range(len(series)): # Find the x
		if(int(series[i]) <= threshold10 ):
			x = t - i
			xIndex = i

	threshold100 = int(series[t])/100 # Get the n/100
	y = float("NaN")
	yIndex = -1
	for j in range(len(series)): # Find the y
		if(int(series[j]) <= threshold100):
			y = xIndex - j
			yIndex = j

	return (x, y)

# Calculate the euclidean distance between two x,y
def get_distance(set1, set2):
	x1 = set1[0]
	y1 = set1[1]

	x2 = set2[0]
	y2 = set2[1]

	xDiff = x1 - x2
	yDiff = y1 - y2

	xDiffPow = pow(xDiff, 2)
	yDiffPow = pow(yDiff, 2)

	xYTotal = xDiffPow + yDiffPow

	return math.sqrt(xYTotal) # Return the sqrt

# Process dict before k_means ("Combined_key": [data])
def dict_process(full_time_series):
	processed = {}

	for i in full_time_series:
		processed[i['Combined_Key']] = calculate_x_y(i)

	return processed

# Calculate the new center of a cluster
def get_new_center(cluster_set):
	new_x = -9999
	new_y = -9999

	cluster_set_length = len(cluster_set)
	if(cluster_set_length == 0):
		return None

	total_x = 0
	total_y = 0

	for i in cluster_set:
		total_x += cluster_set[i][0]
		total_y += cluster_set[i][1]

	new_x = total_x / cluster_set_length
	new_y = total_y / cluster_set_length

	return (new_x, new_y)

# Calculate the distance between a point to a line
def get_distance_to_line(p1_line, p2_line, p3):
	p1_line = np.asarray(p1_line)
	p2_line = np.asarray(p2_line)
	p3 = np.asarray(p3)
	d = norm(np.cross(p2_line - p1_line, p1_line - p3)) / norm(p2_line - p1_line)

	return d

# Gather points that are within t distance of the line
def get_close_to_line(x, y, dataset, t):
	clean_dict = {}
	close_dict = {}
	for county in dataset:
		if(not math.isnan(dataset[county][0])):
			d = get_distance_to_line(x, y, dataset[county])
			if(d <= t):
				close_dict[county] = dataset[county]
			else:
				clean_dict[county] = dataset[county]

	return (close_dict, clean_dict)

# Calculate and return three equally separated points from the line
def get_three_points_from_line(xy1, xy2):
	xy_final_1 = [0,0]
	xy_final_2 = [0,0]
	xy_final_3 = [0,0]

	xy_final_1[0] = xy1[0] + (xy2[0] - xy1[0]) * 0.1
	xy_final_1[1] = xy1[1] + (xy2[1] - xy1[1]) * 0.1

	xy_final_2[0] = xy1[0] + (xy2[0] - xy1[0]) * 0.5
	xy_final_2[1] = xy1[1] + (xy2[1] - xy1[1]) * 0.5

	xy_final_3[0] = xy1[0] + (xy2[0] - xy1[0]) * 0.8
	xy_final_3[1] = xy1[1] + (xy2[1] - xy1[1]) * 0.8

	return (xy_final_1, xy_final_2, xy_final_3)

# Run the k-means algorithm
def k_means_custom(k, dataset, centers): # Dataset contains county names
	if(len(centers) != k):
		raise ValueError("Number of centers should be the same as k")

	# Delete any nans
	clean_dataset = copy.deepcopy(dataset)
	for row in dataset:
		if(math.isnan(dataset[row][0])):
			del clean_dataset[row]


	# Initialize clusters with random unique initial center positions
	k_classified = {}
	tmp_classified = {}
	prev_k_classified = {}

	for i in range(k):
		k_classified["cluster_" + str(i+1)] = {'center': centers[i], 'points': {}}
		prev_k_classified["cluster_" + str(i+1)] = {'center': centers[i], 'points': {}}
		tmp_classified["cluster_" + str(i+1)] = 0

	iteration = 0
	while True:
		iteration += 1
		for county in clean_dataset:
			point_coord = clean_dataset[county]
			for cluster in tmp_classified: # Have to replace min calculations to make it more efficient
				tmp_classified[cluster] = get_distance(point_coord, k_classified[cluster]["center"])

			min_val = min(tmp_classified.values()) # Replace with in for-loop minimum finder

			cluster_name = ""
			for cluster in tmp_classified:
				if tmp_classified[cluster] == min_val:
					cluster_name = cluster
				tmp_classified[cluster] = 0

			k_classified[cluster_name]['points'][county] = clean_dataset[county]

		for cluster in k_classified:
			new_center = get_new_center(k_classified[cluster]['points'])
			if(new_center != None):
				k_classified[cluster]['center'] = new_center

		if(prev_k_classified == k_classified):
			break
		else:
			prev_k_classified = copy.deepcopy(k_classified)

	return k_classified

# Clean up the dataset from any NaNs
def clean_process_data(data):
	# Read the timeseries dataset
	data = request_timeseries()

	# Process the dataset
	processed_data = dict_process(data)

	# List all processed data values
	xy = list(processed_data.values())

	# Clean up any NaN values in the dataset
	x = list(i[0] for i in xy)
	clean_x = [i for i in x if not math.isnan(i)]
	y = list(i[1] for i in xy)
	clean_y = [i for i in y if not math.isnan(i)]

	return (clean_x, clean_y, processed_data)


# Clustering functions

# Run k-means for Category A
def calculate_category_a(clean_x, clean_y, pd):
	# Get each of the maximum values
	xmax = max(clean_x)
	ymax = max(clean_y)

	# Get three equally separated points in the imaginary line
	(xy1, xy2, xy3) = get_three_points_from_line((0,0),(xmax,ymax))

	# Get points within distance of 10 from the line and the other points
	(closest, clean) = get_close_to_line((0,0), (xmax, ymax), pd, 10)

	# Process closest values
	closest_val = list(closest.values())
	close_x = list(i[0] for i in closest_val)
	close_y = list(i[1] for i in closest_val)

	# Run k-means
	k_classified = k_means_custom(3, closest, (xy1, xy2, xy3))

	# List clusters with only points
	clus_1 = list(k_classified['cluster_1']['points'].values())
	clus_2 = list(k_classified['cluster_2']['points'].values())
	clus_3 = list(k_classified['cluster_3']['points'].values())

	clus_a_name = [k_classified['cluster_1']['points'], k_classified['cluster_2']['points'], k_classified['cluster_3']['points']]

	return (clus_1, clus_2, clus_3, clus_a_name, clean, xmax, ymax)

# Run k-means for Category B
def calculate_category_b(clean_x, clean_y, clean, xmax, ymax):
	# Get three equally separated points in the imaginary line
	(xy1, xy2, xy3) = get_three_points_from_line((0,0),(xmax,ymax))

	# Get points within distance of 10 from the line and the other points
	(closest, clean) = get_close_to_line((0,0), (xmax, ymax), clean, 30)

	# Process closest values
	closest_val = list(closest.values())
	close_x = list(i[0] for i in closest_val)
	close_y = list(i[1] for i in closest_val)

	# Calculate new 6 different points for category B
	xy1_2 = xy1.copy()
	xy2_2 = xy2.copy()
	xy3_2 = xy3.copy()
	xy4_2 = xy1.copy()
	xy5_2 = xy2.copy()
	xy6_2 = xy3.copy()

	slope = xmax/ymax

	xy1_2[0] += slope * 24
	xy2_2[0] += slope * 24
	xy3_2[0] += slope * 24
	xy4_2[0] -= slope * 5
	xy4_2[1] += slope * 24
	xy5_2[0] -= slope * 15
	xy5_2[1] += slope * 24
	xy6_2[0] -= slope * 15
	xy6_2[1] += slope * 24

	# Run k-means
	k_classified = k_means_custom(6, closest, (xy1_2, xy2_2, xy3_2, xy4_2, xy5_2, xy6_2))

	# List clusters with only points
	clus_1 = list(k_classified['cluster_1']['points'].values())
	clus_2 = list(k_classified['cluster_2']['points'].values())
	clus_3 = list(k_classified['cluster_3']['points'].values())
	clus_4 = list(k_classified['cluster_4']['points'].values())
	clus_5 = list(k_classified['cluster_5']['points'].values())
	clus_6 = list(k_classified['cluster_6']['points'].values())

	clus_b_name = [k_classified['cluster_1']['points'], k_classified['cluster_2']['points'], k_classified['cluster_3']['points'], k_classified['cluster_4']['points'], k_classified['cluster_5']['points'], k_classified['cluster_6']['points']]

	return (clus_1, clus_2, clus_3, clus_4, clus_5, clus_6, clus_b_name, clean)

# Run k-means for Category C
def calculate_category_c(clean_x, clean_y, clean, xmax, ymax):
	# Get three equally separated points in the imaginary line
	(xy1, xy2, xy3) = get_three_points_from_line((0,0),(xmax,ymax))

	# Get points within distance of 10 from the line and the other points
	(closest, clean) = get_close_to_line((0,0), (xmax, ymax), clean, 1000)

	# Process closest values
	closest_val = list(closest.values())
	close_x = list(i[0] for i in closest_val)
	close_y = list(i[1] for i in closest_val)

	# Calculate new 6 different points for category C
	xy1_3 = xy2.copy()
	xy2_3 = xy2.copy()

	slope = xmax/ymax

	xy1_3[0] += slope * 40
	xy1_3[1] -= slope * 8
	xy2_3[0] -= slope * 40
	xy2_3[1] += slope * 8

	# Run k-means
	k_classified = k_means_custom(2, closest, (xy1_3, xy2_3))

	# List clusters with only points
	clus_1 = list(k_classified['cluster_1']['points'].values())
	clus_2 = list(k_classified['cluster_2']['points'].values())

	clus_c_name = [k_classified['cluster_1']['points'], k_classified['cluster_2']['points']]

	return (clus_1, clus_2, clus_c_name, clean)

# Plot all clusters
def plot_clusters(clus_a_1, clus_a_2, clus_a_3, clus_b_1, clus_b_2, clus_b_3, clus_b_4, clus_b_5, clus_b_6, clus_c_1, clus_c_2):
	# List cluster points
	clus_a_1_x = list(i[0] for i in clus_a_1)
	clus_a_1_y = list(i[1] for i in clus_a_1)
	clus_a_2_x = list(i[0] for i in clus_a_2)
	clus_a_2_y = list(i[1] for i in clus_a_2)
	clus_a_3_x = list(i[0] for i in clus_a_3)
	clus_a_3_y = list(i[1] for i in clus_a_3)

	clus_b_1_x = list(i[0] for i in clus_b_1)
	clus_b_1_y = list(i[1] for i in clus_b_1)
	clus_b_2_x = list(i[0] for i in clus_b_2)
	clus_b_2_y = list(i[1] for i in clus_b_2)
	clus_b_3_x = list(i[0] for i in clus_b_3)
	clus_b_3_y = list(i[1] for i in clus_b_3)
	clus_b_4_x = list(i[0] for i in clus_b_4)
	clus_b_4_y = list(i[1] for i in clus_b_4)
	clus_b_5_x = list(i[0] for i in clus_b_5)
	clus_b_5_y = list(i[1] for i in clus_b_5)
	clus_b_6_x = list(i[0] for i in clus_b_6)
	clus_b_6_y = list(i[1] for i in clus_b_6)

	clus_c_1_x = list(i[0] for i in clus_c_1)
	clus_c_1_y = list(i[1] for i in clus_c_1)
	clus_c_2_x = list(i[0] for i in clus_c_2)
	clus_c_2_y = list(i[1] for i in clus_c_2)

	f, ax = plt.subplots()

	# Exponential, growth is rising fast
	ax.scatter(clus_a_1_x, clus_a_1_y, color="red") 

	# Exponential, growth is rising
	ax.scatter(clus_a_2_x, clus_a_2_y, color="orange")

	# Exponential, growth is rising slowly but consistently
	ax.scatter(clus_a_3_x, clus_a_3_y, color="pink")

	# Sub exponent, growth is slowing down
	ax.scatter(clus_b_1_x, clus_b_1_y, color="yellow")

	# Sub exponent, growth is slowing down very much
	ax.scatter(clus_b_2_x, clus_b_2_y, color="black")

	ax.scatter(clus_b_3_x, clus_b_3_y, color="green")
	ax.scatter(clus_b_4_x, clus_b_4_y, color="magenta")

	# Super exponent, growth in the recent days (around 40 days) has grown exponentially high
	ax.scatter(clus_b_5_x, clus_b_5_y, color="cyan")

	# Super Exponent, growth is a little more exponential (dangerous)
	ax.scatter(clus_b_6_x, clus_b_6_y, color="gray")

	# Sub exponent, growth has slowed down within at least the last 45 days
	ax.scatter(clus_c_1_x, clus_c_1_y, color="red", marker="x")

	# Super Exponent, growth very high in the recent days (could be an artifact) (very dangerous)
	ax.scatter(clus_c_2_x, clus_c_2_y, color="orange", marker="x")

	plt.show()

# Return final k-means with county names
def get_final_kmeans():
	(clean_x, clean_y, pd) = clean_process_data(request_timeseries())

	(clus_a_1, clus_a_2, clus_a_3, clus_a_name, clean, xmax, ymax) = calculate_category_a(clean_x, clean_y, pd)
	(clus_b_1, clus_b_2, clus_b_3, clus_b_4, clus_b_5, clus_b_6, clus_b_name, clean) = calculate_category_b(list(i[0] for i in clean), list(i[1] for i in clean), clean, xmax, ymax)
	(clus_c_1, clus_c_2, clus_c_name, clean) = calculate_category_c(list(i[0] for i in clean), list(i[1] for i in clean), clean, xmax, ymax)

	return {"clus_a_1": clus_a_name[0], "clus_a_2": clus_a_name[1], "clus_a_3": clus_a_name[2], "clus_b_1": clus_b_name[0], "clus_b_2": clus_b_name[1], "clus_b_3": clus_b_name[2], "clus_b_4": clus_b_name[3], "clus_b_5": clus_b_name[4], "clus_b_6": clus_b_name[5], "clus_c_1": clus_c_name[0], "clus_c_2": clus_c_name[1]}


# Main Driver
if __name__ == '__main__':

	(clean_x, clean_y, pd) = clean_process_data(request_timeseries())

	(clus_a_1, clus_a_2, clus_a_3, clus_a_name, clean, xmax, ymax) = calculate_category_a(clean_x, clean_y, pd)
	(clus_b_1, clus_b_2, clus_b_3, clus_b_4, clus_b_5, clus_b_6, clus_b_name, clean) = calculate_category_b(list(i[0] for i in clean), list(i[1] for i in clean), clean, xmax, ymax)
	(clus_c_1, clus_c_2, clus_c_name, clean) = calculate_category_c(list(i[0] for i in clean), list(i[1] for i in clean), clean, xmax, ymax)

	plot_clusters(clus_a_1, clus_a_2, clus_a_3, clus_b_1, clus_b_2, clus_b_3, clus_b_4, clus_b_5, clus_b_6, clus_c_1, clus_c_2)
