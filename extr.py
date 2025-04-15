import numpy as np

# convert temperature from kelvin to degrees celcius
def conv_to_degreescelcius(data):
    data.t2m = data.t2m - 273.15

# calculate the 99th percentile of a list
def calc_percentile(a_list):
    threshold = np.percentile(a_list, 99)
    return threshold

# creates the list the 95th percentile should be calculated on and then calls calc_percentile
def calc_perc(lst):
    a_list = []
    for l in lst:
        for i in range(len(l)):
            a_list.append(l[i])
    return calc_percentile(a_list)