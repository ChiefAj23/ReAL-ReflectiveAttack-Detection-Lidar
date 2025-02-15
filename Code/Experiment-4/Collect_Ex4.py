import csv
from rplidar import RPLidar, RPLidarException
import os
import math
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

# Parameters
PORT_NAME = '/dev/ttyUSB0'
BAUDRATE = 115200
TIMEOUT = 1
OUTPUT_FILE = 'tissue_n_battery_n.csv'
SPECULAR1 = False
SPECULAR2 = False

def calc_width_polar(min_point, max_point):
    min_angle, min_dist = math.radians(min_point[0]), min_point[1]
    max_angle, max_dist = math.radians(max_point[0]), max_point[1]
    print(max_angle, min_angle)
    result = math.sqrt(((min_dist)**2) + ((max_dist)**2) - (2*min_dist*max_dist*math.cos(max_angle-min_angle)))
    return result

def run_lidar(port, baudrate, timeout, output_file, spec1, spec2):
    lidar = RPLidar(port, baudrate, timeout)
    write_out = True
    try:
        lidar.connect()
        lidar.start_motor()
        info = lidar.get_info()
        health = lidar.get_health()
        if health[0] != 'Good':
            raise RPLidarException(f"lidar health is not good: {health}")

        data = []
        lines = 10000
        #lines = 100
        i = 0
        for scan in lidar.iter_scans(max_buf_meas=500, min_len=5):
            for quality, angle, distance in scan:
                if (distance > 50 and distance < 350):
                    data.append((angle, distance))
                    #print(f"quality: {quality}, angle: {angle}, distance: {distance}")

            # Stop collecting data after "repeat" iterations
            #print(f'{i=}')
            #i += 1
            #if i >= repeat:
            #    break

            # Stop collecting data after "lines" data points
            print(f'{len(data)=}')
            if len(data) >= lines:
                break

	# DBSCAN
        data.sort()
        distances = [i[1] for i in data]
        angles = [i[0] for i in data]
        x_coords = (distances * np.cos(np.deg2rad(angles)))
        y_coords = (distances * np.sin(np.deg2rad(angles)))
        coords = np.column_stack((x_coords, y_coords))
        polar = np.column_stack((angles, distances))
        db = DBSCAN(eps=50, min_samples=1)
        labels = db.fit_predict(coords)
        unique_labels = set(labels)
        data_c = []
        del data
        del distances
        del angles
        for i, label in enumerate(unique_labels):
            if label == -1:
                continue
            data_c.append([])
            mask = (labels == label)
            xy = coords[mask]
            for pair in xy:
                distance = round(math.sqrt(pair[0]**2 + pair[1]**2), 2)
                angle = round(np.rad2deg(math.atan2(pair[1], pair[0])), 5)
                angle = angle if angle > 0 else angle + 360
                data_c[i].append((angle, distance))
        if write_out:
            with open(output_file, 'a+', newline='') as csvfile:
                fieldnames = ['angle1', 'distance1', 'object1', \
                              'angle2', 'distance2', 'object2']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                i = 0
                while len(data_c[0]) != 0 and len(data_c[1]) != 0 and i != 25000:
                    angle1, distance1 = data_c[0].pop()
                    angle2, distance2 = data_c[1].pop()
                    writer.writerow({'angle1': angle1, 'distance1': distance1, 'object1': 1 if spec1 else 0, \
                                     'angle2': angle2, 'distance2': distance2, 'object2': 1 if spec2 else 0})
                    i += 1
    except RPLidarException as e:
        print(f"RPLidar exception: {e}")
    finally:
        lidar.stop_motor()
        lidar.disconnect()

def graph_cartesian(x_coords, y_coords, labels):
        plt.scatter(x_coords, y_coords, c=labels, cmap='viridis')
        plt.xlabel("X (mm)")
        plt.ylabel("Y (mm)")

def graph_polar(angles, distances, labels):
        plt.scatter(angles, distances, c=labels, cmap='viridis')
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Distance (mm)")

if __name__ == '__main__':
    assert os.path.exists(PORT_NAME), f'Device {PORT_NAME} not found.'
    run_lidar(PORT_NAME, BAUDRATE, TIMEOUT, OUTPUT_FILE, SPECULAR1, SPECULAR2)
    #print(f"Data saved to {OUTPUT_FILE}")
