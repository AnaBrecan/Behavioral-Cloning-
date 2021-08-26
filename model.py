import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np
import csv
import cv2


lines = []
with open('/opt/carnd_p3/data/driving_log.csv') as csvfile: 
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)
        
print(lines[0]) 
print(lines[1])
center_images = []
left_images = []
right_images = []
measurements = []

for line in lines:
    source_path_center = line[0]
    source_path_left = line[1]
    source_path_right = line[2]
    
    #file_path_center = source_path_center.split('/')[-1]
    #file_path_left = source_path_left.split('/')[-1]
    #file_path_right = source_path_right.split('/')[-1]
    
    #print(file_path_center)
    #break;
    
    current_path_center = 'opt/carnd_p3/data/'+source_path_center
    print(current_path_center)
    break;
    current_path_left = 'opt/carnd_p3/data/'+source_path_left
    current_path_right = 'opt/carnd_p3/data/'+source_path_right
    
    image_center = cv2.imread(current_path_center)
    image_left = cv2.imread(current_path_left)
    image_right = cv2.imread(current_path_right)
    
    image_center_rgb = cv2.cvtColor(image_center, cv2.COLOR_BGR2RGB)
    image_left_rgb = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)
    image_right_rgb = cv2.cvtColor(image_right, cv2.COLOR_BGR2RGB)
    
    center_images.append(image_center_rgb)
    left_images.append(image_left_rgb)
    right_images.append(image_right_rgb)
    
    measurement = float(line[3])
    measurements.append(measurement)
    
    
X_train_center = np.array(center_images)
X_train_left = np.array(left_images)
X_train_right = np.array(right_images)
y_train = np.array(measurements)

print(type(X_train_center[0]))
print(X_train_center[0].size)

