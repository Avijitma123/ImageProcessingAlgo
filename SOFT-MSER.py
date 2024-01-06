from PIL import Image
from numpy import asarray
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import cv2
import tensorflow.keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

# Function to find connected components and their areas
def find_connected_components(img, lower_threshold, upper_threshold):
    data = {}
    for j in range(lower_threshold, upper_threshold + 1):
        label_set = {}
        ret, thresh = cv2.threshold(img, j, 255, cv2.THRESH_BINARY_INV)
        connectivity = 8
        output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        for i in range(0, numLabels):
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]
            st = str(i) + "=" + str(int(cX)) + "," + str(int(cX))
            label_set[st] = area
        data[j] = label_set
    return data

# Function to compute the difference between two soft sets
def compute_difference(Soft1, Soft2, t1, t2):
    p = {}
    x = [v for k, v in Soft1.items()]
    y = [v for k, v in Soft2.items()]
    key1 = [(int(tp1[0]), int(tp1[1])) for k in Soft1.keys() for tp1 in k.split("=")[1].split(",")]
    key2 = [(int(tp2[0]), int(tp2[1])) for k in Soft2.keys() for tp2 in k.split("=")[1].split(",")]

    if len(x) > len(y):
        key1 = [dp for dp in key1 if dp in key2]
    elif len(x) < len(y):
        key2 = [dp for dp in key2 if dp in key1]

    x_e = [((int(tp1[0]), int(tp1[1])), int(v)) for k, v in Soft1.items() for tp1 in k.split("=")[1].split(",") if
           (int(tp1[0]), int(tp1[1])) in key1]
    y_e = [((int(tp2[0]), int(tp2[1])), int(v)) for k, v in Soft2.items() for tp2 in k.split("=")[1].split(",") if
           (int(tp2[0]), int(tp2[1])) in key2]

    Y_e = dict(y_e)
    prev = set(x_e)
    curr = set(y_e)

    s = str(t1) + "-" + str(t2)
    s_c = []
    for i in range(len(x_e)):
        q_x = x_e[i][0]
        size_q_x = x_e[i][1]
        if q_x in Y_e.keys():
            d = size_q_x - Y_e[q_x]
            print(s, d)
            s_c.append((q_x, d))

    centa[s] = s_c

# Function to find the count of zeros in a list
def count_zeros(lst):
    count = 0
    for ele in lst:
        if ele[1] == 0:
            count += 1
    return count

# Function to find the average of a list
def find_avg(lst):
    return int(sum(lst) / len(lst))

# Function to mark components on the image based on a threshold
def mark_components(image, th):
    ret, thresh = cv2.threshold(image, th, 255, cv2.THRESH_BINARY)
    connectivity = 8
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    output_image = image.copy()
    k = 255
    for i in range(1, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]
        if area > 50:
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (k, 0, 0), 3)
            k = k - 50
    return output_image

# Function to mark components in the inverse of an image based on a threshold
def mark_components_inv(image, th):
    inv_image = mark_components(image, th)
    ret, thresh = cv2.threshold(image, th, 255, cv2.THRESH_BINARY_INV)
    connectivity = 8
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    output_image = inv_image.copy()
    k = 255
    for i in range(1, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]
        if area > 50:
            inv_component.append([x, y, w, h, area])
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 3)
            k = k - 50

    if cv2.waitKey(0) & 0xff == 27:
        filename = "datafinal2" + ".jpg"
        cv2.imwrite(filename, output_image)
        cv2.destroyAllWindows()
    return output_image

# Load your ResNet50 model
model = tensorflow.keras.models.load_model('model_version_2.h5')

# Function to mark components based on additional conditions using a pre-trained model
def mark_components_with_conditions(image, th, model):
    ret, thresh = cv2.threshold(image, th, 255, cv2.THRESH_BINARY)
    connectivity = 8
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    output_image = image.copy()
    k = 255
    for i in range(1, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        im = image[y:y + h + 4, x:x + w + 4]
        resized = cv2.resize(im, (224, 224), interpolation=cv2.INTER_AREA)
        img_array = image.img_to_array(resized)
        img_batch = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_batch)
        p = model.predict(img_preprocessed)
        if area > 50 and w != d[1] and h != d[0] and p == 0:
            nor_component.append([x, y, w, h, area])
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 3)
            k = k - 50

    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
    return output_image

# Example usage
img1 = cv2.imread("Test6.jpeg")
img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
d = img.shape

# Example usage of find_connected_components
data = find_connected_components(img, 120, 200)
print(data)

# Example usage of compute_difference
t1 = 120
t2 = 121
compute_difference(data[t1], data[t2], t1, t2)

# Example usage of mark_components_inv
inv_component = []
output_inv = mark_components_inv(img, T)
print(inv_component)

# Example usage of mark_components_with_conditions
nor_component = []
output_conditions = mark_components_with_conditions(img, T, model)
print(nor_component)
