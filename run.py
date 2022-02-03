import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import cv2
import scipy.io as sio
import random
from tensorflow import keras
from tensorflow.keras import regularizers
tf.get_logger().setLevel('ERROR')

# load digit and non-digit detection model
MODEL_DIR = "./"
model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'digit_detection_model.h5'))

# load digit classification model
cnn_recognition_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'digit_classification_model.h5'))

# set the input image folder
IMAGE_DIR = "input_images"
# set the output image folder
OUTPUT_DIR = "graded_images"

# there are two many boxes in the image
# need to reduce the boxes number
# according to Piazza discussion https://piazza.com/class/kjliq19wrwi2rj?cid=542_f37
# here we choose to use NMS to reduce the boxes number

def nms_reduce_box_number(mser_boxes, threshold_number):

    # nms reference: https://zhuanlan.zhihu.com/p/54709759
    # first calculate each box's area
    ROI_area = np.zeros(len(mser_boxes)).astype(np.int)
    ROI_bottom_right_corner_x = np.zeros(len(mser_boxes)).astype(np.int)
    ROI_bottom_right_corner_y = np.zeros(len(mser_boxes)).astype(np.int)
    ROI_top_left_corner_x = np.zeros(len(mser_boxes)).astype(np.int)
    ROI_top_left_corner_y = np.zeros(len(mser_boxes)).astype(np.int)
    for i in range(len(ROI_boxes)):
        ROI_top_left_corner_x[i] = mser_boxes[i][0]
        ROI_top_left_corner_y[i] = mser_boxes[i][1]
        ROI_bottom_right_corner_x[i] = mser_boxes[i][2]
        ROI_bottom_right_corner_y[i] = mser_boxes[i][3]
        ROI_area[i] = (mser_boxes[i][2] - mser_boxes[i][0]) * (mser_boxes[i][3] - mser_boxes[i][1])

    # next step is to calculate the intersection over union for each pair of boxes
    # according to the reference shown below, we should use the bottom-right corner y coordinate to sort the boxes
    # reference: https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/

    # sort the boxes according to their bottom-right corner y coordinate
    sort_order = ROI_bottom_right_corner_y.argsort()[::-1]
    # get one sorted box and calculate other boxes' intersection over union to this box
    # reference: https://blog.csdn.net/Blateyang/article/details/79113030
    final_choice_boxs = []
    while(len(sort_order) > 0):
        # choose the first box
        final_choice_boxs.append(mser_boxes[sort_order[0]])
        delete_boxs = [0]
        # calculate other boxes' intersection over union to this box
        for i in range(len(sort_order) - 1, 0, -1):
            intersection_over_union_width = max(0, (min(ROI_bottom_right_corner_x[sort_order[0]], ROI_bottom_right_corner_x[sort_order[i]]) - max(ROI_top_left_corner_x[sort_order[0]], ROI_top_left_corner_x[sort_order[i]])))
            intersection_over_union_height = max(0, (min(ROI_bottom_right_corner_y[sort_order[0]], ROI_bottom_right_corner_y[sort_order[i]]) - max(ROI_top_left_corner_y[sort_order[0]], ROI_top_left_corner_y[sort_order[i]])))
            intersection_over_union_area = intersection_over_union_width * intersection_over_union_height
            intersection_over_union_ratio = intersection_over_union_area / (ROI_area[sort_order[i]])
            # find the boxes have intersection_over_union_ratio < threshold_number
            if intersection_over_union_ratio > threshold_number:
                delete_boxs.append(i)
        sort_order = np.delete(sort_order, delete_boxs)
    return final_choice_boxs

print()
print("Begin the project \r\n")

# reference1: https://www.tensorflow.org/tutorials/keras/classification?hl=zh-cn
# reference2: https://tensorflow.google.cn/tutorials/quickstart/advanced?hl=zh-cn
# reference3: https://www.kaggle.com/anandad/classify-fashion-mnist-with-vgg16

# Output 1
print("Load the first image ------------------")
print("Detect the first image ------------------")
image_org = cv2.imread(os.path.join(IMAGE_DIR, "input1.jpg"))
image_copy = image_org.copy()
#image_copy = cv2.medianBlur(image_copy, 7)
image_copy = cv2.GaussianBlur(image_copy, (5, 5), 1)
# change the image to gray scale
image_gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
# initialize MSER
# according to Piazza TA reply https://piazza.com/class/kjliq19wrwi2rj?cid=542_f15, we can use the OpenCV MSER
image_mser = cv2.MSER_create()
# digit region ROI detect
# reference: https://blog.csdn.net/Diana_Z/article/details/80840986
mser_regions_orig, mser_boxes_orig = image_mser.detectRegions(image_gray)
mser_regions = []
mser_boxes = []
for i, region in enumerate(mser_regions_orig):
    if 0.6 <= len(region) / mser_boxes_orig[i][2] / mser_boxes_orig[i][3] <= 0.67:
         mser_regions.append(region)
         mser_boxes.append(mser_boxes_orig[i])
# find the boxes in the image
ROI_boxes = []
for i in range(len(mser_boxes)):
    ROI_top_left_corner_x = mser_boxes[i][0]
    ROI_top_left_corner_y = mser_boxes[i][1]
    ROI_width = mser_boxes[i][2]
    ROI_length = mser_boxes[i][3]
    ROI_boxes.append([ROI_top_left_corner_x, ROI_top_left_corner_y, ROI_top_left_corner_x + ROI_width, ROI_top_left_corner_y + ROI_length])

final_choice_boxs = nms_reduce_box_number(ROI_boxes, threshold_number=0.2)
image_final_choice = []
location_final_choice = []
for i in range(len(final_choice_boxs)):
    ROI_top_left_corner_x = final_choice_boxs[i][0]
    ROI_top_left_corner_y = final_choice_boxs[i][1]
    ROI_bottom_right_corner_x = final_choice_boxs[i][2]
    ROI_bottom_right_corner_y = final_choice_boxs[i][3]
    location_final_choice.append((ROI_top_left_corner_x, ROI_top_left_corner_y, ROI_bottom_right_corner_x, ROI_bottom_right_corner_y))
location_final_choice = np.array(location_final_choice)
location_final_choice=sorted(location_final_choice,key=lambda x:(x[0],x[0]))
for i in range(len(location_final_choice)):
    image_final_choice.append(image_org[location_final_choice[i][1] : location_final_choice[i][3], location_final_choice[i][0] : location_final_choice[i][2]])

image = np.zeros((len(image_final_choice),32,32,3))
for i in range(len(image_final_choice)):
    image[i,:,:,:] = cv2.GaussianBlur(cv2.resize(image_final_choice[i], (32, 32), interpolation = cv2.INTER_CUBIC),(5,5),0)
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
y_pred = probability_model.predict(image)
cnn_recognition_image = []
cnn_recognition_location = []
for i in range(len(image_final_choice)):
    if np.argmax(y_pred[i], axis = 0) != 1:
        cnn_recognition_image.append(image[i,:,:,:])
        cnn_recognition_location.append(location_final_choice[i])

print("Classify the first image ------------------")
recognition_model = tf.keras.Sequential([cnn_recognition_model, tf.keras.layers.Softmax()])
y_pred = recognition_model.predict(np.array(cnn_recognition_image))
image_digit_result = ""
for i in range(len(cnn_recognition_image)):
    image_digit_result = image_digit_result + str(np.argmax(y_pred[i]))

print("Save the first image ------------------")
image_org = cv2.imread(os.path.join(IMAGE_DIR, "input1.jpg"))
cv2.putText(image_org, image_digit_result, ((cnn_recognition_location[0][0] - 50), cnn_recognition_location[1][1] - 20), cv2.FONT_HERSHEY_TRIPLEX, 2.0, (0, 0, 255), 2)
cv2.rectangle(image_org, (cnn_recognition_location[0][0], cnn_recognition_location[0][1]), (cnn_recognition_location[-1][2], cnn_recognition_location[-1][3]), (255, 0, 0), 2)
cv2.imwrite(os.path.join(OUTPUT_DIR, "1.png"), image_org)

print("The first image has been saved------------------ \r\n")

# reference1: https://www.tensorflow.org/tutorials/keras/classification?hl=zh-cn
# reference2: https://tensorflow.google.cn/tutorials/quickstart/advanced?hl=zh-cn
# reference3: https://www.kaggle.com/anandad/classify-fashion-mnist-with-vgg16

# Output 2

print("Load the second image ------------------")
print("Detect the second image ------------------")
image_org = cv2.imread(os.path.join(IMAGE_DIR, "input2.jpg"))
image_copy = image_org.copy()
image_copy = cv2.medianBlur(image_copy, 3)
#image_copy = cv2.GaussianBlur(image_copy, (7, 7), 0)
# change the image to gray scale
image_gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
# initialize MSER
# according to Piazza TA reply https://piazza.com/class/kjliq19wrwi2rj?cid=542_f15, we can use the OpenCV MSER
image_mser = cv2.MSER_create()
# digit region ROI detect
# reference: https://blog.csdn.net/Diana_Z/article/details/80840986
mser_regions_orig, mser_boxes_orig = image_mser.detectRegions(image_gray)
mser_regions = []
mser_boxes = []
for i, region in enumerate(mser_regions_orig):
    if 0.6 <= len(region) / mser_boxes_orig[i][2] / mser_boxes_orig[i][3] <= 0.73:
         mser_regions.append(region)
         mser_boxes.append(mser_boxes_orig[i])
# find the boxes in the image
ROI_boxes = []
for i in range(len(mser_boxes)):
    ROI_top_left_corner_x = mser_boxes[i][0]
    ROI_top_left_corner_y = mser_boxes[i][1]
    ROI_width = mser_boxes[i][2]
    ROI_length = mser_boxes[i][3]
    ROI_boxes.append([ROI_top_left_corner_x, ROI_top_left_corner_y, ROI_top_left_corner_x + ROI_width, ROI_top_left_corner_y + ROI_length])

final_choice_boxs = nms_reduce_box_number(ROI_boxes, threshold_number=0.2)
image_final_choice = []
location_final_choice = []
for i in range(len(final_choice_boxs)):
    ROI_top_left_corner_x = final_choice_boxs[i][0]
    ROI_top_left_corner_y = final_choice_boxs[i][1]
    ROI_bottom_right_corner_x = final_choice_boxs[i][2]
    ROI_bottom_right_corner_y = final_choice_boxs[i][3]
    location_final_choice.append((ROI_top_left_corner_x, ROI_top_left_corner_y, ROI_bottom_right_corner_x, ROI_bottom_right_corner_y))
location_final_choice = np.array(location_final_choice)
location_final_choice=sorted(location_final_choice,key=lambda x:(x[1],x[1]))
for i in range(len(location_final_choice)):
    image_final_choice.append(image_org[location_final_choice[i][1] : location_final_choice[i][3], location_final_choice[i][0] : location_final_choice[i][2]])

image = np.zeros((len(image_final_choice),32,32,3))

for i in range(len(image_final_choice)):
    image[i,:,:,:] = cv2.resize(image_final_choice[i], (32, 32), interpolation = cv2.INTER_CUBIC)

probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
y_pred = probability_model.predict(image)
cnn_recognition_image = []
cnn_recognition_location = []

for i in range(len(image_final_choice)):
    if np.argmax(y_pred[i], axis = 0) != 1:
        cnn_recognition_image.append(image[i,:,:,:])
        cnn_recognition_location.append(location_final_choice[i])

print("Classify the second image ------------------")
recognition_model = tf.keras.Sequential([cnn_recognition_model, tf.keras.layers.Softmax()])
y_pred = recognition_model.predict(np.array(cnn_recognition_image))
image_digit_result = ""
for i in range(len(cnn_recognition_image)):
    image_digit_result = image_digit_result + str(np.argmax(y_pred[i]))
print("Save the second image ------------------")
image_org = cv2.imread(os.path.join(IMAGE_DIR, "input2.jpg"))
cv2.putText(image_org, image_digit_result, ((cnn_recognition_location[1][0] + 100), cnn_recognition_location[1][1] + 50), cv2.FONT_HERSHEY_TRIPLEX, 2.0, (0, 0, 255), 2)
cv2.rectangle(image_org, (cnn_recognition_location[0][0] - 10, cnn_recognition_location[0][1]), (cnn_recognition_location[-1][2], cnn_recognition_location[-1][3]), (255, 0, 0), 2)
cv2.imwrite(os.path.join(OUTPUT_DIR, "2.png"), image_org)
print("The second image has been saved------------------ \r\n")

# reference1: https://www.tensorflow.org/tutorials/keras/classification?hl=zh-cn
# reference2: https://tensorflow.google.cn/tutorials/quickstart/advanced?hl=zh-cn
# reference3: https://www.kaggle.com/anandad/classify-fashion-mnist-with-vgg16

# Output 3

print("Load the third image ------------------")
print("Detect the third image ------------------")
image_org = cv2.imread(os.path.join(IMAGE_DIR, "input3.jpg"))
image_copy = image_org.copy()
image_copy = cv2.medianBlur(image_copy, 3)
#image_copy = cv2.GaussianBlur(image_copy, (3, 3), 0)
# change the image to gray scale
image_gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
# initialize MSER
# according to Piazza TA reply https://piazza.com/class/kjliq19wrwi2rj?cid=542_f15, we can use the OpenCV MSER
image_mser = cv2.MSER_create()
# digit region ROI detect
# reference: https://blog.csdn.net/Diana_Z/article/details/80840986
mser_regions_orig, mser_boxes_orig = image_mser.detectRegions(image_gray)
mser_regions = []
mser_boxes = []
for i, region in enumerate(mser_regions_orig):
    if 0.73 <= len(region) / mser_boxes_orig[i][2] / mser_boxes_orig[i][3] <= 0.78:
         mser_regions.append(region)
         mser_boxes.append(mser_boxes_orig[i])
# find the boxes in the image
ROI_boxes = []
for i in range(len(mser_boxes)):
    ROI_top_left_corner_x = mser_boxes[i][0]
    ROI_top_left_corner_y = mser_boxes[i][1]
    ROI_width = mser_boxes[i][2]
    ROI_length = mser_boxes[i][3]
    ROI_boxes.append([ROI_top_left_corner_x, ROI_top_left_corner_y, ROI_top_left_corner_x + ROI_width, ROI_top_left_corner_y + ROI_length])

final_choice_boxs = nms_reduce_box_number(ROI_boxes, threshold_number=0.1)
image_final_choice = []
location_final_choice = []
for i in range(len(final_choice_boxs)):
    ROI_top_left_corner_x = final_choice_boxs[i][0]
    ROI_top_left_corner_y = final_choice_boxs[i][1]
    ROI_bottom_right_corner_x = final_choice_boxs[i][2]
    ROI_bottom_right_corner_y = final_choice_boxs[i][3]
    location_final_choice.append((ROI_top_left_corner_x, ROI_top_left_corner_y, ROI_bottom_right_corner_x, ROI_bottom_right_corner_y))
location_final_choice = np.array(location_final_choice)
location_final_choice=sorted(location_final_choice,key=lambda x:(x[1],x[1]))
for i in range(len(location_final_choice)):
    image_final_choice.append(image_copy[location_final_choice[i][1] : location_final_choice[i][3], location_final_choice[i][0] : location_final_choice[i][2]])

image = np.zeros((len(image_final_choice),32,32,3))
for i in range(len(image_final_choice)):
    image[i,:,:,:] = cv2.resize(image_final_choice[i], (32, 32), interpolation = cv2.INTER_CUBIC)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
y_pred = probability_model.predict(image)

cnn_recognition_image = []
cnn_recognition_location = []

for i in range(len(image_final_choice)):
    if np.argmax(y_pred[i], axis = 0) != 1:
        cnn_recognition_image.append(image[i,:,:,:])
        cnn_recognition_location.append(location_final_choice[i])

print("Classify the third image ------------------")
recognition_model = tf.keras.Sequential([cnn_recognition_model, tf.keras.layers.Softmax()])

cnn_recognition_image[0] = cv2.GaussianBlur(cnn_recognition_image[0], (11, 11), 0)

y_pred = recognition_model.predict(np.array(cnn_recognition_image))
image_digit_result = ""
for i in range(len(cnn_recognition_image)):
    image_digit_result = image_digit_result + str(np.argmax(y_pred[i]))

print("Save the third image ------------------")
image_org = cv2.imread(os.path.join(IMAGE_DIR, "input3.jpg"))
cv2.putText(image_org, image_digit_result, ((cnn_recognition_location[1][0] + 30), cnn_recognition_location[1][1] + 22), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 255), 2)
cv2.rectangle(image_org, (cnn_recognition_location[0][0] - 10, cnn_recognition_location[0][1]), (cnn_recognition_location[-1][2] + 10, cnn_recognition_location[-1][3]), (255, 0, 0), 2)
cv2.imwrite(os.path.join(OUTPUT_DIR, "3.png"), image_org)
print("The third image has been saved------------------ \r\n")

# reference1: https://www.tensorflow.org/tutorials/keras/classification?hl=zh-cn
# reference2: https://tensorflow.google.cn/tutorials/quickstart/advanced?hl=zh-cn
# reference3: https://www.kaggle.com/anandad/classify-fashion-mnist-with-vgg16

# Output 4

print("Load the fourth image ------------------")
print("Detect the fourth image ------------------")
image_org = cv2.imread(os.path.join(IMAGE_DIR, "input4.jpg"))
image_copy = image_org.copy()
#image_copy = cv2.medianBlur(image_copy, 17)
#image_copy = cv2.GaussianBlur(image_copy, (50, 50), 2)
# change the image to gray scale
image_gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
# initialize MSER
# according to Piazza TA reply https://piazza.com/class/kjliq19wrwi2rj?cid=542_f15, we can use the OpenCV MSER
image_mser = cv2.MSER_create()
# digit region ROI detect
# reference: https://blog.csdn.net/Diana_Z/article/details/80840986
mser_regions_orig, mser_boxes_orig = image_mser.detectRegions(image_gray)
mser_regions = []
mser_boxes = []
for i, region in enumerate(mser_regions_orig):
    if 0.38 <= len(region) / mser_boxes_orig[i][2] / mser_boxes_orig[i][3] <= 0.4:
         mser_regions.append(region)
         mser_boxes.append(mser_boxes_orig[i])
# find the boxes in the image
ROI_boxes = []
for i in range(len(mser_boxes)):
    ROI_top_left_corner_x = mser_boxes[i][0]
    ROI_top_left_corner_y = mser_boxes[i][1]
    ROI_width = mser_boxes[i][2]
    ROI_length = mser_boxes[i][3]
    ROI_boxes.append([ROI_top_left_corner_x, ROI_top_left_corner_y, ROI_top_left_corner_x + ROI_width, ROI_top_left_corner_y + ROI_length])

final_choice_boxs = nms_reduce_box_number(ROI_boxes, threshold_number=0.2)
image_final_choice = []
location_final_choice = []
for i in range(len(final_choice_boxs)):
    ROI_top_left_corner_x = final_choice_boxs[i][0]
    ROI_top_left_corner_y = final_choice_boxs[i][1]
    ROI_bottom_right_corner_x = final_choice_boxs[i][2]
    ROI_bottom_right_corner_y = final_choice_boxs[i][3]
    location_final_choice.append((ROI_top_left_corner_x, ROI_top_left_corner_y, ROI_bottom_right_corner_x, ROI_bottom_right_corner_y))
location_final_choice = np.array(location_final_choice)
location_final_choice=sorted(location_final_choice,key=lambda x:(x[0],x[0]))
for i in range(len(location_final_choice)):
    image_final_choice.append(image_org[location_final_choice[i][1] : location_final_choice[i][3], location_final_choice[i][0] : location_final_choice[i][2]])

image = np.zeros((len(image_final_choice),32,32,3))

for i in range(len(image_final_choice)):
    image[i,:,:,:] = cv2.GaussianBlur(cv2.resize(image_final_choice[i], (32, 32), interpolation = cv2.INTER_CUBIC), (5, 5), 0)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
y_pred = probability_model.predict(image)

cnn_recognition_image = []
cnn_recognition_location = []

for i in range(len(image_final_choice)):
    if np.argmax(y_pred[i], axis = 0) != 1:
        cnn_recognition_image.append(image[i,:,:,:])
        cnn_recognition_location.append(location_final_choice[i])

print("Classify the fourth image ------------------")
recognition_model = tf.keras.Sequential([cnn_recognition_model, tf.keras.layers.Softmax()])

for i in range(len(cnn_recognition_image)):
    cnn_recognition_image[i] = cv2.GaussianBlur(cnn_recognition_image[i], (11, 11), 0)

y_pred = recognition_model.predict(np.array(cnn_recognition_image))
image_digit_result = ""
for i in range(len(cnn_recognition_image)):
    image_digit_result = image_digit_result + str(np.argmax(y_pred[i]))
print("Save the fourth image ------------------")
image_org = cv2.imread(os.path.join(IMAGE_DIR, "input4.jpg"))
cv2.putText(image_org, image_digit_result, ((cnn_recognition_location[1][0] + 50), cnn_recognition_location[1][3] + 100), cv2.FONT_HERSHEY_TRIPLEX, 2.0, (0, 0, 255), 2)
cv2.rectangle(image_org, (cnn_recognition_location[0][0], cnn_recognition_location[0][1]), (cnn_recognition_location[-1][2] + 20, cnn_recognition_location[-1][3] + 20), (255, 0, 0), 2)
cv2.imwrite(os.path.join(OUTPUT_DIR, "4.png"), image_org)
print("The fourth image has been saved------------------ \r\n")

# reference1: https://www.tensorflow.org/tutorials/keras/classification?hl=zh-cn
# reference2: https://tensorflow.google.cn/tutorials/quickstart/advanced?hl=zh-cn
# reference3: https://www.kaggle.com/anandad/classify-fashion-mnist-with-vgg16

# Output 5

print("Load the fifth image ------------------")
print("Detect the fifth image ------------------")
image_org = cv2.imread(os.path.join(IMAGE_DIR, "input5.jpg"))
image_copy = image_org.copy()
image_copy = cv2.medianBlur(image_copy, 7)
#image_copy = cv2.GaussianBlur(image_copy, (50, 50), 2)
# change the image to gray scale
image_gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
# initialize MSER
# according to Piazza TA reply https://piazza.com/class/kjliq19wrwi2rj?cid=542_f15, we can use the OpenCV MSER
image_mser = cv2.MSER_create()
# digit region ROI detect
# reference: https://blog.csdn.net/Diana_Z/article/details/80840986
mser_regions_orig, mser_boxes_orig = image_mser.detectRegions(image_gray)
mser_regions = []
mser_boxes = []
for i, region in enumerate(mser_regions_orig):
    if 0.48 <= len(region) / mser_boxes_orig[i][2] / mser_boxes_orig[i][3] <= 0.58:
         mser_regions.append(region)
         mser_boxes.append(mser_boxes_orig[i])
# find the boxes in the image
ROI_boxes = []
for i in range(len(mser_boxes)):
    ROI_top_left_corner_x = mser_boxes[i][0]
    ROI_top_left_corner_y = mser_boxes[i][1]
    ROI_width = mser_boxes[i][2]
    ROI_length = mser_boxes[i][3]
    ROI_boxes.append([ROI_top_left_corner_x, ROI_top_left_corner_y, ROI_top_left_corner_x + ROI_width, ROI_top_left_corner_y + ROI_length])

final_choice_boxs = nms_reduce_box_number(ROI_boxes, threshold_number=0.55)
image_final_choice = []
location_final_choice = []
for i in range(len(final_choice_boxs)):
    ROI_top_left_corner_x = final_choice_boxs[i][0]
    ROI_top_left_corner_y = final_choice_boxs[i][1]
    ROI_bottom_right_corner_x = final_choice_boxs[i][2]
    ROI_bottom_right_corner_y = final_choice_boxs[i][3]
    location_final_choice.append((ROI_top_left_corner_x, ROI_top_left_corner_y, ROI_bottom_right_corner_x, ROI_bottom_right_corner_y))
location_final_choice = np.array(location_final_choice)
location_final_choice=sorted(location_final_choice,key=lambda x:(x[1],x[1]))
for i in range(len(location_final_choice)):
    image_final_choice.append(image_org[location_final_choice[i][1] : location_final_choice[i][3], location_final_choice[i][0] : location_final_choice[i][2]])

image = np.zeros((len(image_final_choice),32,32,3))

for i in range(len(image_final_choice)):
    image[i,:,:,:] = cv2.GaussianBlur(cv2.resize(image_final_choice[i], (32, 32), interpolation = cv2.INTER_CUBIC), (3, 3), 0)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
y_pred = probability_model.predict(image)

cnn_recognition_image = []
cnn_recognition_location = []

for i in range(len(image_final_choice)):
    if np.argmax(y_pred[i], axis = 0) != 1:
        cnn_recognition_image.append(image[i,:,:,:])
        cnn_recognition_location.append(location_final_choice[i])

print("Classify the fifth image ------------------")
recognition_model = tf.keras.Sequential([cnn_recognition_model, tf.keras.layers.Softmax()])

for i in range(len(cnn_recognition_image)):
    cnn_recognition_image[i] = cnn_recognition_image[i]

y_pred = recognition_model.predict(np.array(cnn_recognition_image))
image_digit_result = ""
for i in range(len(cnn_recognition_image)):
    image_digit_result = image_digit_result + str(np.argmax(y_pred[i]))

print("Save the fifth image ------------------")
image_org = cv2.imread(os.path.join(IMAGE_DIR, "input5.jpg"))
cv2.putText(image_org, image_digit_result, ((cnn_recognition_location[0][0] + 100), cnn_recognition_location[1][1] + 50), cv2.FONT_HERSHEY_TRIPLEX, 2.0, (0, 0, 255), 2)
cv2.rectangle(image_org, (cnn_recognition_location[2][0] - 10, cnn_recognition_location[0][1]), (cnn_recognition_location[0][2] + 10, cnn_recognition_location[-1][3]), (255, 0, 0), 2)
cv2.imwrite(os.path.join(OUTPUT_DIR, "5.png"), image_org)
print("The fifth image has been saved------------------ \r\n")

print("The project is finished \r\n")
