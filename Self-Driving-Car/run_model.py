import numpy as np
import cv2
import os
import pickle
import time
from subprocess import call
from keras.models import load_model

# check if on windows OS
windows = False
if os.name == 'nt':
    windows = True

# Loading saved model
try:
    model = load_model('/Users/ravneet/Docs/MSCS/AI Autonomous/AI-Autonomous/Self-Driving-Car/models/Autopilot_new_1.keras', safe_mode=False)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Function to predict car steering angle
def predict_model(model, image):
    processed_image = process_image(image)
    prediction = model.predict(processed_image, batch_size=1)
    steering_angle = float(prediction[0])
    steering_angle = steering_angle * 180 / np.pi
    return steering_angle

# Function to resize and reshape the images
def process_image(img):
    image_x = 100
    image_y = 100
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img

def loadFromPickle():
    with open("labels", "rb") as f:
        labels = np.array(pickle.load(f))

    return labels

labels = loadFromPickle()

# Loading car steering image
img = cv2.imread(f"{os.getcwd()}/steering_wheel_image.jpg", 0)
rows,cols = img.shape
smoothed_angle = 0

# Reading car dashcam images frame by frame from dataset
i = 0
while(cv2.waitKey(10) != ord('q')):
    file_path = f"{os.getcwd()}/driving_dataset/{i}.jpg"
    image = cv2.imread(file_path)
    if image is None:
        print(f"Could not read image {file_path}")
        break

    degrees = predict_model(model, image)

    if not windows:
        call("clear")

    #Printing Actual angle and calculated predicted angle
    actual_angle = labels[i]
    print(f"Actual steering angle:    {str(actual_angle)} degrees")
    print(f"Predicted steering angle: {str(degrees)} degrees")
    cv2.imshow("frame", image)
    smoothed_angle = smoothed_angle * 0.9 + degrees * 0.1
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("steering wheel", dst)
    time.sleep(0.01)
    i += 1

cv2.destroyAllWindows()
