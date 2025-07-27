import os
import json
import cv2
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import random
from assets.FrontEnd_Module import featureDetection
import numpy as np
import pickle

# List of all Images --------------
def load_JSON(name):
    with open(name, 'r') as openfile:
        json_object = json.load(openfile)
        return json_object
def store_JSON(name, data):
    data = json.dumps(data, indent=4)
    with open(name, "w") as outfile:
        outfile.write(data)
def testJSON(data):
    print(len(data))
    sum = 0
    for key in data:
        length = len(data[key])
        print(length)
        sum += length
    return sum
def get_Images():
    BASE_DIR = r"D:/coding/Temp_Download/data_odometry_color/dataset/sequences"
    SEQUENCE = [f"{index:02d}" for index in range(11)]
    IMAGES = dict()
    for seq in SEQUENCE:
        IMAGE_FOLDER = os.path.join(BASE_DIR, seq, "image_2")
        IMAGES[seq] = [os.path.join(os.path.abspath(IMAGE_FOLDER), img) for img in os.listdir(IMAGE_FOLDER) if img.endswith('.png')]
    store_JSON(name='IMAGES.json', data=IMAGES)
    return load_JSON(name='IMAGES.json')
# ------------------------

# Test_Train Split --------
def TestTrainSplit(name):
    img = load_JSON(name=name)
    data = {
        "Train": [],
        "Test": [],
    }
    for seq in img:
        Train, Test = train_test_split(img[seq], test_size =0.4, random_state=42)
        data["Train"].extend(Train)
        data["Test"].extend(Test)
    store_JSON(name="./Archieve/TrainTest.json", data=data)
    return load_JSON(name="TrainTest.json")
# -------------------------

# CodeBook -------------------
def getImages(imgs):
    for img in imgs:
        yield cv2.imread(img) 
def generateCodeBook(numOfWords=500, sampleSize=100):
    length_whole = load_JSON(name="./Archieve/TrainTest.json")
    length = len(length_whole["Train"])
    index = 0
    FD = featureDetection()
    descriptors = []
    for img in getImages(trainTest["Train"]):
        kp, desc, frame = FD.FD_SIFT(img)
        if len(desc):
            if len(desc) > sampleSize:
                desc = desc[:sampleSize]
            descriptors.append(desc)
        if index%100==0:
            print(f"{index} images read, {length-index} left")
        index+=1
    print(f"{index} images read, {length-index} left")
    descriptors = np.vstack(descriptors)
    kmeans = KMeans ( n_clusters = numOfWords , random_state =42)
    kmeans.fit(descriptors)
    print(kmeans)
    with open("KMeans.pkl", "wb") as f:
        pickle.dump(kmeans, f) 
    return kmeans
# ----------------------------
# Main Function 
if __name__=="__main__":
    # try:
    #     imgs = load_JSON(name='./Archieve/IMAGES.json')
    # except:
    #     imgs = get_Images()
    # sum1 = testJSON(data=imgs)
    # test_train = TestTrainSplit(name='./Archieve/IMAGES.json')
    # sum2 = testJSON(data=test_train)
    # print(sum1 == sum2)
    trainTest = load_JSON(name="./Archieve/TrainTest.json")
    index = 0
    generateCodeBook()