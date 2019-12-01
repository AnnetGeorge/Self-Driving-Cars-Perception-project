import os
import cv2
import numpy as np
from tqdm import tqdm
import csv
import shutil

def readCSV(filename):
    paths = []
    labels = []

    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            paths.append(row[0])
            labels.append(row[1])

    return paths[1:], labels[1:]

REBUILD_DATA = True

class CarClassifier():

    TRAINING_FOLDER_NAME = "trainval"
    TEST_FOLDER_NAME = "test"
    SAVE_SIZE = 500

    trainCount = 0
    testCount = 0
    numClasses = 4


    #reading labels from csv file
    paths, labels = readCSV('../labels.csv')
    labels_dict = {}
    for i in range(len(paths)):
        labels_dict[paths[i]] = labels[i]


    def make_training_data(self):
        #BUILDING TRAINING DATA
        for folder in tqdm(os.listdir(self.TRAINING_FOLDER_NAME)):
            try:
                #loading training data
                for file in tqdm(os.listdir(os.path.join(self.TRAINING_FOLDER_NAME, folder))):
                    if file.endswith('.jpg'):
                        try:
                            path = os.path.join(self.TRAINING_FOLDER_NAME, folder, file)
                            label = int(self.labels_dict[os.path.join(folder, file)[:-10]])
                            shutil.copyfile(path, os.path.join("train", str(label), folder+file))
                            self.trainCount += 1

                        except Exception as e:
                            pass

            except Exception as e:
                pass

        print(self.trainCount)
        for folder in tqdm(os.listdir(self.TEST_FOLDER_NAME)):
            try:
                #loading training data
                for file in tqdm(os.listdir(os.path.join(self.TEST_FOLDER_NAME, folder))):

                    if file.endswith('.jpg'):
                        try:
                            path = os.path.join(self.TEST_FOLDER_NAME, folder, file)
                            dest = shutil.copyfile(path, os.path.join("val", str(0), folder+file))
                            self.testCount += 1

                        except Exception as e:
                            pass

            except Exception as e:
                pass

        #BUILDING TESTING DATA

if REBUILD_DATA:
    car_classifier= CarClassifier()
    car_classifier.make_training_data()





