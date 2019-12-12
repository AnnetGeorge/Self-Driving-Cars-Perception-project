import tensorflow as tf
import sys
import os
import numpy as np
import csv
import shutil
from tqdm import tqdm
class CarClassifier():
    TEST_FOLDER_NAME = "test"
    SAVE_SIZE = 500
    testing_data = []
    testCount = 0
    numClasses = 4

    def inference_data(self):
	label_lines = [line.rstrip() for line in tf.gfile.GFile("/tf_files/retrained_labels.txt")]
        with open('output.csv', mode='w') as inf_file:
	    inf_writer = csv.writer(inf_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	    inf_writer.writerow(["guid/image","label"])	
            for folder in tqdm(os.listdir(self.TEST_FOLDER_NAME)):
                try:
                    #loading testing data
                    for file in tqdm(os.listdir(os.path.join(self.TEST_FOLDER_NAME, folder))):
                        if file.endswith('.jpg'):
                            try:
                                path = os.path.join(self.TEST_FOLDER_NAME, folder, file)
                                file_name = path.split("/")
                                file_name = file_name[-2]+"/"+file_name[-1]
				file_name = file_name.split("_")[0]
                                self.testCount += 1
                                image_data = tf.gfile.FastGFile(path, 'rb').read()
                                predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
                                top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
                                print("filename:"+file_name+" prediction:"+ label_lines[top_k[0]])
				inf_writer.writerow([file_name,label_lines[top_k[0]]])
                            except Exception as e:
                                print(e)
                except Exception as e:
                    print(e)
    
if __name__== "__main__":
    test_dir = sys.argv[1]
    car_classifier= CarClassifier()
    car_classifier.TEST_FOLDER_NAME = test_dir
    with tf.gfile.FastGFile("/tf_files/retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        car_classifier.inference_data()
