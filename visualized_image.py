# import common python librairies
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Displaying Images
picture_size = 48
folder_path = "data"

expressions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
for expression in expressions:
    plt.figure(expression, figsize=(12,12))
    for i in range(1, 10, 1):
            plt.subplot(3,3,i)
            # plt.title(expression)
            img  = load_img(folder_path+"/train/"+expression+"/"+ os.listdir(folder_path+"/train/"+ expression)[i], target_size=(picture_size, picture_size))
            plt.imshow(img)
    plt.show()