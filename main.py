import cv2
import tensorflow
from tensorflow import keras
import warnings
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model

def run(source=None):

    model = tensorflow.keras.models.load_model(r'"C:\Users\chait\OneDrive\Desktop\AKHIL\models\different_class_accuracy\weights-best-27-0.99-0.70.hdf5"')

    img = cv2.imread(source)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(img_gray, (5, 5), 0)   
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    img_half = cv2.resize(img, (180, 180))
    img_half = cv2.cvtColor(img_half, cv2.COLOR_BGR2GRAY)
    x = img_to_array(img_half)
    x = np.array(x).reshape(-1, 180, 180, 1)
    prediction = model.predict(x)
    if np.argmax(prediction) == 0:
        # print("Dont worry your Knee was very safe, take healthy food ")
        return 'skin cancer type is a Acnitic Keratosis'
    elif np.argmax(prediction) == 1:
        # print("The Fabric is Good. ")
        return 'skin cancer type is a Dermatofibroma'
    elif np.argmax(prediction) == 2:
        # print("The Fabric is Good. ")
        return 'skin cancer type is a Vascular Lesion'
def analysis():
    return """accuracy of the model :0.92%
    precision  of the model :0.93%
    recall  of the model:0.93%
    f1-score of the model:0.93%"""          

# run(source=r"C:\Users\lenovo\Downloads\skin canser\resized_new_classes-20230302T065126Z-001\resized_new_classes\D\images0.jpg")