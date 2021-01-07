import cv2
import imutils
import numpy as np
import tensorflow as tf
from Utilities import find_puzzle, extract_digit
import numpy as np
import imutils
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

file_name = "model"
models_path = os.listdir(file_name)
models = []
for path in models_path:
    models.append(load_model(file_name+"/"+path))


standard = [
    [[1, 2, 9, 8, 4, 4, 6, 9, 2, 5, 7, 3, 6, 7, 8, 3, 8, 5, 6, 1, 2, 8, 3, 5, 7, 4, 5, 6, 5, 8, 1, 6, 4, 5, 7, 8],
     [5, 4, 1, 7, 8, 6, 3, 5, 9, 1, 5, 6, 7, 2, 5, 9, 4, 1, 2, 3, 6, 2, 4, 3, 7, 8, 1, 5, 4, 2],
     [6, 7, 9, 4, 2, 1, 2, 5, 5, 8, 9, 8, 4, 9, 3, 7, 3, 1, 7, 2, 8, 9, 1, 6, 3, 4],
     [8, 4, 9, 4, 3, 1, 9, 5, 1, 5, 8, 7, 9, 3, 5, 6, 1, 4, 2, 1, 3, 6, 8, 9, 7, 5],
     [9, 7, 6, 4, 7, 4, 8, 9, 1, 9, 8, 5, 6, 3, 8, 7, 2, 5, 6, 3, 1, 1, 8, 4, 2, 6, 1, 4, 3, 6, 7, 1, 1, 3, 4, 8]],
    [[8, 9, 1, 1, 2, 5, 6, 7, 8, 6, 3, 4, 7, 3, 8, 2, 6, 5, 4, 8, 6, 4, 7, 9, 4, 1, 9, 7, 3, 9, 2, 6, 5, 1, 6, 4],
     [5, 7, 6, 9, 8, 6, 4, 2, 3, 7, 8, 3, 7, 9, 5, 2, 2, 1, 4, 6, 3, 5, 8, 4, 8, 7, 9, 7, 2, 6],
     [2, 1, 9, 5, 1, 6, 7, 5, 7, 9, 5, 7, 8, 2, 6, 6, 2, 3, 9, 3, 8, 2, 7, 1, 9, 5],
     [6, 8, 1, 2, 3, 5, 5, 9, 6, 4, 8, 7, 6, 2, 3, 7, 2, 9],
     [9, 3, 6, 1, 5, 5, 7, 4, 3, 7, 1, 2, 9, 8, 3, 6, 8, 9, 6, 2, 6, 5, 3, 2, 4, 7, 9, 3, 4, 5, 9, 2]]
]

for j in range(len(models)):
    model = models[j]
    miss = []
    for flag in range(2):
        for pic in range(5):
            image = cv2.imread("test1/%s-%s.jpg" % (flag+1, pic+1))
            image = imutils.resize(image, width=600)
            (puzzle, warped) = find_puzzle(image, debug=False)
            stepX = warped.shape[1] // 9
            stepY = warped.shape[0] // 9
            check = []
            for y in range(9):
                for x in range(9):
                    startX = x * stepX
                    startY = y * stepY
                    endX = (x + 1) * stepX
                    endY = (y + 1) * stepY
                    cell = warped[startY+8:endY-4, startX+8:endX-4]
                    digit = extract_digit(cell, False)
                    if digit is not None:
                        roi = cv2.resize(digit, (20, 20))
                        paddings = tf.constant([[4, 4], [4, 4]])
                        roi = np.pad(roi, paddings, "constant")  # padding 补足

                        roi = tf.cast(roi, dtype=tf.float32)
                        roi = roi / 255.0
                        roi = img_to_array(roi)
                        check.append(roi)
                        
            count = 0
            for i in range(len(standard[flag][pic])):
                image = check[i]
                image = img_to_array(image)
                image = np.expand_dims(image, axis=0)
                num = model.predict(image).argmax(axis=1)[0]
                if num > 9:
                    num = num-9
                if(num != standard[flag][pic][i]):
                    count += 1
            miss.append(count)
    print(" %s model" % (models_path[j]))
    print(miss)
