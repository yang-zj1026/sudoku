# import the necessary packages
from tensorflow.keras import models, layers


class MyNet:
    def build(width, height, depth, classes):
        model = models.Sequential()
        # 第1层卷积，卷积核大小为3*3，32个，28*28为待训练图片的大小
        model.add(layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=(width, height, depth)))
        model.add(layers.MaxPooling2D((2, 2)))
        # 第2层卷积，卷积核大小为3*3，64个
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        # 第3层卷积，卷积核大小为3*3，64个
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(98, activation='relu'))
        model.add(layers.Dense(classes, activation='softmax'))

        # return the constructed network architecture
        return model


class LeNet:
    def build(width, height, depth, classes):
        model = models.Sequential()
        # 第1层卷积，卷积核大小为5*5，6个，28*28为待训练图片的大小
        model.add(layers.Conv2D(
            6, (5, 5), activation='relu', input_shape=(width, height, depth)))
        model.add(layers.MaxPooling2D((2, 2)))
        # 第2层卷积，卷积核大小为5*5，16个
        model.add(layers.Conv2D(
            16, (5, 5), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(120, activation='relu'))
        # model.add(layers.Dense(84, activation='relu'))
        model.add(layers.Dense(classes, activation='softmax'))

        # return the constructed network architecture
        return model