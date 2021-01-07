# import the necessary packages
from models import LeNet, MyNet
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import numpy as np
import random
# set the random seed
random.seed(511)


trainData = np.load('./Data/mix_train_data.npy')
trainLabels = np.load('./Data/mix_train_labels.npy')
testData = np.load('./Data/mix_test_data.npy')
testLabels = np.load('./Data/mix_test_labels.npy')


# add a channel (i.e., grayscale) dimension to the digits
trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
testData = testData.reshape((testData.shape[0], 28, 28, 1))


# scale data to the range of [0, 1]
trainData = trainData.astype("float32")
testData = testData.astype("float32")

# convert the labels from integers to vectors
le = LabelBinarizer()
trainLabels = le.fit_transform(trainLabels)
testLabels = le.transform(testLabels)

# initialize the initial learning rate, number of epochs to train for, and batch size
INIT_LR = [1e-4, 2e-4, 4e-4, 8e-4, 16e-4]
EPOCHS = [10, 20, 30, 50]
BS = [16, 32, 64, 128, 256]
Filter_Size = [3, 4, 5]


def train():
    for epoch in EPOCHS:
            learning_rate = 1e-4
            bs = 32
            print("setting the parameter: lr=%s,epochs=%s,batchsize=%s" %
                  (learning_rate, epoch, bs))
            # initialize the optimizer and model
            print("[INFO] compiling model...")
            opt = Adam(lr=learning_rate)
            model = MyNet.build(width=28, height=28, depth=1, classes=19)
            model.compile(loss="categorical_crossentropy", optimizer=opt,
                          metrics=["accuracy"])

            # train the network
            print("[INFO] training network...")
            H = model.fit(
                trainData, trainLabels,
                validation_data=(testData, testLabels),
                epochs=epoch,
                batch_size=bs,
                verbose=1
            )

            dic = H.history
            txtname = "train_log/training_log_%s_%s_%s.txt" % (
                 learning_rate, epoch, bs)
            for data in dic.keys():
                with open(txtname, 'a') as f:
                    f.write(data+'\n')
                    for i in range(len(dic[data])):
                        f.write(str(dic[data][i])+"\n")

            # evaluate the network
            print("[INFO] evaluating network...")
            predictions = model.predict(testData)
            print(classification_report(
                testLabels.argmax(axis=1),
                predictions.argmax(axis=1),
                target_names=[str(x) for x in le.classes_]))

            print("[INFO] serializing digit model...")
            model.save("model/mynet_%s_%s_%s.h5" %
                        (learning_rate, epoch, bs), save_format="h5")

if __name__=="__main__":
    train()