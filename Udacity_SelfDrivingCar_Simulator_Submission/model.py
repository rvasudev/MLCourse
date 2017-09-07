import numpy as np
from sklearn.utils import shuffle
import time
import csv
from keras.models import Sequential
from keras.layers import Cropping2D, Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
import sklearn
import cv2
from keras.utils.visualize_util import plot


# Load the sample files names into trainin and validation data sets.
# We are only loading the image paths at this point - not the ctual images.
# We will be using a generator to load the actual images in smaller batches
# to avoid having to load all the images into memory.
def loadDataFiles(training_file):
    path = 'data/'
    samples = []
    with open(path + training_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    # Split the data into training and validation data. Validation data is 20% of the
    # training data
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    return train_samples, validation_samples


# Build a model in Keras and return the model
def getModel():

    '''
    The overall model is as follows:

        Layer (type)                     Output Shape          Param #     Connected to
        ====================================================================================================
        cropping2d_1 (Cropping2D)        (None, 60, 320, 3)    0           cropping2d_input_1[0][0]
        ____________________________________________________________________________________________________
        lambda_1 (Lambda)                (None, 60, 320, 3)    0           cropping2d_1[0][0]
        ____________________________________________________________________________________________________
        convolution2d_1 (Convolution2D)  (None, 56, 316, 24)   1824        lambda_1[0][0]
        ____________________________________________________________________________________________________
        activation_1 (Activation)        (None, 56, 316, 24)   0           convolution2d_1[0][0]
        ____________________________________________________________________________________________________
        maxpooling2d_1 (MaxPooling2D)    (None, 18, 105, 24)   0           activation_1[0][0]
        ____________________________________________________________________________________________________
        dropout_1 (Dropout)              (None, 18, 105, 24)   0           maxpooling2d_1[0][0]
        ____________________________________________________________________________________________________
        convolution2d_2 (Convolution2D)  (None, 14, 101, 36)   21636       dropout_1[0][0]
        ____________________________________________________________________________________________________
        activation_2 (Activation)        (None, 14, 101, 36)   0           convolution2d_2[0][0]
        ____________________________________________________________________________________________________
        maxpooling2d_2 (MaxPooling2D)    (None, 4, 33, 36)     0           activation_2[0][0]
        ____________________________________________________________________________________________________
        dropout_2 (Dropout)              (None, 4, 33, 36)     0           maxpooling2d_2[0][0]
        ____________________________________________________________________________________________________
        convolution2d_3 (Convolution2D)  (None, 2, 31, 48)     15600       dropout_2[0][0]
        ____________________________________________________________________________________________________
        activation_3 (Activation)        (None, 2, 31, 48)     0           convolution2d_3[0][0]
        ____________________________________________________________________________________________________
        flatten_1 (Flatten)              (None, 2976)          0           activation_3[0][0]
        ____________________________________________________________________________________________________
        dense_1 (Dense)                  (None, 1164)          3465228     flatten_1[0][0]
        ____________________________________________________________________________________________________
        activation_4 (Activation)        (None, 1164)          0           dense_1[0][0]
        ____________________________________________________________________________________________________
        dropout_3 (Dropout)              (None, 1164)          0           activation_4[0][0]
        ____________________________________________________________________________________________________
        dense_2 (Dense)                  (None, 100)           116500      dropout_3[0][0]
        ____________________________________________________________________________________________________
        activation_5 (Activation)        (None, 100)           0           dense_2[0][0]
        ____________________________________________________________________________________________________
        dense_3 (Dense)                  (None, 1)             101         activation_5[0][0]
        ====================================================================================================

    '''


    # Create the Sequential model
    model = Sequential()

    # set up cropping2D layer
    model.add(Cropping2D(cropping=((80, 20), (0, 0)), input_shape=(160, 320, 3), dim_ordering="tf"))

    # Normalizing layer
    model.add(Lambda(lambda image: image/127.5 - 1.))

    # Add a 5x5 conv layer
    model.add(Convolution2D(24, 5, 5, border_mode='valid', dim_ordering="tf"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), dim_ordering="tf"))
    model.add(Dropout(0.25))

    # Add a 5x5 convolution
    model.add(Convolution2D(36, 5, 5, border_mode='valid', dim_ordering="tf"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), dim_ordering="tf"))
    model.add(Dropout(0.25))

    # apply a 3x3 convolution
    model.add(Convolution2D(48, 3, 3, border_mode='valid', dim_ordering="tf"))
    model.add(Activation('relu'))

    # Add a flatten layer
    model.add(Flatten())

    # Add a fully connected layer
    model.add(Dense(1164))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # Add a fully connected layer
    model.add(Dense(100))
    model.add(Activation('relu'))


    # Add a fully connected layer
    model.add(Dense(1))

    model.summary()

    plot(model, to_file='model.png')

    return model

'''

def trans_image(image, steer, trans_range):
    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = steer + tr_x / trans_range * 2 * .2
    tr_y = 40 * np.random.uniform() - 40 / 2
    # tr_y = 0
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, Trans_M, (cols, rows))

    return image_tr, steer_ang
'''

def augmentImages(images, angles):
    aug_images, aug_angles = [], []
    for img, angle in zip(images, angles):
        img_flipped = np.fliplr(img)
        aug_images.append(img_flipped)
        aug_angles.append(-angle)

        # Translate the images by a small margin
        # Skipped as it is not improving the accuracy

        # Add shadows to random images
        # Skipped as it is not improving the accuracy

    return aug_images, aug_angles


def generator(samples, batch_size):
    num_samples = len(samples)
    path = 'data/'
    # create adjusted steering measurements for the side camera images
    correction = 0.25  # this is a parameter to tune

    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = path + batch_sample[0].strip()
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                left_image = cv2.imread(path + batch_sample[1].strip())
                left_angle = center_angle + correction
                images.append(left_image)
                angles.append(left_angle)

                right_image = cv2.imread(path + batch_sample[2].strip())
                right_angle = center_angle - correction
                images.append(right_image)
                angles.append(right_angle)

                aug_images, aug_angles = augmentImages([center_image, left_image, right_image], [center_angle, left_angle, right_angle])
                images.extend(aug_images)
                angles.extend(aug_angles)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

class printbatch(Callback):
    def on_batch_end(self, epoch, logs={}):
        print(logs)

def trainAndSaveModel():
    start_time = time.time()

    train_samples, val_samples = loadDataFiles('driving_log.csv')

    train_generator = generator(train_samples, batch_size=32)
    val_generator = generator(train_samples, batch_size=32)

    # Get the model
    model = getModel()


    # Configures the learning process and metrics
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    pb = printbatch()
    model.fit_generator(generator=train_generator,
                        samples_per_epoch=6*len(train_samples),
                        validation_data=val_generator,
                        nb_val_samples=6*len(val_samples),
                        nb_epoch=5, callbacks=[pb])

    end_time = time.time()
    print("Total time to train model = ", (end_time - start_time))

    model.save('model.h5')

trainAndSaveModel()
