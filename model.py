import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Lambda, Dense, Cropping2D, Dropout
from keras.layers import Conv2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import sklearn

# Open data in csv file
samples = []
data_path = ['data/']  # , 'data2/']
for path in data_path:
    with open('../' + path + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
print("Total samples: ", len(samples))
# Split samples in training (80%) and validation (20%) samples
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# Python generator to feed the data to Keras fit_generator to alleviate memory constrains
def gen(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                # store the three sets of images with corresponding angles measurement
                correction = {0: 0, 1: 0.2, 2: -0.2}
                for i in range(3):
                    source_path = batch_sample[i]
                    image = cv2.imread(source_path)
                    # cv2 load images in BGR colorspace. Next line converts to RGB to be compatible with driving.py
                    image = np.dstack((image[:, :, 2], image[:, :, 1], image[:, :, 0]))
                    images.append(image)
                    angle = float(batch_sample[3]) + correction[i]
                    angles.append(angle)
                    # Add data augmentation by flipping the image
                    images.append(cv2.flip(image, 1))
                    angles.append(angle * -1.0)
                # Convert to numpy array as required by Keras
                x_batch = np.array(images)
                y_batch = np.array(angles)
                yield x_batch, y_batch


# Hyperparameters
batch_size = 64
learning_rate = 1e-3
# Instantiate generators
train_gen = gen(train_samples, batch_size=batch_size)
val_gen = gen(validation_samples, batch_size=batch_size)

# Define model
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((65, 25), (0, 0))))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
history_object = model.fit_generator(generator=train_gen, steps_per_epoch=np.ceil(len(train_samples) / batch_size),
                                     validation_data=val_gen,
                                     validation_steps=np.ceil(len(validation_samples) / batch_size),
                                     epochs=3)
# Save model file
model.save('model.h5')

### plot the training and validation loss for each epoch (to run on my computer)
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()


### plot the training and validation loss for each epoch
# fig, ax = plt.subplots( nrows=1, ncols=1 )
# ax.plot(history_object.history['loss'], marker='o')
# ax.plot(history_object.history['val_loss'], marker='o')
# ax.set_ylim([0, 0.05])
# ax.set_title(graph_title)
# ax.set_ylabel('mean squared error loss')
# ax.set_xlabel('epoch')
# ax.legend(['training set', 'validation set'], loc='upper right')
# plt.show()
# fig.savefig(filesave)
