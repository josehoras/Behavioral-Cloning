import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Lambda, Dense, Cropping2D, Dropout, BatchNormalization, Activation
from keras.layers import Conv2D
from keras import regularizers
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.callbacks import Callback, LearningRateScheduler
from sklearn.model_selection import train_test_split
import sklearn


def translate(im, translation, w, h):
    x_offset = translation * w * np.random.uniform(-1, 1)
    y_offset = translation * h * np.random.uniform(-1, 1)
    translation_mat = np.array([[1, 0, x_offset], [0, 1, y_offset]])
    im = cv2.warpAffine(im, translation_mat, (w, h))
    return im


# Hyperparameters
batch_size = 100
epochs = 5
lr = 1e-3
decay = 1


def lr_update(epoch, current_lr):
    if epoch == 0:
        new_rate = current_lr
    else:
        new_rate = current_lr * decay
    return new_rate


# Open data in csv file
filesave = 'lr=1e-3_dout.jpg'
graph_title = 'lr=1e-3_dout'
my_path = '/home/jose/GitHub/Behavioral_Cloning/'
my_path = '/app/'
data_path = ['data/', 'data2/']
samples = []
for path in data_path:
    csv_path = my_path + path
    with open(csv_path + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
           samples.append(line)

print("Total samples: ", len(samples))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


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
                correction = {0: 0, 1: 0.1, 2: -0.1}
                for i in range(3):
                    source_path = batch_sample[i]
                    file_path = my_path
                    for s in source_path.split('/')[-3:]:
                        file_path += '/' + s
                    image = cv2.imread(file_path)
                    # cv2 load images in BGR colorspace. Next line converts to RGB to be compatible with driving.py
                    # image = np.dstack((image[:,:,2], image[:,:,1], image[:,:,0]))
                    # print(np.array(image).shape)
                    images.append(image)
                    angle = float(batch_sample[3]) + correction[i]
                    angles.append(angle)
                    # Data augmentation
                    images.append(cv2.flip(image, 1))
                    angles.append(angle * -1.0)
                    # images.append(translate(image, 0.1, 320, 160))
                    # angles.append(angle)
                # Convert to numpy array as required by Keras
                x_batch = np.array(images)
                y_batch = np.array(angles)
                # print(x_batch.shape)
                # print(y_batch.shape)
                yield x_batch, y_batch


# Instantiate generators
train_gen = gen(train_samples, batch_size=batch_size)
val_gen = gen(validation_samples, batch_size=batch_size)
(next(train_gen))
# Define model
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
# model.add(Dropout(.5))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
# model.add(Dropout(.5))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
# model.add(Dropout(.5))
model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(Dropout(.5))
model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(Dropout(.5))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer=Adam(lr=lr))
history_object = model.fit_generator(generator=train_gen, steps_per_epoch=np.ceil(len(train_samples) / batch_size),
                                     validation_data=val_gen,
                                     validation_steps=np.ceil(len(validation_samples) / batch_size),
                                     epochs=epochs,
                                     callbacks=[LearningRateScheduler(lr_update, verbose=1)])
model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
fig, ax = plt.subplots( nrows=1, ncols=1 )
ax.plot(history_object.history['loss'], marker='o')
ax.plot(history_object.history['val_loss'], marker='o')
ax.set_ylim([0, 0.05])
# ax2.plot(acc_x, val_acc_series, marker='o', label="Validation")
ax.set_title(graph_title)
ax.set_ylabel('mean squared error loss')
ax.set_xlabel('epoch')
ax.legend(['training set', 'validation set'], loc='upper right')
plt.show()
fig.savefig(filesave)
