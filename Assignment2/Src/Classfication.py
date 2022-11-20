from keras.utils.np_utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, BatchNormalization
from keras import backend
import keras
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def preprocessing(train_data, test_data):
    img_rows, img_cols = train_data.shape[1], train_data.shape[2]
    if backend.image_data_format() == 'channels_first':
        train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols)
        test_data = test_data.reshape(test_data.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        train_data = train_data.reshape(train_data.shape[0], img_rows, img_cols, 1)
        test_data = test_data.reshape(test_data.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)  #1在前在后

    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')
    train_data /= 255
    test_data /= 255

    return train_data, test_data, input_shape


def classfication(x_train, y_train, x_test, y_test, batch_size, epochs, lr, num_classes):

    model = Sequential()
    # Convolutional layers
    model.add(Input(shape=input_shape))
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     activation='relu',
                     kernel_regularizer='l2'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64,
                     kernel_size=(3, 3),
                     activation='relu',
                     kernel_regularizer='l2'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=128,
                     kernel_size=(3, 3),
                     activation='relu',
                     kernel_regularizer='l2'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(units=256,
                    activation='relu',
                    kernel_regularizer='l2'))
    model.add(Dense(units=128,
                    activation='relu',
                    kernel_regularizer='l2'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(learning_rate=lr),
                metrics=['categorical_accuracy'])

    his = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test mse:', score[1])
    return model, his



images_npy = '../Datasets/images.npy'
labels_npy = '../Datasets/labels.npy'

images = np.load(images_npy)
labels = np.load(labels_npy)
x_train, x_test, y_train, y_test = train_test_split(images, labels, train_size=0.8)

classes_num = 12 * 2

x_train, x_test, input_shape = preprocessing(x_train, x_test)
y_train = y_train[:, 0] * 2 + (y_train[:, 1] * 2 / 60).astype('int32')
y_test = y_test[:, 0] * 2 + (y_test[:, 1] * 2 / 60).astype('int32')
# y_train = y_train[:, 0] * 20 + (y_train[:, 1] * 20 / 60).astype('int32')
# y_test = y_test[:, 0] * 20 + (y_test[:, 1] * 20 / 60).astype('int32')

y_train_trans2class = to_categorical(y_train, classes_num)
y_test_trans2class = to_categorical(y_test, classes_num)

# y_train = y_train[:, 0] + y_train[:, 1] / 60  #时 分
# y_test = y_test[:, 0] + y_test[:, 1] / 60

history, his = classfication(x_train, y_train_trans2class, x_test, y_test_trans2class, 128, 100, 0.001, classes_num)


acc = his.history['loss']
loss = his.history['categorical_accuracy']
accVal = his.history['val_loss']
lossVal = his.history['val_categorical_accuracy']
epochs = range(1, len(acc) + 1)
plt.title('Accuracy and Loss')
plt.plot(epochs, acc, 'green', label='Training acc')
plt.plot(epochs, loss, 'blue', label='Training loss')
plt.plot(epochs, accVal, 'orange', label='Validation acc')
plt.plot(epochs, lossVal, 'red', label='Validation loss')
plt.legend()
plt.show()