'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt


def network(x_train, y_train,
            x_test, y_test,
            batch_size,
            num_classes,
            epochs,
            initializer,
            regularizer,
            activation,
            dropout,
            lr,
            plot_num):

    # input image dimensions
    img_rows, img_cols = x_train.shape[1], x_train.shape[2]

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    # print('x_train shape:', x_train.shape)
    # print(x_train.shape[0], 'train samples')
    # print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    kernel_initializer = initializer,
                    kernel_regularizer = regularizer,
                    activation = activation,
                    input_shape=input_shape))

    model.add(Conv2D(64, (3, 3), 
                    kernel_regularizer = regularizer,
                    activation = activation))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    if dropout:
        model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, 
                    kernel_regularizer = regularizer,
                    activation = activation))
    if dropout:
        model.add(Dropout(0.5))
    model.add(Dense(num_classes, 
                    kernel_regularizer = regularizer,
                    activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer = keras.optimizers.SGD(learning_rate=lr),
                metrics=['accuracy'])

    history = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    score1 = model.evaluate(x_train, y_train, verbose=0)

    print('Train loss:', score1[0])
    print('Train accuracy:', score1[1])
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # accVal = history.history['val_accuracy']
    lossVal = history.history['val_loss']
    return lossVal
    
    # acc = history.history['accuracy']
    # loss = history.history['loss']
    # accVal = history.history['val_accuracy']
    # lossVal = history.history['val_loss']
    # epochs = range(1, len(acc) + 1)
    
    # # plt.title('Accuracy and Loss')
    # # fig = plt.figure()
    # ax1 = plt.subplot()
    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Loss', fontsize=15)
    # ax1.set_ylabel('Accuracy', fontsize=15)
    # ax1.set_xlabel('epochs', fontsize=15)
    # L1, = ax1.plot(epochs, acc, 'green', label='Training Accuracy')
    # L2, = ax2.plot(epochs, loss, 'blue', label='Training Loss')
    # L3, = ax1.plot(epochs, accVal, 'orange', label='Test Accuracy')
    # L4, = ax2.plot(epochs, lossVal, 'red', label='Test Loss')
    # plt.legend(handles = [L1, L2, L3, L4], loc = 'upper right', fontsize=10)
    # plt.savefig('../fig/'+'Basic_minist_cnn'+'.jpg')
    # plt.show()