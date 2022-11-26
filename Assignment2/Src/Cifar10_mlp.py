'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
import matplotlib.pyplot as plt
import keras

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

    x_train = x_train.reshape(50000, 1024 * 3)
    x_test = x_test.reshape(10000, 1024 * 3)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Dense(512,
                    kernel_initializer = initializer,
                    kernel_regularizer = regularizer,
                    activation = activation,
                    input_shape=(1024 * 3, )))
    if dropout:
        model.add(Dropout(0.2))
    model.add(Dense(512,
                    kernel_regularizer = regularizer,
                    activation = activation))
    if dropout:
        model.add(Dropout(0.2))
    model.add(Dense(num_classes, 
                    kernel_regularizer = regularizer,
                    activation='softmax'))

    model.compile(loss='categorical_crossentropy',
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