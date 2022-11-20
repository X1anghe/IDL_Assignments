from keras.utils.np_utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, BatchNormalization
from keras import backend
import keras
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def preprocessing(train_data, test_data):
    img_rows, img_cols = train_data.shape[1], train_data.shape[2]
    if backend.image_data_format() == 'channels_first':
        train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols)
        test_data = test_data.reshape(test_data.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        train_data = train_data.reshape(train_data.shape[0], img_rows, img_cols, 1)
        test_data = test_data.reshape(test_data.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1) 

    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')
    train_data /= 255
    test_data /= 255

    return train_data, test_data, input_shape

def custom_mae(y_true, y_pred):
    abs_diff = tf.math.abs(y_true - y_pred)
    min_diff = tf.minimum(tf.math.add(12.0, abs_diff), abs_diff)
    return tf.reduce_mean(min_diff)

def classfication(x_train, y_train, x_test, y_test, batch_size, epochs, lr, num_classes):

    input = Input(shape=input_shape)

    conv1 = Conv2D(32, (3, 3), 
                    activation='relu', 
                    kernel_regularizer='l2')(input)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    normal1 = BatchNormalization()(pool1)
    drop1 = Dropout(0.25)(normal1)

    conv3 = Conv2D(64, (3, 3), 
                    activation='relu',
                    kernel_regularizer='l2')(drop1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv3)
    normal2 = BatchNormalization()(pool2)
    drop2 = Dropout(0.25)(normal2)

    conv3 = Conv2D(128, (3, 3), 
                    activation='relu',
                    kernel_regularizer='l2')(drop2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    normal2 = BatchNormalization()(pool3)
    drop3 = Dropout(0.25)(normal2)

    flatten = Flatten()(drop3)
    Dense1 = Dense(256,
                    activation='relu',
                    kernel_regularizer='l2')(flatten)
    Dense2 = Dense(128,
                activation='relu',
                kernel_regularizer='l2')(Dense1)

    # Multiclass
    output_class = Dense(num_classes, activation='softmax', name='hours')(Dense2)
    # Regression
    output_regress = Dense(1, activation='linear', name='minutes')(Dense2)

    multi_model = Model(inputs = input, outputs = [output_class, output_regress])

    multi_model.compile(loss=['categorical_crossentropy', 'mae'],
                optimizer=keras.optimizers.Adam(learning_rate=lr),
                metrics=['categorical_accuracy', 'mse'])

    his = multi_model.fit(x_train, [y_train[0], y_train[1]],
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, [y_test[0], y_test[1]]))

              
    score = multi_model.evaluate(x_test, [y_test[0], y_test[1]], verbose=0)
    print('Test loss:', score[0])
    print('Test mse:', score[1])
    return multi_model, his



images_npy = '../Datasets/images.npy'
labels_npy = '../Datasets/labels.npy'

images = np.load(images_npy)
labels = np.load(labels_npy)
x_train, x_test, y_train, y_test = train_test_split(images, labels, train_size=0.8)
x_train, x_test, input_shape = preprocessing(x_train, x_test)

classes_num = 12

hours_train = y_train[:,0]
minutes_train = y_train[:,1]
hours_test = y_test[:,0]
minutes_test = y_test[:,1]

y_train_hours = to_categorical(hours_train, classes_num)
y_train_mintues = minutes_train / 60
y_test_hours = to_categorical(hours_test, classes_num)
y_test_mintues = minutes_test / 60

y_train = [y_train_hours, y_train_mintues]
y_test = [y_test_hours, y_test_mintues]

history, his = classfication(x_train, y_train, x_test, y_test, 128, 100, 0.001, classes_num)


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