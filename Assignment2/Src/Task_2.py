#!/usr/bin/env python
# coding: utf-8

# ### Data preprocessing

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from LabelTrans_Regression import label_transform

images_npy = '../Datasets/images.npy'
labels_npy = '../Datasets/labels.npy'
images = np.load(images_npy)
labels = np.load(labels_npy)


# In[32]:


print(labels[0])

gamma_transfer = exposure.adjust_gamma(images[0], 3)

plt.imshow(gamma_transfer, cmap ='gray')


# In[30]:


plt.imshow(images[0], cmap ='gray')


# In[2]:



enhance_images = exposure.adjust_gamma(images, 3)


# ---

# ### Basic Model Shape

# In[8]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, BatchNormalization
from keras import backend
import tensorflow as tf
import keras

model = Sequential()
model.add(Input(shape=(150, 150, 1)))
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

model.add(Flatten())
model.add(Dense(units=256,
                activation='relu',
                kernel_regularizer='l2'))
model.add(Dense(units=128,
                activation='relu',
                kernel_regularizer='l2'))
model.add(Dense(units=1, activation='linear'))

model.summary()


# ---

# ### Regression

# In[3]:


# Enhance
import numpy as np
from sklearn.model_selection import train_test_split
from Regression import regression, preprocessing
import matplotlib.pyplot as plt

batch_size = 32
epochs = 100
lr = 0.001

images_npy = '../Datasets/images.npy'
labels_npy = '../Datasets/labels.npy'
# images = np.load(images_npy)
images = enhance_images
labels = np.load(labels_npy)

x_train, x_test, y_train, y_test = train_test_split(images, labels, train_size=0.8)
x_train, x_test, input_shape = preprocessing(x_train, x_test)
y_train = y_train[:, 0] + y_train[:, 1] / 60  # single output labels
y_test = y_test[:, 0] + y_test[:, 1] / 60

regression_history, regression_his = regression(x_train, y_train, x_test, y_test, batch_size, epochs, lr, input_shape)


# In[4]:


regression_x_test, regression_y_test = x_test, y_test


# In[5]:


# Training plot with enhance
train_loss = regression_his.history['loss']
train_mse = regression_his.history['custom_mae']
val_loss = regression_his.history['val_loss']
val_mse = regression_his.history['val_custom_mae']
epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, 'orange', label='train_loss')
plt.plot(epochs, train_mse, label='train_custom_mae')
plt.plot(epochs, val_loss, label='val_loss')
plt.plot(epochs, val_mse, label='val_custom_mae')
plt.legend()
plt.show()


# In[18]:


regression_pred = regression_history.predict(x_test)
print('min predict:', np.min(regression_pred), 'max predict:', np.max(regression_pred))
print("y_predict", '   ', 'y_test')
for i in range(5):
    print(regression_pred[i], ' ', y_test[i])


# In[26]:


regerss_sum = 0
for i in range(3600):
    regerss_sum += min(abs(regression_pred[i]%1 - regression_y_test[i]%1), abs(1 + regression_pred[i]%1 - regression_y_test[i]%1))
regerss_sum / 3600 * 60


# In[31]:


def sincos_mse(y_true, y_pred):
    return np.sqrt((np.sum(np.square(y_true - y_pred)* [3600, 3600, 1, 1]) ) / 3601)
    
regression_pred_minutes = regression_pred % 1
regression_pred_hours = regression_pred - regression_pred_minutes
regression_y_minutes = regression_y_test % 1
regression_Y_hours = regression_y_test - regression_y_minutes
trans_regression_pred = label_transform([regression_pred_hours, regression_pred_minutes])
trans_regression_y_test = label_transform([regression_Y_hours, regression_y_minutes])

sincos_mse(trans_regression_pred, trans_regression_y_test)


# In[32]:


# Without Enhance
import numpy as np
from sklearn.model_selection import train_test_split
from Regression import regression, preprocessing
import matplotlib.pyplot as plt

batch_size = 32
epochs = 100
lr = 0.001

images_npy = '../Datasets/images.npy'
labels_npy = '../Datasets/labels.npy'
images = np.load(images_npy)
# images = enhance_images
labels = np.load(labels_npy)

x_train, x_test, y_train, y_test = train_test_split(images, labels, train_size=0.8)
x_train, x_test, input_shape = preprocessing(x_train, x_test)
y_train = y_train[:, 0] + y_train[:, 1] / 60  # single output labels
y_test = y_test[:, 0] + y_test[:, 1] / 60

regression_w_history, regression_w_his = regression(x_train, y_train, x_test, y_test, batch_size, epochs, lr, input_shape)


# In[33]:


regression_x_w_test, regression_y_w_test = x_test, y_test


# In[58]:


# Training plot without enhance
w_eH = regression_w_his
w_eHis = regression_w_history
train_loss = w_eH.history['loss']
train_mse = w_eH.history['custom_mae']
val_loss = w_eH.history['val_loss']
val_mse = w_eH.history['val_custom_mae']
epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, 'orange', label='train_loss')
plt.plot(epochs, train_mse, label='train_custom_mae')
plt.plot(epochs, val_loss, label='val_loss')
plt.plot(epochs, val_mse, label='val_custom_mae')
plt.legend()
plt.show()


# ---

# ### Classfication

# In[4]:


from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from Classfication import classfication, preprocessing

classes_num = 12 * 2
batch_size = 128
epochs = 100
lr = 0.001

images_npy = '../Datasets/images.npy'
labels_npy = '../Datasets/labels.npy'

# images = np.load(images_npy)
images = enhance_images 
labels = np.load(labels_npy)
x_train, x_test, y_train, y_test = train_test_split(images, labels, train_size=0.8)

x_train, x_test, input_shape = preprocessing(x_train, x_test)
y_train = y_train[:, 0] * 2 + (y_train[:, 1] * 2 / 60).astype('int32')
y_test = y_test[:, 0] * 2 + (y_test[:, 1] * 2 / 60).astype('int32')

y_train_trans2class = to_categorical(y_train, classes_num)
y_test_trans2class = to_categorical(y_test, classes_num)

classfication_history, classfication_his = classfication(x_train, y_train_trans2class, x_test, y_test_trans2class, batch_size, epochs, lr, classes_num, input_shape)


# In[5]:


acc = classfication_his.history['categorical_accuracy']
loss = classfication_his.history['loss']
accVal = classfication_his.history['val_categorical_accuracy']
lossVal = classfication_his.history['val_loss']
epochs = range(1, len(loss) + 1)

# plt.title('Accuracy and Loss')
# fig = plt.figure()
ax1 = plt.subplot()
ax2 = ax1.twinx()
ax1.set_ylabel('Loss', fontsize=15)
ax2.set_ylabel('Accuracy', fontsize=15)
ax1.set_xlabel('epochs', fontsize=15)
L1, = ax2.plot(epochs, acc, 'blue', label='Training Accuracy')
L2, = ax1.plot(epochs, loss, 'green', label='Training Loss')
L3, = ax2.plot(epochs, accVal, 'red', label='Test Accuracy')
L4, = ax1.plot(epochs, lossVal, 'orange', label='Test Loss')
plt.legend(handles = [L1, L2, L3, L4], loc = 'upper left', fontsize=10)


# #### 720 categories

# In[25]:


from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from Classfication import classfication, preprocessing

classes_num = 720
batch_size = 128
epochs = 100
lr = 0.001

images_npy = '../Datasets/images.npy'
labels_npy = '../Datasets/labels.npy'

# images = np.load(images_npy)
images = enhance_images 
labels = np.load(labels_npy)
x_train, x_test, y_train, y_test = train_test_split(images, labels, train_size=0.8)

x_train, x_test, input_shape = preprocessing(x_train, x_test)
y_train = y_train[:, 0] * 60  + (y_train[:, 1]).astype('int32')
y_test = y_test[:, 0] * 60 + (y_test[:, 1]).astype('int32')

y_train_trans2class = to_categorical(y_train, classes_num)
y_test_trans2class = to_categorical(y_test, classes_num)

history, his = classfication(x_train, y_train_trans2class, x_test, y_test_trans2class, batch_size, epochs, lr, classes_num, input_shape)


# In[26]:


acc = his.history['categorical_accuracy']
loss = his.history['loss']
accVal = his.history['val_categorical_accuracy']
lossVal = his.history['val_loss']
epochs = range(1, len(loss) + 1)

# plt.title('Accuracy and Loss')
# fig = plt.figure()
ax1 = plt.subplot()
ax2 = ax1.twinx()
ax1.set_ylabel('Loss', fontsize=15)
ax2.set_ylabel('Accuracy', fontsize=15)
ax1.set_xlabel('epochs', fontsize=15)
L1, = ax2.plot(epochs, acc, 'blue', label='Training Accuracy')
L2, = ax1.plot(epochs, loss, 'green', label='Training Loss')
L3, = ax2.plot(epochs, accVal, 'red', label='Test Accuracy')
L4, = ax1.plot(epochs, lossVal, 'orange', label='Test Loss')
plt.legend(handles = [L1, L2, L3, L4], loc = 'upper left', fontsize=10)


# ### Multi head
# 

# In[6]:


from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from MultiheadModel import multi_classfication, preprocessing, minutes_transform

batch_size = 128
epochs = 100
lr = 0.001
classes_num = 12

images_npy = '../Datasets/images.npy'
labels_npy = '../Datasets/labels.npy'

# images = np.load(images_npy)
images = enhance_images 
labels = np.load(labels_npy)
x_train, x_test, y_train, y_test = train_test_split(images, labels, train_size=0.8)
x_train, x_test, input_shape = preprocessing(x_train, x_test)

hours_train = y_train[:,0]
minutes_train = y_train[:,1]
hours_test = y_test[:,0]
minutes_test = y_test[:,1]

y_train_hours = to_categorical(hours_train, classes_num)
y_train_mintues = minutes_train / 60
y_test_hours = to_categorical(hours_test, classes_num)
y_test_mintues = minutes_test  / 60
# y_train_mintues = minutes_transform(minutes_train)
# y_test_mintues = minutes_transform(minutes_test)

y_train = [y_train_hours, y_train_mintues]
y_test = [y_test_hours, y_test_mintues]

history, his = multi_classfication(x_train, y_train, x_test, y_test, batch_size, epochs, lr, classes_num, input_shape)


# In[29]:


minutes_mse = his.history['minutes_mse'][1:]
minutes_loss = his.history['minutes_loss'][1:]
Vminutes_mse = his.history['val_minutes_mse'][1:]
Vminutes_loss = his.history['val_minutes_loss'][1:]
epochs = range(1, len(minutes_mse) + 1)
ax1 = plt.subplot()
ax1.set_ylabel('Loss', fontsize=15)
ax1.set_xlabel('epochs', fontsize=15)
ax1.plot(epochs, minutes_mse, 'orange', label='minutes_mse')
ax1.plot(epochs, minutes_loss, label='minutes_loss')
ax1.plot(epochs, Vminutes_mse, label='val_minutes_mse')
ax1.plot(epochs, Vminutes_loss, label='val_minutes_loss')
plt.legend()
plt.show()


# In[30]:


hours_acc = his.history['hours_categorical_accuracy']
hours_loss = his.history['hours_loss']

Vhours_acc = his.history['val_hours_categorical_accuracy']
Vhours_loss = his.history['val_hours_loss']
epochs = range(1, len(hours_loss) + 1)

# plt.title('Accuracy and Loss')
# fig = plt.figure()
ax1 = plt.subplot()
ax2 = ax1.twinx()
ax1.set_ylabel('Loss', fontsize=15)
ax2.set_ylabel('Accuracy', fontsize=15)
ax1.set_xlabel('epochs', fontsize=15)
L1, = ax2.plot(epochs, hours_acc, 'blue', label='hours_categorical_accuracy')
L2, = ax1.plot(epochs, hours_loss, 'green', label='hours_loss')
L3, = ax2.plot(epochs, Vhours_acc, 'red', label='val_hours_categorical_accuracy')
L4, = ax1.plot(epochs, Vhours_loss, 'orange', label='val_hours_loss')
plt.legend(handles = [L1, L2, L3, L4], loc = 'right', fontsize=10)


# In[9]:


pred = history.predict(x_test)


# In[10]:


print("y_predict", '   ', 'y_test')
for i in range(100):
    print(np.argmax(pred[0][i]), ' ', np.argmax(y_test[0][i]))


# In[11]:


# minutes difference and average minutes difference
sum = 0
for i in range(len(pred[1])):
    sum += abs(pred[1][i] - y_test[1][i])
    print(pred[1][i] , ' ', y_test[1][i] )

sum / len(pred[1]) * 60


# In[33]:


pred_hours = np.array([np.argmax(pred[0][i]) for i in range(3600)])
pred_minutes = np.array([pred[1][i] for i in range(3600)])
y_test_hours = np.array([np.argmax(y_test[0][i]) for i in range(3600)])
y_test_minutes = np.array([y_test[1][i] for i in range(3600)])


# In[34]:


def sincos_mse(y_true, y_pred):
    return np.sqrt((np.sum(np.square(y_true - y_pred)* [3600, 3600, 1, 1]) ) / 3601)
    
trans_pred = label_transform([pred_hours, pred_minutes])
trans_y_test = label_transform([y_test_hours, y_test_minutes])

sincos_mse(trans_pred, trans_y_test)


# ### Label trans regression

# In[35]:


from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from LabelTrans_Regression import label_trans_regression, preprocessing, label_transform

batch_size = 32
epochs = 100
lr = 0.001

images_npy = '../Datasets/images.npy'
labels_npy = '../Datasets/labels.npy'

# images = np.load(images_npy)
images = enhance_images
labels = np.load(labels_npy)

# labels_trans = label_transform(labels)

x_train, x_test, y_train, y_test = train_test_split(images, labels, train_size=0.8)

y_o_train = y_train
y_o_test = y_test

y_train = label_transform(y_train)
y_test = label_transform(y_test)

x_train, x_test, input_shape = preprocessing(x_train, x_test)

label_trans_regression_history, label_trans_regression_his = label_trans_regression(x_train, y_train, x_test, y_test, batch_size, epochs, input_shape)


# In[42]:


train_loss = label_trans_regression_his.history['loss']
train_mse = label_trans_regression_his.history['mae']
val_loss = label_trans_regression_his.history['val_loss']
val_mse = label_trans_regression_his.history['val_mae']
epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, 'orange', label='train_loss')
plt.plot(epochs, train_mse, label='train_mse')
plt.plot(epochs, val_loss, label='val_loss')
plt.plot(epochs, val_mse, label='val_mse')
plt.legend()
plt.show()


# In[44]:


label_trans_pred = label_trans_regression_history.predict(x_test)


# In[56]:


def sincos_mse(y_true, y_pred):
    return np.sqrt((np.sum(np.square(y_true - y_pred)* [3600, 3600, 1, 1]) ) / 3601)


# In[57]:


sincos_mse(label_trans_pred, y_test)


# ### re-Label trans  
# return real clock from sin&cos format

# In[47]:


def label_transform_map(h_list, m_list):
    hours_array = np.zeros(shape=(len(h_list), 2))
    minutes_array = np.zeros(shape=(len(m_list), 2))
    for i in range(len(h_list)):
        hour = h_list[i]
        hours_array[i] = (np.sin(hour*(1/6)*np.pi),np.cos(hour*(1/6)*np.pi))

    for i in range(len(m_list)):
        minute = m_list[i]
        minutes_array[i] = (np.sin(minute * (1/3) * np.pi), np.cos(minute * (1/3)*np.pi))

    return hours_array, minutes_array


# In[48]:


def distance(v1, v2):   
    dis_array = np.zeros(shape=(len(v2)))
    for i in range(len(v2)):
        dis_array[i] = float(np.sqrt(sum((v1 - v2[i])**2)))
    return dis_array


# In[49]:


hours_list = np.arange(0, 12).astype('float32')
minutes_list = np.arange(0, 60).astype('float32')

h_map, m_map = label_transform_map(hours_list, minutes_list)


# In[51]:


pred = label_trans_pred
hours = pred[:, 0:2]
minutes = pred[:, 2:]

def real_clock(hours, minutes):
    length = len(hours)
    result_list = np.zeros(shape=(length, 2))
    for i in range(length):
        hours_dis = distance(hours[i], h_map)
        minutes_dis = distance(minutes[i], m_map)

        h_min = np.argsort(hours_dis)[0:2]
        m_min = np.argsort(minutes_dis)[0]
        if (h_min[0] != 0) | (h_min[1] != 11):
            if m_min > 30:
                result_list[i] = (h_min[0]-1, m_min)
            else:
                result_list[i] = (h_min[0], m_min)
        else:
            if h_min[0] == 0:
                if m_min > 30:
                    result_list[i] = (11, m_min)
                else:
                    result_list[i] = (0, m_min)
            if h_min[0] == 11:
                if m_min > 30:
                    result_list[i] = (10, m_min)
                else:
                    result_list[i] = (11, m_min)
    return result_list


# In[55]:


result_list = real_clock(hours, minutes)
true_result = real_clock(y_test[:, 0:2], y_test[:, 2:])


# In[131]:


sum = 0
for i in range(3600):
    sum += min(abs(result_list[i, 1] - y_o_test[i, 1]), abs(60 + result_list[i, 1] - y_o_test[i, 1]))


# In[132]:


sum/3600


# In[119]:


for i in range(100):
    print(result_list[i], ' ', y_o_test[i])


# ---

# ### Other attempts

# In[46]:


# two classification Multi-head

from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from MultiheadModelAlex import multi_classficationAlex, preprocessing, minutes_transform
from skimage import exposure

batch_size = 128
epochs = 100
lr = 0.001
classes_num = 30

images_npy = '../Datasets/images.npy'
labels_npy = '../Datasets/labels.npy'

images = np.load(images_npy)
enhance_images = exposure.adjust_gamma(images, 3)
images = enhance_images 
labels = np.load(labels_npy)
x_train, x_test, y_train, y_test = train_test_split(images, labels, train_size=0.8)
x_train, x_test, input_shape = preprocessing(x_train, x_test)

hours_train = y_train[:,0]
minutes_train = y_train[:,1]
hours_test = y_test[:,0]
minutes_test = y_test[:,1]

minutes_train = (minutes_train / 2).astype('int32')
minutes_test = (minutes_test / 2).astype('int32')

y_train_hours = to_categorical(hours_train, 12)
y_test_hours = to_categorical(hours_test, 12)
y_train_mintues = to_categorical(minutes_train, classes_num)
y_test_mintues = to_categorical(minutes_test, classes_num)

y_train = [y_train_mintues, y_train_hours]
y_test = [y_test_mintues, y_test_hours]

history, his = multi_classficationAlex(x_train, y_train, x_test, y_test, batch_size, epochs, lr, classes_num, input_shape)


# In[2]:


# different fully connected layer

from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from MultiheadModelBeta import multi_classficationBeta, preprocessing, minutes_transform

batch_size = 128
epochs = 100
lr = 0.001
classes_num = 12

images_npy = '../Datasets/images.npy'
labels_npy = '../Datasets/labels.npy'

# images = np.load(images_npy)
images = enhance_images 
labels = np.load(labels_npy)
x_train, x_test, y_train, y_test = train_test_split(images, labels, train_size=0.8)
x_train, x_test, input_shape = preprocessing(x_train, x_test)

hours_train = y_train[:,0]
minutes_train = y_train[:,1]
hours_test = y_test[:,0]
minutes_test = y_test[:,1]

y_train_hours = to_categorical(hours_train, classes_num)
y_train_mintues = minutes_train / 60
y_test_hours = to_categorical(hours_test, classes_num)
y_test_mintues = minutes_test  / 60
# y_train_mintues = minutes_transform(minutes_train)
# y_test_mintues = minutes_transform(minutes_test)

y_train = [y_train_hours, y_train_mintues]
y_test = [y_test_hours, y_test_mintues]

history, his = multi_classfication(x_train, y_train, x_test, y_test, batch_size, epochs, lr, classes_num, input_shape)


# ---

# ### Model Structure

# In[3]:


# Multi head model
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, BatchNormalization
from keras import backend
import tensorflow as tf
import keras

input = Input(shape=(150, 150, 1))

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
normal3 = BatchNormalization()(pool3)
drop3 = Dropout(0.25)(normal3)

flatten = Flatten()(drop3)
Dense1 = Dense(256,
                activation='relu',
                kernel_regularizer='l2')(flatten)
Dense2 = Dense(128,
            activation='relu',
            kernel_regularizer='l2')(Dense1)
output_class = Dense(12, activation='softmax', name='hours')(Dense2)
# Regression
output_regress = Dense(1, activation='linear', name='minutes')(Dense2)
multi_model = Model(inputs = input, outputs = [output_class, output_regress])
multi_model.summary()

