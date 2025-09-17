# keras
import keras
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils import plot_model
import kt_utils

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = kt_utils.load_dataset()

X_train = X_train_orig / 255.0
X_test = X_test_orig / 255.0

Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print(X_train.shape)    # (600, 64, 64, 3)
print(Y_train.shape)    # (600, 1)
print(X_test.shape)     # (150, 64, 64, 3)
print(Y_test.shape)     # (150, 1)

# 建立模型
def model(input_shape):
    X_input = Input(input_shape)
    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((2, 2), name='max_pool')(X)

    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    model = Model(inputs=X_input, outputs=X, name='HappyModel')

    return model

def HappyModel(input_shape):
    X_input = Input(input_shape)
    X = ZeroPadding2D((1, 1))(X_input)
    X = Conv2D(8, kernel_size=(3, 3), strides=(1, 1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(X)

    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(16, kernel_size=(3, 3), strides=(1, 1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(X)

    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(32, kernel_size=(3, 3), strides=(1, 1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(X)

    X = Flatten()(X)
    Y = Dense(1, activation='sigmoid')(X)

    model = Model(inputs=X_input, outputs=Y, name='HappyModel')

    return model

# 创建模型
happyModel = HappyModel((64, 64, 3))
# 编译模型
happyModel.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999,epsilon=1e-08, decay=0.0),
                   loss='binary_crossentropy', metrics=['accuracy'])
# 训练模型
happyModel.fit(x=X_train, y=Y_train, batch_size=16, epochs=20)
# 测试/评估模型
preds = happyModel.evaluate(x=X_test, y=Y_test)
print()
print('Test loss:', preds[0])
print('Test accuracy:', preds[1])