import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Input, AveragePooling2D
import keras
from keras.datasets import cifar10
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.utils import np_utils

classes_num = 10
batch_size = 32
epochs_num = 200


def resnet_block(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu'):
    x = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    if activation:
        x = Activation('relu')(x)
    return x


# 20 layer Resnet
def resnet(input_shape):
    inputs = Input(shape=input_shape)  # Input layer

    # first layer
    x = resnet_block(inputs)
    print('layer1,xshape:', x.shape)

    for i in range(6):
        a = resnet_block(inputs=x)
        b = resnet_block(inputs=a, activation=None)
        x = keras.layers.add([x, b])
        x = Activation('relu')(x)
    # out：32*32*16

    for i in range(6):
        if i == 0:
            a = resnet_block(inputs=x, strides=2, num_filters=32)
        else:
            a = resnet_block(inputs=x, num_filters=32)
        b = resnet_block(inputs=a, activation=None, num_filters=32)
        if i == 0:
            x = Conv2D(32, kernel_size=3, strides=2, padding='same',
                       kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
        x = keras.layers.add([x, b])
        x = Activation('relu')(x)
    # out:16*16*32

    for i in range(6):
        if i == 0:
            a = resnet_block(inputs=x, strides=2, num_filters=64)
        else:
            a = resnet_block(inputs=x, num_filters=64)

        b = resnet_block(inputs=a, activation=None, num_filters=64)
        if i == 0:
            x = Conv2D(64, kernel_size=3, strides=2, padding='same',
                       kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
        x = keras.layers.add([x, b])
        x = Activation('relu')(x)
    # out:8*8*64

    x = AveragePooling2D(pool_size=2)(x)
    # out:4*4*64
    y = Flatten()(x)
    # out:1024
    outputs = Dense(10, activation='softmax',
                    kernel_initializer='he_normal')(y)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# changing lr
def lr_sch(epoch):
    if epoch <50:
        return 1e-3
    if 50<=epoch<100:
        return 1e-4
    if epoch>=100:
        return 1e-5


def train():

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    y_train = keras.utils.np_utils.to_categorical(y_train, classes_num)
    y_test = keras.utils.np_utils.to_categorical(y_test, classes_num)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # train_datagan = ImageDataGenerator(rescale=1./255, rotation_range=10, width_shift_range=0.1,
    # height_shift_range=0.1, fill_mode='wrap')
    # test_datagen = ImageDataGenerator(rescale=1./255)

    model = resnet((32, 32, 3))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    checkpoint = ModelCheckpoint(filepath='./resnet_model/cifar10_resnet_ckpt.h5', monitor='val_accuracy', verbose=1,
                                 save_best_only=True)
    lr_scheduler = LearningRateScheduler(lr_sch)
    lr_reducer = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, mode='max', min_lr=1e-3)
    callbacks = [checkpoint, lr_scheduler, lr_reducer]
    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs_num, validation_data=(x_test, y_test),
                     verbose=1, callbacks=callbacks)

    # hist = model.fit_generator(train_datagan.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch = 8000, epochs = epochs_num, validation_data=(x_test,y_test), verbose=1, callbacks=callbacks)

    hist_dict = hist.history
    print("train acc:")
    print(hist_dict['accuracy'])
    print("validation acc:")
    print(hist_dict['val_accuracy'])

    train_acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    epochs = range(1, len(train_acc) + 1)
    plt.plot(epochs, train_acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig("accuracy.png")
    plt.figure()
    plt.plot(epochs, train_loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig("loss.png")
