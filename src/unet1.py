import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, UpSampling2D, Concatenate, Input, Dropout
from keras.models import Model

def conv_block(inputs, num_filters, dilation_rate):
    x = Conv2D(num_filters, 3, padding="same", dilation_rate=dilation_rate)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same", dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block(inputs, num_filters, dilation_rate):
    x = conv_block(inputs, num_filters, dilation_rate)
    p = MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(inputs, skip, num_filters, dilation_rate):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = UpSampling2D(size=(2, 2), interpolation='nearest')(inputs)
    x = Concatenate()([x, skip])
    x = conv_block(x, num_filters, dilation_rate)
    return x

def build_unet(input_shape, num_classes):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64, 3)
    s2, p2 = encoder_block(p1, 128, 2)
    s3, p3 = encoder_block(p2, 256, 2)
    s4, p4 = encoder_block(p3, 512, 1)

    b1 = conv_block(p4, 1024, 1)

    d1 = decoder_block(b1, s4, 512, 1)
    d2 = decoder_block(d1, s3, 256, 1)
    d3 = decoder_block(d2, s2, 128, 1)
    d4 = decoder_block(d3, s1, 64, 1)

    outputs = Conv2D(num_classes, 1, padding="same", activation="softmax")(d4)

    model = Model(inputs, outputs)
    return model


if __name__ == "__main__":
    input_shape = (512, 512, 3)
    model = build_unet(input_shape, 11)
    model.summary()
