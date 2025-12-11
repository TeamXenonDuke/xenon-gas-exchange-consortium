"""VNet architecture.

Diogo Amorim, 2018-07-10
V-Net implementation in Keras 2
https://arxiv.org/pdf/1606.04797.pdf
"""
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Concatenate, Add


def BatchNormalization(name: str) -> tf.keras.layers.Layer:
    """Batch normalization.

    Args:
        name: str name of the layer.
    """
    return tf.keras.layers.BatchNormalization(name=name)


def Deconvolution3D(inputs, filters, kernel_size, subsample, name):

    strides = tuple(subsample)

    x = tf.keras.layers.Conv3DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=None,
        padding="same",
        data_format="channels_last",
        use_bias=True,
        # kernel_initializer=tf.keras.glorot_uniform_initializer(),
        kernel_initializer="glorot_uniform",
        bias_initializer=tf.zeros_initializer(),
        name=name,
        bias_regularizer=None,
    )(inputs)

    return x


def downward_layer(input_layer, n_convolutions, n_output_channels, number):
    inl = input_layer

    for nnn in range(n_convolutions):
        inl = tf.keras.layers.Conv3D(
            filters=(n_output_channels // 2),
            kernel_size=5,
            padding="same",
            kernel_initializer="he_normal",
            name="conv_" + str(number) + "_" + str(nnn),
        )(inl)
        inl = BatchNormalization(name="batch_" + str(number) + "_" + str(nnn))(inl)
        inl = tf.keras.layers.ReLU(name="relu_" + str(number) + "_" + str(nnn))(inl)

    add_l = Add(name=f"add_{number}")([inl, input_layer])
    downsample = tf.keras.layers.Conv3D(
        filters=n_output_channels,
        kernel_size=2,
        strides=2,
        padding="same",
        kernel_initializer="he_normal",
        name="conv_" + str(number) + "_" + str(nnn + 1),
    )(add_l)
    downsample = BatchNormalization(name="batch_" + str(number) + "_" + str(nnn + 1))(
        downsample
    )
    downsample = tf.keras.layers.ReLU(name="relu_" + str(number) + "_" + str(nnn + 1))(
        downsample
    )
    return downsample, add_l


def upward_layer(input0, input1, n_convolutions, n_output_channels, number):
    merged = Concatenate(axis=4, name=f"merge_{number}")([input0, input1])
    inl = merged
    for nnn in range(n_convolutions):

        inl = tf.keras.layers.Conv3D(
            (n_output_channels * 4),
            kernel_size=5,
            padding="same",
            kernel_initializer="he_normal",
            name="conv_" + str(number) + "_" + str(nnn),
        )(inl)
        inl = BatchNormalization(name="batch_" + str(number) + "_" + str(nnn))(inl)
        inl = tf.keras.layers.ReLU(name="relu_" + str(number) + "_" + str(nnn))(inl)

    add_l = Add(name=f"add_{number}")([inl, merged])
    upsample = Deconvolution3D(
        add_l,
        n_output_channels,
        (2, 2, 2),
        subsample=(2, 2, 2),
        name="dconv_" + str(number) + "_" + str(nnn + 1),
    )
    upsample = BatchNormalization(name="batch_" + str(number) + "_" + str(nnn + 1))(
        upsample
    )
    return tf.keras.layers.ReLU(name="relu_" + str(number) + "_" + str(nnn + 1))(
        upsample
    )


def vnet_saved(
    input_size=(128, 128, 128, 1),
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"],
):
    # loss='categorical_crossentropy', metrics=['categorical_accuracy']):
    # Layer 1
    inputs = tf.keras.layers.Input(input_size)
    conv1 = tf.keras.layers.Conv3D(
        16,
        kernel_size=5,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
        name="conv_1",
    )(inputs)
    conv1 = BatchNormalization(name="batch_1")(conv1)
    conv1 = tf.keras.layers.ReLU(name="relu_1")(conv1)
    repeat1 = Concatenate(axis=-1, name="repeat1")(16 * [inputs])
    add1 = Add(name="add1")([conv1, repeat1])
    down1 = tf.keras.layers.Conv3D(
        32,
        kernel_size=(2, 2, 2),
        strides=2,
        padding="same",
        kernel_initializer="he_normal",
        name="down_1",
    )(add1)
    down1 = BatchNormalization(name="batch_1_2")(down1)
    down1 = tf.keras.layers.ReLU(name="relu_1_2")(down1)

    # Layer 2,3,4
    down2, add2 = downward_layer(down1, 2, 64, 2)
    down3, add3 = downward_layer(down2, 3, 128, 3)
    down4, add4 = downward_layer(down3, 3, 256, 4)

    # Layer 5
    # !Mudar kernel_size=(5, 5, 5) quando imagem > 64!
    conv_5_1 = tf.keras.layers.Conv3D(
        256,
        kernel_size=(5, 5, 5),
        padding="same",
        kernel_initializer="he_normal",
        name="conv_5_1",
    )(down4)
    conv_5_1 = BatchNormalization(name="batch_5_1")(conv_5_1)
    conv_5_1 = tf.keras.layers.ReLU(name="relu_5_1")(conv_5_1)
    conv_5_2 = tf.keras.layers.Conv3D(
        256,
        kernel_size=(5, 5, 5),
        padding="same",
        kernel_initializer="he_normal",
        name="conv_5_2",
    )(conv_5_1)
    conv_5_2 = BatchNormalization(name="batch_5_2")(conv_5_2)
    conv_5_2 = tf.keras.layers.ReLU(name="relu_5_2")(conv_5_2)
    conv_5_3 = tf.keras.layers.Conv3D(
        256,
        kernel_size=(5, 5, 5),
        padding="same",
        kernel_initializer="he_normal",
        name="conv_5_3",
    )(conv_5_2)
    conv_5_3 = BatchNormalization(name="batch_5_3")(conv_5_3)
    conv_5_3 = tf.keras.layers.ReLU(name="relu_5_3")(conv_5_3)
    add5 = Add(name="add5")([conv_5_3, down4])
    upsample_5 = Deconvolution3D(
        add5, 128, (2, 2, 2), subsample=(2, 2, 2), name="dconv_5"
    )

    upsample_5 = BatchNormalization(name="batch_5_4")(upsample_5)
    upsample_5 = tf.keras.layers.ReLU(name="relu_5_4")(upsample_5)

    # Layer 6,7,8
    upsample_6 = upward_layer(upsample_5, add4, 3, 64, 6)
    upsample_7 = upward_layer(upsample_6, add3, 3, 32, 7)
    upsample_8 = upward_layer(upsample_7, add2, 2, 16, 8)

    # Layer 9
    merged_9 = Concatenate(axis=4, name="merge_9")([upsample_8], add1)
    conv_9_1 = tf.keras.layers.Conv3D(
        32,
        kernel_size=(5, 5, 5),
        padding="same",
        kernel_initializer="he_normal",
        name="conv_9_1",
    )(merged_9)
    conv_9_1 = BatchNormalization(name="batch_9_1")(conv_9_1)
    conv_9_1 = tf.keras.layers.ReLU(name="relu_9_1")(conv_9_1)
    add_9 = Add(name="add_9")([conv_9_1, merged_9])
    # conv_9_2 = tf.keras.layers.Conv3D(1, kernel_size=(1, 1, 1), padding='same', kernel_initializer='he_normal')(add_9)
    conv_9_2 = tf.keras.layers.Conv3D(
        1,
        kernel_size=(1, 1, 1),
        padding="same",
        kernel_initializer="he_normal",
        name="conv_9_2",
    )(add_9)
    conv_9_2 = BatchNormalization(name="batch_9_2")(conv_9_2)
    conv_9_2 = tf.keras.layers.ReLU(name="relu_9_2")(conv_9_2)

    # softmax = Softmax()(conv_9_2)
    sigmoid_v = tf.keras.layers.Conv3D(
        1,
        kernel_size=(1, 1, 1),
        padding="same",
        kernel_initializer="he_normal",
        name="conv_sigm_1",
    )(conv_9_2)
    sigmoid_v = BatchNormalization(name="batch_sigm_1")(sigmoid_v)
    sigmoid_v = tf.keras.layers.Activation(activation="sigmoid")(sigmoid_v)

    model = Model(inputs=inputs, outputs=sigmoid_v)
    # model = Model(inputs=inputs, outputs=softmax)

    return model


def vnet(
    input_size=(128, 128, 128, 1),
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"],
):
    # loss='categorical_crossentropy', metrics=['categorical_accuracy']):
    # Layer 1
    input_gas = tf.keras.layers.Input(input_size)

    conv1 = tf.keras.layers.Conv3D(
        16,
        kernel_size=5,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
        name="conv_1",
    )(input_gas)
    conv1 = BatchNormalization(name="batch_1")(conv1)
    conv1 = tf.keras.layers.ReLU(name="relu_1")(conv1)
    repeat1 = Concatenate(axis=-1, name="repeat1")(16 * [input_gas])
    add1 = Add(name="add1")([conv1, repeat1])
    down1 = tf.keras.layers.Conv3D(
        32,
        kernel_size=(2, 2, 2),
        strides=2,
        padding="same",
        kernel_initializer="he_normal",
        name="down_1",
    )(add1)
    down1 = BatchNormalization(name="batch_1_2")(down1)
    down1 = tf.keras.layers.ReLU(name="relu_1_2")(down1)

    # Layer 2,3,4
    down2, add2 = downward_layer(down1, 2, 64, 2)
    down3, add3 = downward_layer(down2, 3, 128, 3)
    down4, add4 = downward_layer(down3, 3, 256, 4)

    # Layer 5
    conv_5_1 = tf.keras.layers.Conv3D(
        256,
        kernel_size=(5, 5, 5),
        padding="same",
        kernel_initializer="he_normal",
        name="conv_5_1",
    )(down4)
    conv_5_1 = BatchNormalization(name="batch_5_1")(conv_5_1)
    conv_5_1 = tf.keras.layers.ReLU(name="relu_5_1")(conv_5_1)
    conv_5_2 = tf.keras.layers.Conv3D(
        256,
        kernel_size=(5, 5, 5),
        padding="same",
        kernel_initializer="he_normal",
        name="conv_5_2",
    )(conv_5_1)
    conv_5_2 = BatchNormalization(name="batch_5_2")(conv_5_2)
    conv_5_2 = tf.keras.layers.ReLU(name="relu_5_2")(conv_5_2)
    conv_5_3 = tf.keras.layers.Conv3D(
        256,
        kernel_size=(5, 5, 5),
        padding="same",
        kernel_initializer="he_normal",
        name="conv_5_3",
    )(conv_5_2)
    conv_5_3 = BatchNormalization(name="batch_5_3")(conv_5_3)
    conv_5_3 = tf.keras.layers.ReLU(name="relu_5_3")(conv_5_3)
    add5 = Add(name="add5")([conv_5_3, down4])
    upsample_5 = Deconvolution3D(
        add5, 128, (2, 2, 2), subsample=(2, 2, 2), name="dconv_5"
    )

    upsample_5 = BatchNormalization(name="batch_5_4")(upsample_5)
    upsample_5 = tf.keras.layers.ReLU(name="relu_5_4")(upsample_5)

    # Layer 6,7,8
    upsample_6 = upward_layer(upsample_5, add4, 3, 64, 6)
    upsample_7 = upward_layer(upsample_6, add3, 3, 32, 7)
    upsample_8 = upward_layer(upsample_7, add2, 2, 16, 8)

    # Layer 9
    merged_9 = Concatenate(axis=4, name="merge_9")([upsample_8, add1])
    conv_9_1 = tf.keras.layers.Conv3D(
        32,
        kernel_size=(5, 5, 5),
        padding="same",
        kernel_initializer="he_normal",
        name="conv_9_1",
    )(merged_9)
    conv_9_1 = BatchNormalization(name="batch_9_1")(conv_9_1)
    conv_9_1 = tf.keras.layers.ReLU(name="relu_9_1")(conv_9_1)
    add_9 = Add(name="add_9")([conv_9_1, merged_9])
    # conv_9_2 = tf.keras.layers.Conv3D(1, kernel_size=(1, 1, 1), padding='same', kernel_initializer='he_normal')(add_9)
    conv_9_2 = tf.keras.layers.Conv3D(
        1,
        kernel_size=(1, 1, 1),
        padding="same",
        kernel_initializer="he_normal",
        name="conv_9_2",
    )(add_9)
    conv_9_2 = BatchNormalization(name="batch_9_2")(conv_9_2)
    conv_9_2 = tf.keras.layers.ReLU(name="relu_9_2")(conv_9_2)

    # softmax = Softmax()(conv_9_2)
    sigmoid_v = tf.keras.layers.Conv3D(
        1,
        kernel_size=(1, 1, 1),
        padding="same",
        kernel_initializer="he_normal",
        name="conv_sigm_1",
    )(conv_9_2)
    sigmoid_v = BatchNormalization(name="batch_sigm_1")(sigmoid_v)
    sigmoid_v = tf.keras.layers.Activation(activation="sigmoid")(sigmoid_v)

    model = Model(inputs=input_gas, outputs=sigmoid_v)
    # model = Model(inputs=inputs, outputs=softmax)

    return model
