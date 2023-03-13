from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, BatchNormalization, Activation, MaxPool2D, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16


def conv_block(inputs, filters, pool=True):
    x = Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    if (pool == True):
        p = MaxPool2D((2,2))(x)
        return x, p
    

    else:
        return x

def build_unet(shape, num_classes, dropouts=0.07):
    inputs = Input(shape)

    """ENCODER"""
    x1, p1 = conv_block(inputs, 16, pool=True)
    p1 = Dropout(dropouts)(p1)
    x2, p2 = conv_block(p1, 32, pool=True)
    p2 = Dropout(dropouts)(p2)
    x3, p3 = conv_block(p2, 64, pool=True)
    p3 = Dropout(dropouts)(p3)
    x4, p4 = conv_block(p3, 128, pool=True)
    p4 = Dropout(dropouts)(p4)

    """BRIDGE"""
    b1 = conv_block(p4, 256, pool= True)

    """DECODER"""
    u1 = Conv2DTranspose(128, (3,3), strides=(2,2), padding="same")(b1[0])
    c1 = Concatenate()([u1,x4])
    c1 = Dropout(dropouts)(c1)
    x5 = conv_block(c1, 128, pool=True)

    u2 = Conv2DTranspose(64, (3,3), strides=(2,2), padding="same")(x5[0])
    c2 = Concatenate()([u2,x3])
    c2 = Dropout(dropouts)(c2)
    x6 = conv_block(c2, 64, pool=True)

    u3 = Conv2DTranspose(32, (3,3), strides=(2,2), padding="same")(x6[0])
    c3 = Concatenate()([u3,x2])
    c3 = Dropout(dropouts)(c3)
    x7 = conv_block(c3, 32, pool=True)

    u4 = Conv2DTranspose(16, (3,3), strides=(2,2), padding="same")(x7[0])
    c4 = Concatenate()([u4, x1])
    c4 = Dropout(dropouts)(c4)
    x8 = conv_block(c4, 16, pool=True)

    """OUTPUT LAYER"""

    outputs = Conv2D(num_classes, 1, padding="same", activation="softmax")(x8[0])

    



    return Model(inputs,outputs)



if __name__ == "__main__":
    model = VGG16(weights='imagenet', include_top = False, input_shape = (128,128,3))
    model = build_unet((128,128,3),3)
    model.summary()