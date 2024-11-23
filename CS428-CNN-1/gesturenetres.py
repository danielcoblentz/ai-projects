from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Add
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.initializers import HeNormal


class GestureNetRes:
    @staticmethod
    def build(width, height, depth, classes):
        initializer = HeNormal()  # Initialize weights with HeNormal

        # Input layer
        inputs = Input(shape=(height, width, depth))
        x = inputs

        # First CONV => RELU => CONV => RELU => Pool layer set with a shortcut connection
        shortcut = Conv2D(16, (1, 1), padding="same", kernel_initializer=initializer)(x)
        x = Conv2D(16, (7, 7), padding="same", kernel_initializer=initializer)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Conv2D(16, (7, 7), padding="same", kernel_initializer=initializer)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Add()([x, shortcut])
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # Second CONV => RELU => CONV => RELU => Pool layer set with a shortcut connection
        shortcut = Conv2D(32, (1, 1), padding="same", kernel_initializer=initializer)(x)
        x = Conv2D(32, (3, 3), padding="same", kernel_initializer=initializer)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, (3, 3), padding="same", kernel_initializer=initializer)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Add()([x, shortcut])
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # Third CONV layer with shortcut connection
        shortcut = Conv2D(64, (1, 1), padding="same", kernel_initializer=initializer)(x)
        x = Conv2D(64, (3, 3), padding="same", kernel_initializer=initializer)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Add()([x, shortcut])
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # First and only set of FC RELU layers
        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
       
        
        # Output layer (softmax classifier)
        x = Dense(classes, activation="softmax")(x)

        # Create the model
        model = Model(inputs=inputs, outputs = outputs)
        return model
