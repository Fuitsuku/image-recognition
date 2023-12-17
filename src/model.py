from keras import layers
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam

def createModel( img_shape ):
    # Create Input + Conv + Pooling + Flattening Layers
    inputL = layers.Input( shape = img_shape ) # Input Layer (Image Shape = (24, 72, 1))
    convL1 = layers.Conv2D( 16, (3,3), padding = 'same', activation = 'relu' )( inputL ) # Convolution Layer 1. Contains 16 filters, 3 by 3, same padding and uses RElu for Activation
    poolL1 = layers.MaxPooling2D( padding = 'same' )( convL1 ) # Max Pooling Layer 1 -> Shrinks Dimensions of Data
    convL2 = layers.Conv2D( 16, (3,3), padding = 'same', activation = 'relu' )( poolL1 ) # Convolution Layer 2. Contains 16 filters, 3 by 3, same padding and uses RElu for Activation
    poolL2 = layers.MaxPooling2D( padding = 'same' )( convL2 ) # Max Pooling Layer 2 -> Shrinks Dimensions of Data
    convL3 = layers.Conv2D( 16, (3,3), padding = 'same', activation = 'relu' )( poolL2 ) # Convolution Layer 3. Contains 16 filters, 3 by 3, same padding and uses RElu for Activation
    batchNormalL = layers.BatchNormalization()( convL3 ) # Normalizes data -> Improves model stability? -> Will try training without to see how it changes
    poolL3 = layers.MaxPooling2D( padding = 'same' )( batchNormalL ) # Max Pooling Layer 2 -> Shrinks Dimensions of Data

    flatten = layers.Flatten()( poolL3 ) # Converts the data to 1 dimensional. Make training and computations faster apparently 

    # Create Output "Layer". It's really 4 branches that each have their own output layer, 4 for each character in the label.
    output = [] # Will contain 4 output channels, one for each character in the label
    for _ in range(4): 
        densL1 = layers.Dense(64, activation = 'relu')(flatten)
        drop = layers.Dropout(0.5)(densL1)
        res = layers.Dense(36, activation = "sigmoid")(drop)

        output.append(res)

    # Contruct and Compile model
    model = Model(inputL, output)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ["accuracy"])

    return model
