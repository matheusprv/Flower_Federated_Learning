from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.models import Model


def generate_model():
    x = input_layer = Input(shape=(28,28))
    x = Flatten()(x)
    x = Dense(256, activation = 'relu') (x)
    x = Dense(64, activation = 'relu') (x)
    output = Dense(10, activation = 'softmax')(x)

    model = Model(inputs=input_layer, outputs=output)
    return model

def get_server_address():
    address = "0.0.0.0"
    port = "8080"

    return address, port