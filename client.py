import flwr as fl
from tensorflow import keras

LOCAL_EPOCHS = 10
BATCH_SIZE = 128

###############################################################################
"""
    Carregando o dataset
"""
import numpy as np

def load_dataset(client_id, total_clients):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Realizando total_clients splits no dataset e pegando a parte referente ao client_id
    x_train = np.split(x_train, total_clients)[client_id - 1]
    y_train = np.split(y_train, total_clients)[client_id - 1]
    x_test = np.split(x_test, total_clients)[client_id - 1]
    y_test = np.split(y_test, total_clients)[client_id - 1]

    return x_train, y_train, x_test, y_test

###############################################################################
"""
    Definindo o modelo
"""

from shared_info import generate_model, get_server_address

model = generate_model()
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

###############################################################################
"""
    Criando o cliente flower
"""
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        super().__init__()
        
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    
    # Lista de parametros do modelo para retornar ao servidor
    def get_parameters(self, config):
        return model.get_weights()

    #Treino local do modelo
    def fit(self, parameters, config):
        
        # Atualizando os parâmetros locais
        self.model.set_weights(parameters)

        validation_split = 0.15

        # Treinando o modelo
        history = self.model.fit(
            x = self.x_train,
            y = self.y_train,
            batch_size = BATCH_SIZE,
            epochs = LOCAL_EPOCHS,
            validation_split = validation_split
        )
       

        # Retornando para o servidor os dados modificados do modelo
        updated_model_parameters    = self.model.get_weights()
        num_examples_used_for_train = int(len(self.x_train) * (1 - validation_split) )
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }

        return updated_model_parameters, num_examples_used_for_train, results

    # Validação local do modelo
    def evaluate(self, parameters, config):

        # Atualizando os parâmetros locais
        self.model.set_weights(parameters)

        # Avaliando o modelo com os dados locais
        loss, accuracy = self.model.evaluate(
            x = self.x_test, 
            y = self.y_test, 
            batch_size = BATCH_SIZE,
        )
        num_examples_used_for_test = len(self.x_test)
        print(f"Accuracy: {accuracy}  -  Loss: {loss}")

        return loss, num_examples_used_for_test, {"accuracy": accuracy}



###############################################################################
"""
    Inicializando o cliente
"""

import sys

if __name__ == "__main__":
    assert (len(sys.argv) == 3)
    client_id = int(sys.argv[1])
    total_clients = int(sys.argv[2])


    x_train, y_train, x_test, y_test = load_dataset(client_id, total_clients)

    client = FlowerClient(model, x_train, y_train, x_test, y_test)
    fl.client.start_client(
            server_address = "localhost:8080", 
            client=client.to_client()
    )