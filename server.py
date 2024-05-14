import flwr as fl
from tensorflow import keras

from shared_info import generate_model, get_server_address

###############################################################################
"""
    Definindo o modelo
"""

model = generate_model()
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])


###############################################################################
"""
    Função de validação do modelo
    Ela é chamada sempre que um round de treinamento é completado
"""

results_list = []

def server_validation(model):
    (_, _), (x, y) = keras.datasets.mnist.load_data()

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: dict):

        model.set_weights(parameters)  
        loss, accuracy = model.evaluate(x, y)

        print(f"Server round: {server_round} - Accuracy: {accuracy} - Loss: {loss}")

        results_list.append({
            "round": server_round,
            "loss": loss, 
            "accuracy": accuracy
        })

        return loss, {"accuracy": accuracy}

    return evaluate

###############################################################################
"""
    Definindo a strategy
"""

strategy = fl.server.strategy.FaultTolerantFedAvg(
    min_fit_clients       = 1,              # Número mínimo de clientes usados para treinar
    min_evaluate_clients  = 1,              # Número mínimo de clientes para validação do modelo
    min_available_clients = 1,              # Número mínimo de clientes no sistema
    evaluate_fn = server_validation(model), # Validação do modelo a cada round de treino

    # Tolerância a falhas
    min_completion_rate_fit      = 0.5,     # Mínimo de clientes a ter completado para que um round possa ser concluido
    min_completion_rate_evaluate = 0.5      # Porcentagem de clientes mínimos necessários para a validação
)

###############################################################################
"""
    Iniciando o servidor
"""

address, port = get_server_address()

fl.server.start_server(
    server_address = f"[::]:{port}",
    config         = fl.server.ServerConfig(num_rounds = 10),
    strategy       = strategy
)

###############################################################################
"""
    Exibindo informações do treinamento
"""
for result in results_list:
    print(result)