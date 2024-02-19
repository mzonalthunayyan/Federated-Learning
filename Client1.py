

# Define Flower client
class CifarClient(fl.client.NumPyClient):
    #get_parameters: send client parameters to the server 
    def get_parameters(self, config):
        return model.get_weights()
    
    #fit: receives parameters from the server 
    def fit(self, parameters, config): 
        model.set_weights(parameters)
        model.fit(train_data, train_labels, epochs=1)
        return model.get_weights(), len(train_data), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(test_data, test_labels)
        return loss, len(test_data), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=CifarClient())


