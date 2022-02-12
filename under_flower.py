import os
import sys
import flwr as fl
import pandas as pd
import tensorflow as tf
import keras
import fedput

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if __name__ == "__main__":
    root = './dataset/labelencoded4Gdatasets/'
    paths = os.listdir(root)
    users4G = fedput.collect_4G_data()
    users5G = fedput.collect_simulated_5G_data()
    usersLumos = fedput.collect_data_lumos()
    usersMN = fedput.collect_mn_wild()
    usersIrish = fedput.collect_data_irish()
    users = [*users4G, *users5G, *usersMN, *usersIrish, usersLumos]
    print('users length:  ', len(users))

    user = users[int(sys.argv[1])]
    print(user[1].shape)

    model = fedput.create_model()
    model.compile(optimizer='adam', loss='msle')
    print(model.summary())

    # Define Flower client
    class CifarClient(fl.client.NumPyClient):
        def get_parameters(self):  # type: ignore
            return model.get_weights()

        def fit(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            model.fit(user[1], user[2], epochs=25, batch_size=256, validation_data=(user[3], user[4]), shuffle=False)
            return model.get_weights(), len(user[1]), {}

        def evaluate(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            loss = model.evaluate(user[3], user[4])
            return loss, len(user[3]), {}

    # Start Flower client
    fl.client.start_numpy_client("localhost:8080", client=CifarClient())
