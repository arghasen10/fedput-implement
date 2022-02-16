import collections

import nest_asyncio
from numpy import mean
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import fedput

tf.executing_eagerly()

nest_asyncio.apply()

np.random.seed(42)

federated_train_data = fedput.create_df_datasets()
print(federated_train_data[0].element_spec)


def model_fn():
    keras_model = fedput.create_model()
    return tff.learning.from_keras_model(
        keras_model,
        # input_spec=collections.OrderedDict(
        #     x=collections.OrderedDict(
        #         a=tf.TensorSpec(shape=[None, 1], dtype=tf.float64),
        #         b=tf.TensorSpec(shape=[None, 1], dtype=tf.float64)),
        #     y=tf.TensorSpec(shape=[None, 1], dtype=tf.float64)),
        input_spec=federated_train_data[0].element_spec,
        loss=tf.keras.losses.MeanSquaredLogarithmicError(),
        metrics=[tf.keras.metrics.Accuracy()]
    )


#
# @tff.tf_computation
# def server_init():
#     model = model_fn()
#     return model.trainable_variables
#
#
# @tff.federated_computation
# def initialize_fn():
#     return tff.federated_value(server_init(), tff.SERVER)
#
#
# def client_update(model, dataset, server_weights, client_optimizer):
#     """Performs training (using the server model weights) on the client's dataset."""
#     # Initialize the client model with the current server weights.
#     client_weights = model.trainable_variables
#     # Assign the server weights to the client model.
#     tf.nest.map_structure(lambda x, y: x.assign(y),
#                           client_weights, server_weights)
#
#     # Use the client_optimizer to update the local model.
#     for batch in dataset:
#         with tf.GradientTape() as tape:
#             # Compute a forward pass on the batch of data
#             outputs = model.forward_pass(batch)
#
#         # Compute the corresponding gradient
#         grads = tape.gradient(outputs.loss, client_weights)
#         grads_and_vars = zip(grads, client_weights)
#
#         # Apply the gradient using a client optimizer.
#         client_optimizer.apply_gradients(grads_and_vars)
#
#     return client_weights
#
#
# def server_update(model, mean_client_weights):
#     """Updates the server model weights as the average of the client model weights."""
#     model_weights = model.trainable_variables
#     # Assign the mean client weights to the server model.
#     tf.nest.map_structure(lambda x, y: x.assign(y),
#                           model_weights, mean_client_weights)
#     return model_weights
#
#
# tf_dataset_type = tff.SequenceType(model_fn().input_spec)
# model_weights_type = server_init.type_signature.result
#
#
# @tff.tf_computation(tf_dataset_type, model_weights_type)
# def client_update_fn(tf_dataset, server_weights):
#     model = model_fn()
#     client_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
#     return client_update(model, tf_dataset, server_weights, client_optimizer)
#
#
# @tff.tf_computation(model_weights_type)
# def server_update_fn(mean_client_weights):
#     model = model_fn()
#     return server_update(model, mean_client_weights)
#
#
# federated_server_type = tff.FederatedType(model_weights_type, tff.SERVER)
# federated_dataset_type = tff.FederatedType(tf_dataset_type, tff.CLIENTS)
#
#
# @tff.federated_computation(federated_server_type, federated_dataset_type)
# def next_fn(server_weights, federated_dataset):
#     # Broadcast the server weights to the clients.
#     server_weights_at_client = tff.federated_broadcast(server_weights)
#     print(server_weights_at_client)
#     # Each client computes their updated weights.
#     client_weights = tff.federated_map(
#         client_update_fn, (federated_dataset, server_weights_at_client))
#
#     # The server averages these updates.
#     mean_client_weights = mean(client_weights)
#
#     # The server updates its model.
#     server_weights = server_update_fn(mean_client_weights)
#
#     return server_weights
#
#
# federated_algorithm = tff.templates.IterativeProcess(
#     initialize_fn=initialize_fn,
#     next_fn=next_fn
# )
#
#
# def evaluate(server_state):
#     keras_model = fedput.create_model()
#     keras_model.compile(
#         loss=tf.keras.losses.MeanSquaredLogarithmicError(),
#         metrics=[tf.keras.metrics.Accuracy()]
#     )
#     keras_model.set_weights(server_state)
#     keras_model.evaluate(federated_train_data)
#
#
# server_state = federated_algorithm.initialize()
# evaluate(server_state)

# for _ in range(15):
#     server_state = federated_algorithm.next(server_state, federated_train_data)


fed_avg = tff.learning.build_federated_averaging_process(
    model_fn=model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(lr=0.3))

state = fed_avg.initialize()
state, metrics = fed_avg.next(state, federated_train_data)
train_metrics = metrics['train']
print('loss={l:.3f}, accuracy={a:.3f}'.format(
    l=train_metrics['loss'], a=train_metrics['accuracy']))

NUM_ROUNDS = 10


def keras_evaluate(state, round_num):
    # Take our global model weights and push them back into a Keras model to
    # use its standard `.evaluate()` method.
    keras_model = fedput.create_model()
    keras_model.compile(
        loss=tf.keras.losses.MeanSquaredLogarithmicError(),
        metrics=[tf.keras.metrics.Accuracy()])
    state.model.assign_weights_to(keras_model)
    loss, accuracy = keras_model.evaluate(federated_train_data, steps=2, verbose=0)
    print('\tEval: loss={l:.3f}, accuracy={a:.3f}'.format(l=loss, a=accuracy))


for round_num in range(NUM_ROUNDS):
    print('Round {r}'.format(r=round_num))
    keras_evaluate(state, round_num)
    state, metrics = fed_avg.next(state, federated_train_data)
    train_metrics = metrics['train']
    print('\tTrain: loss={l:.3f}, accuracy={a:.3f}'.format(
        l=train_metrics['loss'], a=train_metrics['accuracy']))

print('Final evaluation')
keras_evaluate(state, NUM_ROUNDS + 1)
