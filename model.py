import numpy as np
from keras.models import load_model, Model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, LeakyReLU, add
from keras.optimizers import SGD
from keras import regularizers
import tensorflow as tf
import keras.backend as K
try:
    from .util import calculate_id
except (ModuleNotFoundError, ImportError):
    from util import calculate_id


class NeuralNetwork:
    # the reg_const is used for kernel regularizer, the learning rate influences how the optimizer function changes
    # the weights of the neural network. The input consists of 2 game field - one from each players perspective
    def __init__(self, reg_const=0.0001, learning_rate=0.1, input_dim=(2, 6, 7), output_dim=42):
        self.reg_const = reg_const
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim

        # the overall structure of the model starts with the input layer that feeds into multiple residual layers
        # (the exact number is arbitrary and can be changed) to analyse the game state.
        # Afterwards the model splits into policy and value head

        # creating the input of the model
        model_input = Input(shape=self.input_dim, name='main_input')

        # creating an convolutional layer with a number (can also be changed) of 4 by 4 filters
        # padding is used to preserve the dimensions after applying the filters, but no bias towards the layer
        # Here no activation function is used. The data format effects the ordering of dimension of the input
        # The kernel regularizer adds a penalty before over-fitting the weights
        input_layer = Conv2D(filters=64, kernel_size=(4, 4), data_format="channels_first", padding='same', use_bias=False,
                             activation='linear', kernel_regularizer=regularizers.l2(self.reg_const))(model_input)

        # BatchNormalization scales the data to fit into an activation function. The mean is transformed to be 0
        # and the standard deviation to be 1. This accurately represents how high a value truly is.
        input_layer = BatchNormalization(axis=1)(input_layer)

        # LeakyReLU is a modification of the ReLU activation function which specifies when a neuron fires.
        # ReLU cuts all negative values off while LeakyReLU applied a small factor to it - standard activation functions
        input_layer = LeakyReLU()(input_layer)

        # 8 Residual Layers are created. The number of layer and filter size influence the complexity of the model
        hidden_layer_1 = self.addLayer(input_layer, 64, (4, 4))
        hidden_layer_2 = self.addLayer(hidden_layer_1, 64, (4, 4))
        hidden_layer_3 = self.addLayer(hidden_layer_2, 64, (4, 4))
        hidden_layer_4 = self.addLayer(hidden_layer_3, 64, (4, 4))
        hidden_layer_5 = self.addLayer(hidden_layer_4, 64, (4, 4))
        hidden_layer_6 = self.addLayer(hidden_layer_5, 64, (4, 4))
        hidden_layer_7 = self.addLayer(hidden_layer_6, 64, (4, 4))
        hidden_layer_8 = self.addLayer(hidden_layer_7, 64, (4, 4))

        # The model is split into value and policy head to generate the wanted output

        # value head (value of the game state)
        vh = Conv2D(filters=1, kernel_size=(1, 1), data_format="channels_first", padding='same', use_bias=False,
                    activation='linear', kernel_regularizer=regularizers.l2(self.reg_const))(hidden_layer_8)
        # BatchNormalization and the Leaky ReLu activation functions are applied before flattening the 2D output into
        # an 1D array. This array is further transformed by two dense layers to only output one value
        vh = Flatten()(LeakyReLU()(BatchNormalization(axis=1)(vh)))
        # Dense is a simple interconnected layer and 20 is the unit size.
        # It is used to add complexity between the array and the final value
        vh = LeakyReLU()(Dense(20, use_bias=False, activation='linear',
                               kernel_regularizer=regularizers.l2(self.reg_const))(vh))
        # the activation function tanh scales the input to between -1 and 1 which represents the value of a game state
        # the 1 unit represents the output of the value head
        vh = Dense(1, use_bias=False, activation='tanh', kernel_regularizer=regularizers.l2(self.reg_const),
                   name='value_head')(vh)

        # policy head (probability distribution of possible moves)
        ph = Conv2D(filters=2, kernel_size=(1, 1), data_format="channels_first", padding='same', use_bias=False,
                    activation='linear', kernel_regularizer=regularizers.l2(self.reg_const))(hidden_layer_8)
        # same as above
        ph = Flatten()(LeakyReLU()(BatchNormalization(axis=1)(ph)))
        # There is no need for another layer because the output has the same dimensions as the input.
        # This final layer uses a softmax activation function to represent the move as probabilities
        ph = Dense(self.output_dim, use_bias=False, activation='linear',
                   kernel_regularizer=regularizers.l2(self.reg_const), name='policy_head')(ph)

        # the model is created with the specified input and outputs
        model = Model(inputs=[model_input], outputs=[vh, ph])
        # Lastly the model is compiles. A function is specified to calculate the loss (discrepancy to the actual value)
        # for both outputs. As an optimizer function Stochastic Gradient Descent is used in order to minimize the loss
        # when fitting the model. The loss_weight argument specifies that both losses make up 50% of the total loss
        model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': softmax_cross_entropy_with_logits},
                      optimizer=SGD(lr=self.learning_rate, momentum=0.9),
                      loss_weights={'value_head': 0.5, 'policy_head': 0.5})

        self.model = model

    # adds a residual layer to the network
    def addLayer(self, input_layer, filters, kernel_size):
        conv_1 = Conv2D(filters=filters, kernel_size=kernel_size, data_format="channels_first", padding='same',
                        use_bias=False, activation='linear', kernel_regularizer=regularizers.l2(self.reg_const))(input_layer)
        conv_1 = BatchNormalization(axis=1)(conv_1)
        conv_1 = LeakyReLU()(conv_1)

        conv_2 = Conv2D(filters=filters, kernel_size=kernel_size, data_format="channels_first", padding='same',
                        use_bias=False, activation='linear', kernel_regularizer=regularizers.l2(self.reg_const))(conv_1)
        conv_2 = BatchNormalization(axis=1)(conv_2)
        # a residual layer skips a layer and feeds the input to the next one
        conv_2 = add([input_layer, conv_2])
        conv_2 = LeakyReLU()(conv_2)
        return conv_2

    # the game state is converted before being used as input for the model
    def convert_to_model_input(self, game_state):
        input_to_model = calculate_id(game_state.array, array=True)
        input_to_model = np.reshape(input_to_model, self.input_dim)
        return input_to_model

    # input is fed into the model and the output (value and policy) are returned
    def predict(self, x):
        return self.model.predict(x)

    # the gathered data (game states, moves and results) is fed to the network to improve it
    def fit(self, states, targets, epochs):
        return self.model.fit(states, targets, epochs=epochs, verbose=1, validation_split=0.1)

    # saves a model version to memory
    def write(self, args, version):
        try:
            self.model.save("run"+str(args['run'])+"/models/version"+str(version)+".h5")
            print("model written to run"+str(args['run'])+"/models/version"+str(version)+".h5")
        except (FileNotFoundError, OSError, KeyError):
            self.model.save("run1/models/version"+str(version)+".h5")
            print("model written run1/models/version"+str(version)+".h5")

    # reads a model from memory and initializes it
    def read(self, run, version):
        try:
            return load_model('run' + str(run) + "/models/version" + str(version) + '.h5',
            custom_objects={'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits})
        except (FileNotFoundError, OSError):
            return NeuralNetwork().model


# logits are the result of the Dense Layer. Softmax provides probaplities and cross entropy calculates the differnce to the labels
def softmax_cross_entropy_with_logits(y_true, y_pred):
    p = y_pred
    pi = y_true

    zero = tf.zeros(shape=tf.shape(input=pi), dtype=tf.float32)
    where = tf.equal(pi, zero)

    negatives = tf.fill(tf.shape(input=pi), -100.0)
    p = tf.compat.v1.where(where, negatives, p)

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(pi), logits=p)

    return loss
