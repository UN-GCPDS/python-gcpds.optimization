from scipy.sparse import coo_matrix, eye, hstack
from tensorflow.keras.callbacks import Callback
from google.colab import drive
from tensorflow.python.ops.gen_math_ops import imag_eager_fallback
from tensorflow.python.ops.nn_ops import softmax
from keras import backend as k
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from IPython import display
from scipy.optimize import linprog
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import matplotlib. pyplot as plt
from keras.layers import Activation, Dense, Input, BatchNormalization, Dropout, Conv1D, Flatten, MaxPool1D, Dot, Reshape, Conv2D, Concatenate, ReLU, Lambda, MaxPooling2D, Normalization
import cvxpy as cp
import tensorflow as tf
from keras import Model
from IPython import display
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import warnings
import pandas as pd
from IPython import display
from keras.models import Sequential
import os
from pathlib import Path
import scipy.io as sio
import time
mse = mean_squared_error
mae = mean_absolute_error
mape = mean_absolute_percentage_error
warnings.filterwarnings("ignore")


def gen_w(Data_inf, Data_W, Data_T, Data_C, Data_Dc, Data_sto):
    """
    Generates a matrix combining information from wells, pipes, compressors, users, and storages.

    Parameters
    ----------
    Data_inf : List
        Information data.
    Data_W : pd.DataFrame
        Data related to wells.
    Data_T : pd.DataFrame
        Data related to pipes.
    Data_C : pd.DataFrame
        Data related to compressors.
    Data_Dc : pd.DataFrame
        Data related to users.
    Data_sto : pd.DataFrame
        Data related to storages.

    Returns
    -------
    np.ndarray
        A matrix combining all the information.

    Raises
    ------
    ValueError
        If any input data is not in the expected format or type.

    """
    if not all(isinstance(data, pd.DataFrame) for data in [Data_W, Data_T, Data_C, Data_Dc, Data_sto]):
        raise ValueError("All data inputs must be pandas DataFrames")

    num_nodes = len(Data_inf)
    num_wells = len(Data_W)
    wells_matrix = coo_matrix((np.ones(
        num_wells, ), (Data_W['node'] - 1, np.arange(num_wells))), shape=(num_nodes, num_wells))

    num_pipes = len(Data_T)
    pipe_data = np.concatenate(
        (-1.0 * np.ones(num_pipes), np.ones(num_pipes)))
    pipe_row = pd.concat((Data_T['fnode'] - 1, Data_T['tnode'] - 1))
    pipe_col = np.concatenate(2 * [np.arange(num_pipes)])

    pipes_matrix = coo_matrix(
        (pipe_data, (pipe_row, pipe_col)), shape=(num_nodes, num_pipes))

    num_compressors = len(Data_C)
    compressor_data = np.concatenate(
        (-1.0 * np.ones(num_compressors), np.ones(num_compressors)))
    compressor_row = pd.concat((Data_C['fnode'] - 1, Data_C['tnode'] - 1))
    compressor_col = np.concatenate(2 * [np.arange(num_compressors)])
    compressors_matrix = coo_matrix(
        (compressor_data, (compressor_row, compressor_col)), shape=(num_nodes, num_compressors))

    num_users = len(Data_Dc.T)
    users_matrix = hstack(num_users * [eye(num_nodes)])

    num_storages = len(Data_sto)
    storage_matrix = coo_matrix((np.ones(num_storages, ), (
        Data_sto['node'] - 1, np.arange(num_storages))), shape=(num_nodes, num_storages))
    storage_matrix = hstack([storage_matrix, -1.0 * storage_matrix])

    result_matrix = hstack((wells_matrix, pipes_matrix,
                           compressors_matrix, users_matrix, storage_matrix)).toarray()

    return result_matrix


class CustomSigmoidActivation():

    def __init__(self, lower_bound: float, upper_bound: float):
        """
        Initializes the CustomSigmoidActivation with specified bounds.

        Parameters
        ----------
        lower_bound : float
            The lower limit for the activation function.
        upper_bound : float
            The upper limit for the activation function.
        """
        self.lower_bound = tf.cast(lower_bound, tf.float32)
        self.upper_bound = tf.cast(upper_bound, tf.float32)

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """
        Apply the custom sigmoid activation function to the input.

        Parameters
        ----------
        x : tf.Tensor
            The input tensor to apply the activation function.

        Returns
        -------
        tf.Tensor
            The output tensor after applying the activation function.
        """
        x = tf.cast(x, tf.float32)
        x = k.sigmoid(x) * (self.upper_bound -
                            self.lower_bound) + self.lower_bound
        return x

    def get_config(self) -> dict:
        """
        Returns the configuration of the CustomSigmoidActivation function.

        Returns
                -------
        dict
            A dictionary containing the configuration: lower and upper bounds.
        """
        return {
            "lower_bound": self.lower_bound.numpy(),
            "upper_bound": self.upper_bound.numpy(),
        }


def split(feature_data, output_data):
    """
    Splits the feature and output data into training and test sets.

    Parameters
    ----------
    feature_data : array-like
        The input features data.
    output_data : array-like
        The output data corresponding to the input features.

    Returns
    -------
    tuple
        A tuple containing split data: (X_train, X_test, y_train, y_test)

    """

    X_train, X_test, y_train, y_test = train_test_split(
        feature_data, output_data, test_size=0.2)

    return X_train, X_test, y_train, y_test


def create_customized_dense_layer(input_tensor, layer_info, layer_name, activation_function, initializer):
    """
    Creates a custom dense layer using the Keras Dense function.

    Parameters
    ----------
    input_tensor : Tensor
        The input tensor for the dense layer.
    layer_info : list
        A list containing specific information for constructing the dense layer.
    layer_name : str
        The name of the dense layer.
    activation_function : callable
        The activation function to be applied to the output of the dense layer.
    initializer : Initializer
        The initializer for the kernel weights of the dense layer.

    Returns
    -------
    Tensor
        The output tensor from the dense layer.
    """
    num_neurons = len(layer_info)
    dense_layer = Dense(num_neurons, name=layer_name, kernel_initializer=initializer,
                        activation=activation_function)(input_tensor)

    return dense_layer


def create_custom_pres_layer(input_tensor, lower_bound, upper_bound, layer_name, initializer):
    """
    Creates a custom dense layer with a CustomSigmoidActivation activation function.

    Parameters
    ----------
    input_tensor : Tensor
        The input tensor for the dense layer.
    lower_bound : float
        The lower bound for the CustomSigmoidActivation function.
    upper_bound : float
        The upper bound for the CustomSigmoidActivation function.
    layer_name : str
        The name of the dense layer.
    initializer : Initializer
        The initializer for the kernel weights of the dense layer.

    Returns
    -------
    Tensor
        The output tensor from the dense layer.
    """
    num_neurons = len(lower_bound)
    custom_activation = CustomSigmoidActivation(lower_bound, upper_bound)
    dense_layer = Dense(num_neurons, name=layer_name, kernel_initializer=initializer,
                        activation=custom_activation)(input_tensor)

    return dense_layer


def extract_flows_from_model(model, test_data):
    """
    Extracts the output of a specific layer ('Flujos') from a given Keras model.

    Parameters
    ----------
    model : keras.Model
        The Keras model from which to extract the layer output.
    test_data : array-like
        The input data to pass through the model.

    Returns
    -------
    numpy.ndarray
        The output of the 'Flujos' layer in the model.
    """
    intermediate_model = tf.keras.Model(
        inputs=model.inputs, outputs=model.get_layer('Flujos').output)
    layer_output = intermediate_model.predict(test_data, verbose=False)
    return layer_output


class LearningRateReducer(Callback):
    """
    A custom callback to reduce the learning rate based on epoch thresholds.

    Attributes
    ----------
    epoch_threshold : list of int
        Epoch numbers at which the learning rate should be reduced.

    Methods
    -------
    on_epoch_begin(epoch, logs=None):
        Reduces the learning rate if the current epoch is in the epoch_threshold.
    """

    def __init__(self, epoch_threshold):
        """
        Initializes the LRReducer callback with specified epoch thresholds.

        Parameters
        ----------
        epoch_threshold : list of int
            Epoch numbers at which the learning rate should be reduced.
        """
        super(LearningRateReducer, self).__init__()
        self.epoch_threshold = epoch_threshold

    def on_epoch_begin(self, epoch, logs=None):
        """
        Called at the beginning of an epoch during training.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        logs : dict, optional
            Currently no logs are used in this method.
        """
        if epoch in self.epoch_threshold:
            lr = float(self.model.optimizer.lr)
            new_lr = lr * 0.1
            self.model.optimizer.lr.assign(new_lr)


class WellDense(tf.keras.layers.Layer):
    """
    A custom Keras layer that applies a threshold to the sum of inputs.

    Attributes
    ----------
    well : tf.Variable
        The threshold value for the sum of inputs.

    Methods
    -------
    call(inputs):
        Computes the sum of inputs and applies the threshold.
    get_config():
        Returns the layer's configuration.
    """

    def __init__(self, well=80.0, **kwargs):
        """
        Initializes the WellDense layer with a specified threshold.

        Parameters
        ----------
        well : float, optional
            The threshold value for the sum of inputs.
        """
        super().__init__(**kwargs)
        self.well = tf.Variable(
            initial_value=well, trainable=False, dtype=tf.float32)

    def call(self, inputs):
        """
        Computes the sum of inputs and applies the threshold.

        Parameters
        ----------
        inputs : Tensor
            The input tensor to the layer.

        Returns
        -------
        Tensor
            The output tensor after applying the threshold.
        """
        x = inputs
        w = tf.reduce_sum(x, axis=1, keepdims=True)
        w = tf.where(w >= self.well, self.well, w)
        return w

    def get_config(self):
        """
        Returns the configuration of the WellDense layer.

        Returns
        -------
        dict
            A dictionary containing the configuration: well.
        """
        return {
            "well": self.well.numpy(),
        }


class RestrictedDense(tf.keras.layers.Layer):
    """
    A custom dense layer with constraints on its output.

    Attributes
    ----------
    initializer : Initializer
        The initializer for the kernel weights of the layer.
    well : tf.Variable
        A threshold value used in the layer's computations.
    units : int
        The number of units in the dense layer.
    dense : tf.keras.layers.Dense
        The underlying dense layer.

    Methods
    -------
    call(inputs):
        Computes the layer's output with the given inputs.
    get_config():
        Returns the layer's configuration.
    """

    def __init__(self, initializer, well, units, **kwargs):
        """
        Initializes the RestrictedDense layer.

        Parameters
        ----------
        initializer : Initializer
            The initializer for the kernel weights of the layer.
        well : float
            A threshold value used in the layer's computations.
        units : int
            The number of units in the dense layer.
        """
        super().__init__(**kwargs)
        self.initializer = initializer
        self.units = units
        self.well = tf.Variable(
            initial_value=well, trainable=False, dtype=tf.float32)
        self.dense = tf.keras.layers.Dense(
            self.units, activation='elu', kernel_initializer=self.initializer)

    def call(self, inputs):
        """
        Computes the output of the layer with given inputs.

        Parameters
        ----------
        inputs : tuple of Tensors
            Inputs to the layer: (x, In, fw).

        Returns
        -------
        Tensor
            The output tensor of the layer.
        """
        x, In, fw = inputs
        x = self.dense(x)

        x = tf.where(x < 0.0, 0.0, x)
        x = tf.where(x < In, x, In)
        mask = tf.where(fw >= self.well, 1.0, 0.0)
        x = x * mask
        return x

    def get_config(self):
        """
        Returns the configuration of the RestrictedDense layer.

        Returns
        -------
        dict
            A dictionary containing the configuration.
        """
        return {
            "initializer": self.initializer,
            "well": self.well.numpy(),
            "units": self.units,
        }


def mish(x):
    """
    Mish Activation Function.

    Applies the mish function element-wise to the input Tensor.

    Parameters
    ----------
    x : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The transformed tensor after applying the mish activation function.
    """
    return x * tf.math.tanh(tf.math.softplus(x))


def swish(x):
    """
    Swish Activation Function.

    Applies the swish function element-wise to the input Tensor.

    Parameters
    ----------
    x : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The transformed tensor after applying the swish activation function.
    """
    return x * tf.sigmoid(x)


class CustomLossFuntion(tf.keras.losses.Loss):
    """
    A custom loss function that computes a loss based on specific criteria
    and includes a penalty term for certain conditions.

    Attributes
    ----------
    Kt : float
        A coefficient used in the loss computation.
    i : int
        Index used in gathering elements from predictions.
    j : int
        Another index used in gathering elements from predictions.
    Bc : float
        A boundary condition used in the penalty term.
    ic : int
        Index for penalty computation.
    jc : int
        Another index for penalty computation.
    find : int, optional
        A fixed index to slice the predictions for certain operations.
    """

    def __init__(self, Kt, i, j, Bc, ic, jc, find=6):
        super().__init__()
        self.Kt = Kt
        self.i = i
        self.j = j
        self.ic = ic
        self.jc = jc
        self.find = find
        self.lower_bound = tf.constant(1.0, dtype=tf.float32)
        self.upper_bound = tf.constant(Bc, dtype=tf.float32)

    def call(self, y_true, y_pred):
        """
        Computes the custom loss between true labels and predicted labels.

        Parameters
        ----------
        y_true : Tensor
            True labels.
        y_pred : Tensor
            Predicted labels.

        Returns
        -------
        Tensor
            Computed loss value.
        """
        f = tf.cast(y_pred[:, :self.find], tf.float32)
        P = tf.cast(y_pred[:, self.find:], tf.float32)
        pi = tf.gather(P, self.i, axis=1)
        pj = tf.gather(P, self.j, axis=1)
        loss = tf.keras.losses.huber(f, tf.sign(
            pi**2 - pj**2) * tf.sqrt(self.Kt * tf.abs(pi**2 - pj**2)))

        P_ic = tf.gather(P, self.ic, axis=1)
        P_jc = tf.gather(P, self.jc, axis=1)
        Penalty = tf.reduce_mean(
            tf.where(P_jc >= P_ic, 0.0, tf.square(P_jc - P_ic)))

        return loss + Penalty

    def get_config(self):
        """
        Returns the configuration of the custom loss function.

        Returns
        -------
        dict
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            'Kt': self.Kt,
            'i': self.i,
            'j': self.j,
            'ic': self.ic,
            'jc': self.jc,
            'lower_bound': self.lower_bound.numpy(),
            'upper_bound': self.upper_bound.numpy(),
            'find': self.find
        })
        return config


def flow_model(path, fd, seeds=1, s=1):
    """
    Builds and compiles a TensorFlow Keras model based on data from an Excel file and additional parameters.

    Parameters:
    - path (str): Path to the Excel file containing the model's data.
    - fd (array-like): Additional flow data for the model.
    - seeds (int): Seed for random number generation for reproducibility.
    - s (int): Mode switch for the model configuration.

    Returns:
    - tensorflow.keras.Model: The compiled Keras model.
    """
    seed = seeds
    np.random.seed(seed)
    tf.random.set_seed(seed)
    Data_inf = pd.read_excel(path, sheet_name='node.info')
    Data_D = pd.read_excel(path, sheet_name='node.dem')
    Data_Dc = pd.read_excel(path, sheet_name='node.demcost')
    Data_W = pd.read_excel(path, sheet_name='well')
    Data_T = pd.read_excel(path, sheet_name='pipe')
    Data_C = pd.read_excel(path, sheet_name='comp')
    Data_sto = pd.read_excel(path, sheet_name='sto')
    w = gen_w(Data_inf, Data_W, Data_T, Data_C, Data_Dc, Data_sto)
    N = w.shape[0]

    initializer = tf.keras.initializers.GlorotNormal(seed=seed)
    In = Input(shape=(N,))
    N1 = 8
    act = mish
    x = In
    for units in [N1 * 4, N1 * 8, N1 * 16, N1 * 32]:
        x = Dense(units, activation=act, kernel_initializer=initializer)(x)
        x = Normalization()(x)

    F = []
    if s == 0:
        Li = np.concatenate((Data_T['Fg_min'].values, [0] * len(Data_C), Data_D['Res'].values * 0.0, Data_D['Ind'].values * 0.0, Data_D['Com'].values * 0.0,
                            Data_D['NGV'].values * 0.0, Data_D['Ref'].values,
                            Data_D['Pet'].values * 0.0, [0] * len(Data_sto.values) * 2))
        Ls = np.concatenate((Data_T['Fg_max'].values, Data_C['fmaxc'].values, fd, Data_D['Ind'].values * 0.0, Data_D['Com'].values * 0.0, Data_D['NGV'].values * 0.0, Data_D['Ref'].values * 0.0, Data_D['Pet'].values * 0.0,
                            (Data_sto['V0'].values - Data_sto['V_max'].values) * 0.0, (Data_sto['Vmax'].values - Data_sto['V0'].values) * 0.0))
        F = []
        F0 = WellDense(well=Data_W['Imax'].values[0], name='F_0')(In)
        F1 = create_customized_dense_layer(
            x, Li, 'F_1', CustomSigmoidActivation(Li, Ls), initializer)
        F = Concatenate(name='F')([F0, F1])  # ,F2,F3])

    elif s == 1:
        fdin = len(Data_W['Imin'].values) + \
            len([0] * len(Data_T)) + len([0] * len(Data_C))
        fdfn = len([0] * len(Data_sto.values) * 2) + \
            len(Data_inf['Pmin'].values)
        F0 = WellDense(well=Data_W['Imax'].values[0], name='F_0')(In)
        Li = Data_T['Fg_min'].values
        Ls = Data_T['Fg_max'].values
        F1 = create_customized_dense_layer(
            x, Li, 'F_1', CustomSigmoidActivation(Li, Ls), initializer)
        Li = [0] * len(Data_C)
        Ls = Data_C['fmaxc'].values
        F2 = create_customized_dense_layer(
            x, Li, 'F_2', CustomSigmoidActivation(Li, Ls), initializer)
        F3 = RestrictedDense(
            initializer=initializer, well=Data_W['Imax'].values[0], name='F_3', units=len(Data_inf))([x, In, F0])
        Li = np.concatenate((Data_D['Ind'].values * 0.0, Data_D['Com'].values * 0.0, Data_D['NGV'].values * 0.0, Data_D['Ref'].values * 0.0,
                            Data_D['Pet'].values * 0.0, [0] * len(Data_sto.values) * 2))
        Ls = np.concatenate((Data_D['Ind'].values * 0.0, Data_D['Com'].values * 0.0, Data_D['NGV'].values * 0.0, Data_D['Ref'].values * 0.0, Data_D['Pet'].values * 0.0,
                            Data_sto['V0'].values - Data_sto['V_max'].values, Data_sto['Vmax'].values - Data_sto['V0'].values))
        F4 = create_customized_dense_layer(
            x, Li, 'F_4', CustomSigmoidActivation(Li, Ls), initializer)
        F = Concatenate(name='F')([F0, F1, F2, F3, F4])

    Out1 = tf.matmul(F, tf.constant(w.T, dtype=tf.float32))

    Kt = tf.constant(Data_T['Kij'].values, dtype=tf.float32)
    i = Data_T['fnode'].values - 1
    j = Data_T['tnode'].values - 1
    ftin = len(Data_W)
    ftfn = len(Data_T)
    f = F[:, ftin:ftfn + ftin]
    Li = Data_inf['Pmin'].values
    Ls = Data_inf['Pmax'].values

    x1 = Dense(N1 * 4, act, kernel_initializer=initializer)(f)
    x1 = Normalization()(x1)

    x1 = Dense(N1 * 8, act, kernel_initializer=initializer)(x1)
    x1 = Normalization()(x1)

    x1 = Dense(N1 * 16, act, kernel_initializer=initializer)(x1)
    x1 = Normalization()(x1)

    x1 = Dense(N1 * 32, act, kernel_initializer=initializer)(x1)
    x1 = Normalization()(x1)

    ib = Data_C['fnode'].values - 1
    jb = Data_C['tnode'].values - 1
    P = create_custom_pres_layer(x1, Li, Ls, 'P', initializer)

    Out2 = Concatenate()([f, P])
    Out3 = P

    Bc = Data_C['ratio'].values

    model = Model(In, [Out1, Out2])
    model.compile(loss=['huber', CustomLossFuntion(Kt, i, j, Bc, ib, jb, ftfn)],
                  optimizer=tf.keras.optimizers.Adamax(1e-2), loss_weights=[0.7, 0.3])
    return model


def plots(Balance, Wey, re, Costos, s):
    """
    Creates a set of boxplot visualizations for various data frames.

    Parameters:
    - Balance (DataFrame): DataFrame containing balance data.
    - Wey (DataFrame): DataFrame containing Weymouth data.
    - re (DataFrame): DataFrame containing ratio data.
    - Costos (DataFrame): DataFrame containing cost data.
    - s (int): Mode switch for the plot titles.

    This function generates a 2x2 grid of boxplots, each representing data from
    one of the provided DataFrames. The boxplots for 'Balance' and 'Costos' are
    set to a logarithmic scale. The title of the plots changes based on the value of 's'.
    """
    # Create a 2x2 subplot grid
    fig, axs = plt.subplots(2, 2, figsize=(29, 20))
    fontsize = 20
    rotation_angle = 45

    # Configure the Balance boxplot
    Balance.boxplot(ax=axs[0, 0], boxprops=dict(color='blue'), medianprops=dict(
        color='orangered'), whiskerprops=dict(color='blue'))
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_ylabel('MAPE', fontsize=fontsize)
    axs[0, 0].set_xlabel('Network dimension', fontsize=fontsize)
    axs[0, 0].tick_params(axis='x', labelsize=fontsize,
                          rotation=rotation_angle)
    axs[0, 0].tick_params(axis='y', labelsize=fontsize)

    # Configure the Wey boxplot
    Wey.boxplot(ax=axs[0, 1], boxprops=dict(color='blue'), medianprops=dict(
        color='orangered'), whiskerprops=dict(color='blue'))
    axs[0, 1].set_yscale('log')
    axs[0, 1].set_ylabel('MAPE', fontsize=fontsize)
    axs[0, 1].set_xlabel('Network dimension', fontsize=fontsize)
    axs[0, 1].tick_params(axis='x', labelsize=fontsize,
                          rotation=rotation_angle)
    axs[0, 1].tick_params(axis='y', labelsize=fontsize)

    # Configure the re boxplot
    re.boxplot(ax=axs[1, 0], boxprops=dict(color='blue'), medianprops=dict(
        color='orangered'), whiskerprops=dict(color='blue'))
    axs[1, 0].set_ylabel('MAPE', fontsize=fontsize)
    axs[1, 0].set_xlabel('Network dimension', fontsize=fontsize)
    axs[1, 0].tick_params(axis='x', labelsize=fontsize,
                          rotation=rotation_angle)
    axs[1, 0].tick_params(axis='y', labelsize=fontsize)

    # Configure the Costos boxplot
    Costos.boxplot(ax=axs[1, 1], boxprops=dict(color='blue'), medianprops=dict(
        color='orangered'), whiskerprops=dict(color='blue'))
    axs[1, 1].set_yscale('symlog')
    axs[1, 1].set_ylabel('Cost difference', fontsize=fontsize)
    axs[1, 1].set_xlabel('Network dimension', fontsize=fontsize)
    axs[1, 1].tick_params(axis='x', labelsize=fontsize,
                          rotation=rotation_angle)
    axs[1, 1].tick_params(axis='y', labelsize=fontsize)

    # Adjust layout and set the main title based on 's'
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if s == 0:
        fig.suptitle('Without Dynamic Function', fontsize=24, y=1.05)
    elif s == 1:
        fig.suptitle('With Dynamic Function', fontsize=24, y=1.05)

    # Show the plot
    plt.show()


def load_data(path, key):
    """
    Loads data from a MATLAB file specified by 'path' and extracts data based on 'key'.

    Parameters:
    - path (str): The path to the MATLAB file.
    - key (str): The key corresponding to the data within the MATLAB file.

    Returns:
    - list: A list containing the extracted data.

    The function iterates through eight sets of data within the MATLAB file, extracting
    each set and appending it to a list. The extracted data is expected to be in a
    specific format within the MATLAB file.
    """
    aux = []
    cont = 0
    for i in range(9):
        aux.append([])
        X_test = sio.loadmat(path)[key][i]
        for j in X_test:
            aux[cont].append(j[0])
        cont = cont + 1
    return aux


def evaluate_balance(FT, W, X):
    """
    Evaluates the balance of predictions using Mean Absolute Percentage Error (MAPE).

    Parameters:
    - FT (list): A list containing two sets of forecasted data.
    - W (array-like): Weight matrix used in the calculation.
    - X (array-like): Actual data for comparison.

    Returns:
    - list: A list containing two lists of MAPE values for each forecast set in FT.

    This function computes the MAPE between the actual data (X) and each set of
    forecasted data in FT, using the provided weight matrix W.
    """
    aux1 = [mape(X[i], FT[0][i] @ W.T) for i in range(len(FT[0]))]
    aux2 = [mape(X[i], FT[1][i] @ W.T) for i in range(len(FT[1]))]

    return [aux1, aux2]


def evaluate_weymouth(X, Y):
    """
    Evaluates the Weymouth equation performance by calculating the MAPE (Mean Absolute Percentage Error)
    between actual values (X) and predicted values (Y).

    Parameters:
    - X (list of lists): Actual values for comparison.
    - Y (list of lists): Predicted values to be compared against X.

    Returns:
    - list of lists: Calculated MAPE values for each set of comparisons.

    The function calculates the MAPE for two sets of data (indicated by indices 0 and 1) in X and Y.
    """
    # Using list comprehensions for more concise code
    aux1 = [mape(X[0][i], Y[0][i]) for i in range(len(X[0]))]
    aux2 = [mape(X[1][i], Y[1][i]) for i in range(len(X[1]))]

    return [aux1, aux2]


def evaluate_cost(X, Y):
    """
    Calculates the difference between predicted costs and actual costs.

    Parameters:
    - X (list): A list of actual cost values.
    - Y (list): A list of predicted cost values.

    Returns:
    - list: A list of differences between predicted and actual costs.

    This function iterates over each element in X and Y, calculates the
    difference (Y[i] - X[i]) for each corresponding element, and returns
    these differences in a list.
    """
    # Using list comprehension for concise calculation
    return [Y[i] - X[i] for i in range(len(X))]


class TimeHistoryTest(Callback):
    """
    A custom callback to track the time taken for predictions during model inference.

    Attributes:
    - prediction_times (list): A list to store the time taken for each prediction batch.

    Methods:
    - on_predict_begin: Called at the beginning of prediction.
    - on_predict_batch_begin: Called at the beginning of a batch during prediction.
    - on_predict_batch_end: Called at the end of a batch during prediction.
    """

    def on_predict_begin(self, logs=None):
        """
        Initialize the prediction_times list at the beginning of prediction.
        """
        self.prediction_times = []

    def on_predict_batch_begin(self, batch, logs=None):
        """
        Record the start time of a batch prediction.

        Parameters:
        - batch: Index of the batch.
        - logs: Additional information about the batch.
        """
        self.batch_time_start = time.time()

    def on_predict_batch_end(self, batch, logs=None):
        """
        Calculate and record the time taken for a batch prediction.

        Parameters:
        - batch: Index of the batch.
        - logs: Additional information about the batch.
        """
        batch_time_end = time.time() - self.batch_time_start
        self.prediction_times.append(batch_time_end)


def extract_data(times, b):
    """
    Extracts a specific subset of data from a multi-dimensional array.

    Parameters:
    - times (multi-dimensional array): The source array from which data is extracted.
    - b (int): The specific index in the first dimension to extract data from.

    Returns:
    - list: A list containing the extracted data.

    This function iterates over 320 elements in the specified sub-array of 'times',
    extracting a particular value from each element.
    """
    # Using a list comprehension for concise extraction
    return [times[b, :][i][0][0] for i in range(320)]


def plot_time(b='fit', s1='/content/drive/Shareddrives/red_gas_col/', s2='/content/drive/Shareddrives/red_gas_col/Prueba/Data/', s=1):
    """
    Generates and displays boxplot visualizations for time measurements of different network models.

    Parameters:
    - b (str): Indicates the type of operation to perform, 'fit' for fitting or 'predict' for prediction.
    - s1 (str): Base path for the file location.
    - s2 (str): Secondary path for data and model files.
    - s (int): Mode indicator for selecting model paths.

    The function first determines the file paths for the Excel and model files based on the mode 's'.
    It then loads the test data and times data. If 'b' is 'predict', it iterates over models, performs
    predictions, measures the time taken for predictions, and stores these times in a DataFrame.
    This DataFrame is then used to create boxplot visualizations. If 'b' is not 'predict', it loads
    time measurements from a CSV file and creates boxplots for these measurements.

    The boxplots provide a comparison of time taken by different network models for predictions or fitting.
    """

    s1_path = Path(s1)
    s2_path = Path(s2)

    filepath = ['ng_case8.xlsx', 'ng_case9.xlsx', 'ng_case10.xlsx', 'ng_case11.xlsx',
                'ng_case12.xlsx', 'ng_case13.xlsx', 'ng_case14.xlsx', 'ng_case15.xlsx', 'ng_caseCol_18.xlsx']
    if s == 0:
        modelpath = ['model2_8.h5', 'model2_9.h5', 'model2_10.h5', 'model2_11.h5',
                     'model2_12.h5', 'model2_13.h5', 'model2_14.h5', 'model2_15.h5', 'model2_Col.h5']
    elif s == 1:
        modelpath = ['model8.h5', 'model9.h5', 'model10.h5', 'model11.h5',
                     'model12.h5', 'model13.h5', 'model14.h5', 'model15.h5', 'modelCol.h5']

    X_test = load_data(s2_path / 'inputs_P.mat', 'inputs')
    times = np.array(sio.loadmat(s2_path / 'times_P.mat')['times'])

    fontsize = 20
    rotation_angle = 45
    if b == 'predict':
        Times_p = pd.DataFrame()
        for im in range(9):
            path = s1_path / filepath[im]
            fd = np.array(X_test[im]).max(axis=0)
            model2 = flow_model(str(path), fd, seeds=1, s=s)
            model2.load_weights(s2_path / modelpath[im])

            time_callback = TimeHistoryTest()
            predictions = model2.predict(np.array(X_test[im]), batch_size=1, callbacks=[
                                         time_callback], verbose=False)
            if im == 8:
                Times_p['NN(Col_18)'] = time_callback.prediction_times
                Times_p['S(Col_18)'] = extract_data(times, im)
            else:
                Times_p['NN(' + str(im + 8)
                        + ')'] = time_callback.prediction_times
                Times_p['S(' + str(im + 8) + ')'] = extract_data(times, im)
        plt.figure(figsize=(20, 20))

        Times_p.boxplot(boxprops=dict(color='blue'), medianprops=dict(
            color='orangered'), whiskerprops=dict(color='blue'))
        plt.yscale('log')
        # Ajusta aquí el tamaño para la etiqueta del eje X
        plt.xlabel('Network dimension', fontsize=20)
        # Ajusta aquí el tamaño para la etiqueta del eje Y
        plt.ylabel('Time[s]', fontsize=20)
        # Configurar las etiquetas de los ejes y su tamaño de fuente
        # Ajusta aquí el tamaño para los tick labels del eje X
        plt.xticks(fontsize=20, rotation=45)
        # Ajusta aquí el tamaño para los tick labels del eje Y
        plt.yticks(fontsize=20)
        plt.show()
    else:
        N = 15920
        timesr = pd.read_csv(s2_path / 'Time.csv').values
        im = pd.DataFrame()
        for i in range(9):
            if i == 8:
                im['NN(Col_18)'] = timesr[:, i]
                im['S(Col_18)'] = extract_data(times, i)
            else:
                im['NN(' + str(i + 8) + ')'] = timesr[:, i]
                im['S(' + str(i + 8) + ')'] = extract_data(times, i)
        plt.figure(figsize=(20, 20))

        im.boxplot(boxprops=dict(color='blue'), medianprops=dict(
            color='orangered'), whiskerprops=dict(color='blue'))
        plt.yscale('log')
        # Ajusta aquí el tamaño para la etiqueta del eje X
        plt.xlabel('Network dimension', fontsize=20)
        # Ajusta aquí el tamaño para la etiqueta del eje Y
        plt.ylabel('Time[s]', fontsize=20)
        # Configurar las etiquetas de los ejes y su tamaño de fuente
        # Ajusta aquí el tamaño para los tick labels del eje X
        plt.xticks(fontsize=20, rotation=45)
        # Ajusta aquí el tamaño para los tick labels del eje Y
        plt.yticks(fontsize=20)
        plt.show()


def ng_case_evaluate(s1='/content/drive/Shareddrives/red_gas_col/', s2='/content/drive/Shareddrives/red_gas_col/Prueba/Data/', s=1):
    """
    Evaluates multiple gas network cases using pre-trained models and generates evaluation metrics.

    Parameters:
    - s1 (str): Base directory path where the network case files are stored.
    - s2 (str): Directory path where input and output data files, and model files are stored.
    - s (int): Mode indicator (0 or 1) to choose between two sets of model paths.

    This function performs the following steps:
    - Loads test input and output data.
    - Based on the mode 's', selects a set of model paths.
    - Iterates over multiple network cases, loading the necessary data for each case.
    - For each case, it loads a pre-trained model and performs predictions.
    - Computes evaluation metrics such as Weymouth, Balance, Cost, and Pressure Ratios (Pj/Pi).
    - Aggregates the results into DataFrames for each metric.
    - Finally, calls the `plots` function to visualize the results.

    Returns a tuple of DataFrames: (Balance, Weymouth, Pj/Pi, Cost)
    """
    s1_path = Path(s1)
    s2_path = Path(s2)

    X_test = load_data(s2_path / 'inputs_P.mat', 'inputs')
    y_test = load_data(s2_path / 'outputs_P.mat', 'outputs')

    if s == 0:
        modelpath = ['model2_8.h5', 'model2_9.h5', 'model2_10.h5', 'model2_11.h5',
                     'model2_12.h5', 'model2_13.h5', 'model2_14.h5', 'model2_15.h5', 'model2_Col.h5']
    elif s == 1:
        modelpath = ['model8.h5', 'model9.h5', 'model10.h5', 'model11.h5',
                     'model12.h5', 'model13.h5', 'model14.h5', 'model15.h5', 'modelCol.h5']
    filepath = ['ng_case8.xlsx', 'ng_case9.xlsx', 'ng_case10.xlsx', 'ng_case11.xlsx',
                'ng_case12.xlsx', 'ng_case13.xlsx', 'ng_case14.xlsx', 'ng_case15.xlsx', 'ng_caseCol_18.xlsx']
    Weymouth = pd.DataFrame()
    Balance = pd.DataFrame()
    Costos = pd.DataFrame()
    PjPi = pd.DataFrame()

    for im in range(9):
        path = s1 + filepath[im]
        Data_inf = pd.read_excel(path, sheet_name='node.info')
        Data_D = pd.read_excel(path, sheet_name='node.dem')
        Data_Dc = pd.read_excel(path, sheet_name='node.demcost')
        Data_W = pd.read_excel(path, sheet_name='well')
        Data_T = pd.read_excel(path, sheet_name='pipe')
        Data_C = pd.read_excel(path, sheet_name='comp')
        Data_sto = pd.read_excel(path, sheet_name='sto')
        Cost = np.concatenate((Data_W['Cg'].values, Data_T['C_O'].values, Data_C['costc'], Data_Dc['al_Res'].values,
                               Data_Dc['al_Ind'].values, Data_Dc['al_Com'].values, Data_Dc[
                                   'al_NGV'].values, Data_Dc['al_Ref'].values, Data_Dc['al_Pet'].values,
                               Data_sto['C_S+'].values - Data_sto['C_V'].values, -1 * (Data_sto['C_S-'] - Data_sto['C_V']).values, Data_sto['C_V'].values)).reshape(-1, 1)

        w = gen_w(Data_inf, Data_W, Data_T, Data_C, Data_Dc, Data_sto)
        y = np.array(y_test[im])
        fd = np.array(X_test[im]).max(axis=0)

        model2 = flow_model(str(path), fd, seeds=1, s=s)
        model2.load_weights(s2_path / modelpath[im])

        model_A = tf.keras.Model(
            inputs=model2.inputs, outputs=model2.get_layer('F').output)
        model_B = tf.keras.Model(
            inputs=model2.inputs, outputs=model2.get_layer('P').output)
        Fe = model_A.predict(np.array(X_test[im]), verbose=False)
        Pe = model_B.predict(np.array(X_test[im]), verbose=False)
        i = Data_T['fnode'].values.astype(int) - 1
        j = Data_T['tnode'].values.astype(int) - 1
        CK = Data_T['Kij'].values.reshape(-1,)
        nod = len(Data_inf)
        F = y[:, 1:-(nod)]
        F = np.concatenate((F, np.zeros((F.shape[0], 2))), axis=1)
        P = y[:, -(nod):]
        Fte = Fe[:, 1:1 + len(Data_T)]
        Ft = F[:, 1:1 + len(Data_T)]
        Pie = Pe[:, i]
        Pje = Pe[:, j]
        Pte = np.sign(Pie**2 - Pje**2) * \
            np.sqrt(CK * np.abs(Pie**2 - Pje**2))

        Pi = P[:, i]
        Pj = P[:, j]
        Pt = np.sign(Pi**2 - Pj**2) * np.sqrt(CK * np.abs(Pi**2 - Pj**2))
        balance = evaluate_balance([Fe, F], w, np.array(X_test[im]))
        wey = evaluate_weymouth([Fte, Ft], [Pte, Pt])
        costos = evaluate_cost(Fe @ Cost[:-1], F @ Cost[:-1])
        ic = Data_C['fnode'].values - 1
        jc = Data_C['tnode'].values - 1
        Bc = Data_C['fnode'].values
        are = np.round(Pe[:, jc] / Pe[:, ic], 4)
        ind1 = np.unique(np.concatenate((np.where(~(are[:, 0] <= Bc[0]) | ~(are[:, 0] >= 1.0))[
            0], np.where(~(are[:, 1] <= Bc[1]) | ~(are[:, 1] >= 1.0))[0])))
        re1 = np.zeros(y.shape[0])

        if len(ind1) != 0:
            re1[ind1] = 100
        are = P[:, jc] / P[:, ic]
        ind1 = np.unique(np.concatenate((np.where(~(are[:, 0] <= Bc[0]) | ~(are[:, 0] >= 1.0))[
                         # .shape[0]/are.shape[0]*100
                         0], np.where(~(are[:, 1] <= Bc[1]) | ~(are[:, 1] >= 1.0))[0])))
        re2 = np.zeros(y.shape[0])

        if len(ind1) != 0:
            re2[ind1] = 100

        if im <= 7:
            Weymouth['NN(' + str(nod) + ')'] = wey[0]
            Weymouth['S(' + str(nod) + ')'] = wey[1]
            Balance['NN(' + str(nod) + ')'] = balance[0]
            Balance['S(' + str(nod) + ')'] = balance[1]
            Costos['NN(' + str(nod) + ')'] = np.array(costos)[:, 0]

            PjPi['NN(' + str(nod) + ')'] = re1
            PjPi['S(' + str(nod) + ')'] = re2
        else:
            Weymouth['NN(Col_18)'] = wey[0]
            Weymouth['S(Col_18)'] = wey[1]
            Balance['NN(Col_18)'] = balance[0]
            Balance['S(Col_18)'] = balance[1]
            Costos['NN(Col_18)'] = np.array(costos)[:, 0]

            PjPi['NN(Col_18)'] = re1
            PjPi['S(Col_18)'] = re2

    plots(Balance, Weymouth, PjPi, Costos, s=s)
    return(Balance, Weymouth, PjPi, Costos)


def evaluate_balance_two(FT, W, X):
    """
    Evaluate the Mean Absolute Percentage Error (MAPE) between actual and predicted data.

    Parameters:
    - FT (list of numpy arrays): List of predicted flow data from models.
    - W (numpy array): Weight matrix used for the calculation.
    - X (list of numpy arrays): List of actual flow data for comparison.

    Returns:
    - list: A list containing MAPE values for each corresponding set of predicted and actual data.

    This function computes the MAPE for each set of predicted data in 'FT' against the actual data in 'X'.
    The computation involves a dot product of predicted data with the transpose of weight matrix 'W'.
    The function iterates over the elements of 'FT' and 'X' to compute MAPE for each set of data.
    """
    # List comprehension for more concise and efficient calculation
    return [mape(X[i], FT[i] @ W.T) for i in range(len(FT))]


def evaluate_weymouth_two(X, Y):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) between two sets of flow data.

    Parameters:
    - X (list of numpy arrays): The first set of flow data (usually the actual flow data).
    - Y (list of numpy arrays): The second set of flow data (usually the predicted flow data).

    Returns:
    - list: A list of MAPE values, each corresponding to the MAPE between the elements of X and Y.

    The function iterates over the corresponding elements of X and Y, calculating the MAPE
    for each pair of elements. The results are stored and returned in a list.
    """
    # Using list comprehension for efficient computation
    return [mape(X[i], Y[i]) for i in range(len(X))]


def ng_evaluate_atip(s1='/content/drive/Shareddrives/red_gas_col/', s2='/content/drive/Shareddrives/red_gas_col/Prueba/Data/', s=1):
    """
    Evaluate different neural network models on various gas network cases and plot their performance metrics.

    Parameters:
    - s1 (str): Base directory path for the model and Excel data files.
    - s2 (str): Path for the input data files.
    - s (int): Scenario indicator for choosing different sets of model paths.

    The function processes multiple gas network cases in a loop. For each case, it loads necessary data,
    constructs and loads a neural network model, and predicts outputs. It then evaluates these predictions
    using custom functions for different metrics such as Weymouth formula, balance, cost, and pressure ratios.
    The function visualizes these metrics using boxplot graphs.

    Returns:
    - tuple: A tuple containing lists of evaluation metrics (Balance, Weymouth, PjPi, Costos) for each network case.
    """
    s1_path = Path(s1)
    s2_path = Path(s2)

    if s == 0:
        modelpath = ['model2_8.h5', 'model2_9.h5', 'model2_10.h5', 'model2_11.h5',
                     'model2_12.h5', 'model2_13.h5', 'model2_14.h5', 'model2_15.h5', 'model2_Col.h5']
    elif s == 1:
        modelpath = ['model8.h5', 'model9.h5', 'model10.h5', 'model11.h5',
                     'model12.h5', 'model13.h5', 'model14.h5', 'model15.h5', 'modelCol.h5']
    filepath = ['ng_case8.xlsx', 'ng_case9.xlsx', 'ng_case10.xlsx', 'ng_case11.xlsx',
                'ng_case12.xlsx', 'ng_case13.xlsx', 'ng_case14.xlsx', 'ng_case15.xlsx', 'ng_caseCol_18.xlsx']
    fig, axs = plt.subplots(2, 2, figsize=(29, 20))
    Weymouth = []  # pd.DataFrame()
    Balance = []  # pd.DataFrame()
    Costos = []  # pd.DataFrame()
    PjPi = []  # pd.DataFrame()
    for im in range(9):
        path = s1_path / filepath[im]
        Data_inf = pd.read_excel(path, sheet_name='node.info')
        Data_D = pd.read_excel(path, sheet_name='node.dem')
        Data_Dc = pd.read_excel(path, sheet_name='node.demcost')
        Data_W = pd.read_excel(path, sheet_name='well')
        Data_T = pd.read_excel(path, sheet_name='pipe')
        Data_C = pd.read_excel(path, sheet_name='comp')
        Data_sto = pd.read_excel(path, sheet_name='sto')
        Cost = np.concatenate((Data_W['Cg'].values, Data_T['C_O'].values, Data_C['costc'], Data_Dc['al_Res'].values,
                               Data_Dc['al_Ind'].values, Data_Dc['al_Com'].values, Data_Dc[
                                   'al_NGV'].values, Data_Dc['al_Ref'].values, Data_Dc['al_Pet'].values,
                               Data_sto['C_S+'].values - Data_sto['C_V'].values, -1 * (Data_sto['C_S-'] - Data_sto['C_V']).values, Data_sto['C_V'].values)).reshape(-1, 1)
        w = gen_w(Data_inf, Data_W, Data_T, Data_C, Data_Dc, Data_sto)
        if im == 8:
            X_test = sio.loadmat(
                s2_path / f'inputs_None_{im + 8}.mat')['inputs'][0]

        else:
            X_test = sio.loadmat(
                s2_path / f'inputs_None_{im + 8}.mat')['inputs']

        fd = np.array(X_test).max(axis=0)
        model2 = flow_model(str(path), fd, seeds=1, s=s)
        model2.load_weights(s2_path / modelpath[im])
        model_A = tf.keras.Model(
            inputs=model2.inputs, outputs=model2.get_layer('F').output)
        model_B = tf.keras.Model(
            inputs=model2.inputs, outputs=model2.get_layer('P').output)
        Fe = model_A.predict(np.array(X_test), verbose=False)
        Pe = model_B.predict(np.array(X_test), verbose=False)
        i = Data_T['fnode'].values.astype(int) - 1
        j = Data_T['tnode'].values.astype(int) - 1
        CK = Data_T['Kij'].values.reshape(-1,)

        Fte = Fe[:, 1:1 + len(Data_T)]
        Pie = Pe[:, i]
        Pje = Pe[:, j]
        Pte = np.sign(Pie**2 - Pje**2) * \
            np.sqrt(CK * np.abs(Pie**2 - Pje**2))
        balance = evaluate_balance_two(Fe, w, np.array(X_test))
        wey = evaluate_weymouth_two(Fte, Pte)
        ic = Data_C['fnode'].values - 1
        jc = Data_C['tnode'].values - 1
        Bc = Data_C['fnode'].values
        are = np.round(Pe[:, jc] / Pe[:, ic], 4)
        ind1 = np.unique(np.concatenate((np.where(~(are[:, 0] <= Bc[0]) | ~(are[:, 0] >= 1.0))[
                         # .shape[0]/are.shape[0]*100
                         0], np.where(~(are[:, 1] <= Bc[1]) | ~(are[:, 1] >= 1.0))[0])))
        re1 = np.zeros(X_test.shape[0])
        if len(ind1) != 0:
            re1[ind1] = 100
        costos = (Fe @ Cost[:-1]).flatten()
        Weymouth.append(wey)
        Balance.append(balance)
        Costos.append(costos)
        PjPi.append(re1)

    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[1, 0]
    ax4 = axs[1, 1]

    # ax1.boxplot(Balance)
    ax1.boxplot(Balance, boxprops=dict(color='blue'), medianprops=dict(
        color='orangered'), whiskerprops=dict(color='blue'))
    fontsize = 20
    rotation_angle = 45
    a = ['NN(8)', 'NN(9)', 'NN(10)', 'NN(11)', 'NN(12)',
         'NN(13)', 'NN(14)', 'NN(15)', 'NNCol(16)']
    # ax1.set_title('Balance')
    ax1.set_ylabel('MAPE', fontsize=20)
    ax1.set_yscale('log')
    ax1.set_xlabel('Network dimension', fontsize=20)
    ax1.set_xticklabels(a)
    ax1.tick_params(axis='x', labelsize=fontsize, rotation=rotation_angle)
    ax1.tick_params(axis='y', labelsize=fontsize)

    ax2.boxplot(Weymouth, boxprops=dict(color='blue'), medianprops=dict(
        color='orangered'), whiskerprops=dict(color='blue'))
    # ax2.set_title('Weymouth')
    ax2.set_yscale('log')
    ax2.set_ylabel('MAPE', fontsize=20)
    ax2.set_xlabel('Network dimension', fontsize=20)
    ax2.set_xticklabels(a)
    ax2.tick_params(axis='x', labelsize=fontsize, rotation=rotation_angle)
    ax2.tick_params(axis='y', labelsize=fontsize)

    ax3.boxplot(PjPi, boxprops=dict(color='blue'), medianprops=dict(
        color='orangered'), whiskerprops=dict(color='blue'))
    # ax3.set_title('Pj/Pi')
    ax3.set_ylabel('MAPE', fontsize=20)
    ax3.set_xlabel('Network dimension', fontsize=20)
    ax3.set_xticklabels(a)
    ax3.tick_params(axis='x', labelsize=fontsize, rotation=rotation_angle)
    ax3.tick_params(axis='y', labelsize=fontsize)

    ax4.boxplot(Costos, boxprops=dict(color='blue'), medianprops=dict(
        color='orangered'), whiskerprops=dict(color='blue'))
    # ax4.set_title('Costos')#MAPE---COSTOS
    ax4.set_yscale('log')
    ax4.set_ylabel('Costs', fontsize=20)
    ax4.set_xlabel('Network dimension', fontsize=20)
    ax4.set_xticklabels(a)
    ax4.tick_params(axis='x', labelsize=fontsize, rotation=rotation_angle)
    ax4.tick_params(axis='y', labelsize=fontsize)

    # Ajustar el diseño para evitar solapamientos
    plt.tight_layout()

    # Mostrar la gráfica
    plt.show()
    return(Balance, Weymouth, PjPi, Costos)


def visualize_atipic(dataframe, columns):
    """
    Identifies and visualizes the percentage of outlier values in specified columns of a DataFrame.

    Parameters:
    - dataframe (DataFrame): The DataFrame in which to identify outliers.
    - columns (list): List of column names in the DataFrame to analyze.

    This function utilizes the Interquartile Range (IQR) method to identify outliers.
    A value is considered an outlier if it is below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR,
    where Q1 and Q3 are the first and third quartiles, respectively.

    The function generates a bar plot displaying the percentage of outliers in each column.
    """
    outlier_percentages = []
    colors = []  # List to store colors for each column

    plt.figure(figsize=(20, 8))
    for index, column in enumerate(columns):
        Q1 = dataframe[column].quantile(0.25)
        Q3 = dataframe[column].quantile(0.75)
        IQR = Q3 - Q1
        # Calculate the percentage of outliers
        outliers = ((dataframe[column] < Q1 - 1.5 * IQR) |
                    (dataframe[column] > Q3 + 1.5 * IQR)) & (dataframe[column] >= 1.0)
        percentage = (outliers.sum() / len(dataframe[column])) * 100
        outlier_percentages.append(percentage)
        # Assign color based on index parity
        colors.append('orange' if index % 2 == 0 else 'blue')

    plt.bar(columns, outlier_percentages, color=colors)
    plt.xlabel('Network Dimension', fontsize=20)
    plt.ylabel('Percentage of Outliers', fontsize=20)
    plt.xticks(fontsize=20, rotation=45)
    plt.yticks(fontsize=20)
    plt.show()


def visualize_non_convergence(data_path='/content/drive/Shareddrives/red_gas_col/Prueba/Data/inputs_None_', network_range=(8, 17)):
    """
    Visualizes the percentage of non-convergence cases for different network dimensions.

    Parameters:
    - data_path (str): Path to the directory containing input data files.
    - network_range (tuple): Range of network files to analyze (inclusive).

    The function loads data from specified files and calculates the percentage of
    non-convergence cases. It then generates a bar plot to visually represent this data.
    """

    data_path = Path(data_path)
    percentages = []
    labels = []

    for file_index in range(*network_range):
        file_path = data_path.parent / f'{data_path.name}{file_index}.mat'
        data = sio.loadmat(file_path)['inputs']
        non_convergence_percentage = (data.shape[0] / 1000) * 100
        percentages.append(non_convergence_percentage)

        label = f'N(Col_18)' if file_index == 16 else f'N({data.shape[1]})'
        labels.append(label)

    # Create bar plot
    plt.figure(figsize=(20, 8))
    plt.bar(labels, percentages, color='blue')
    plt.xlabel('Network Dimension', fontsize=15)
    plt.ylabel('Percentage of Non-Convergence', fontsize=15)
    plt.xticks(fontsize=10, rotation=45)
    plt.yticks(fontsize=10)
    plt.show()

def identify_atypical_values(files):
    """
    Identify and visualize the percentage of atypical values in a list of arrays.

    Parameters:
    - files: List of arrays. Each array represents a different data set.

    The function calculates the percentage of values in each array that are greater than 1,
    indicating atypical values. It then visualizes these percentages using a bar plot.
    """
    percentages = []
    labels = []

    # Process each file to calculate the percentage of atypical values
    for i, file in enumerate(files):
        data_array = np.array(file)
        atypical_percentage = (
            data_array[data_array > 1].shape[0] / data_array.shape[0]) * 100
        percentages.append(atypical_percentage)

        # Create labels for the x-axis based on the index of the file
        label = 'R(' + str(i + 8) + ')' if i != 8 else 'N(Col18)'
        labels.append(label)

    # Plotting
    plt.bar(labels, percentages, color='blue')
    plt.xlabel('Network Dimension')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45)
    plt.show()


def dynamic_val(s1='/content/drive/Shareddrives/red_gas_col/', s2='/content/drive/Shareddrives/red_gas_col/Prueba/Data/'):
    """
    Generates sinusoidal behavior and adjusts random data accordingly.

    Returns:
    - X_test: Randomly generated data.
    """
    # Step 1: Generate desired sinusoidal behavior
    x = np.linspace(0, 2 * np.pi, 1000)  # Generates x values from 0 to 2*pi
    amplitude = 150                         # Sine wave amplitude
    frequency = 1                       # Sine wave frequency
    # Generates the sine wave
    sine_wave = abs(amplitude * np.sin(frequency * x))

    # Step 2: Create a matrix of random data
    data = np.random.rand(1000, 18)  # Random data

    # Step 3: Adjust random data
    # The idea is to scale the data so that its sum along the rows equals the corresponding sine wave value
    # Current sum of data along the rows
    data_sum = np.sum(data, axis=1)
    adjustment_factor = sine_wave / data_sum  # Adjustment factor for each row

    s1_path = Path(s1)
    s2_path = Path(s2)

    # Adjust the data
    for i in range(data.shape[1]):
        data[:, i] *= adjustment_factor
    X_test = data

    modelpath = 'modelCol.h5'
    filepath = 'ng_caseCol_18.xlsx'
    path = s1_path / filepath 
    Data_inf = pd.read_excel(path, sheet_name='node.info')
    Data_D = pd.read_excel(path, sheet_name='node.dem')
    Data_Dc = pd.read_excel(path, sheet_name='node.demcost')
    Data_W = pd.read_excel(path, sheet_name='well')
    Data_T = pd.read_excel(path, sheet_name='pipe')
    Data_C = pd.read_excel(path, sheet_name='comp')
    Data_sto = pd.read_excel(path, sheet_name='sto')
    Cost = np.concatenate((Data_W['Cg'].values, Data_T['C_O'].values, Data_C['costc'], Data_Dc['al_Res'].values,
                           Data_Dc['al_Ind'].values, Data_Dc['al_Com'].values, Data_Dc[
                               'al_NGV'].values, Data_Dc['al_Ref'].values, Data_Dc['al_Pet'].values,
                           Data_sto['C_S+'].values - Data_sto['C_V'].values, -1 * (Data_sto['C_S-'] - Data_sto['C_V']).values, Data_sto['C_V'].values)).reshape(-1, 1)
    w = gen_w(Data_inf, Data_W, Data_T, Data_C, Data_Dc, Data_sto)
    fd = X_test.max(axis=0)
    # Assuming Fd, Fw, and data are your data.
    plt.figure(figsize=(15, 5))
    model2 = flow_model(path, fd, seeds=1, s=1)
    model2.load_weights(s2_path / modelpath)
    model_A = tf.keras.Model(inputs=model2.inputs,
                             outputs=model2.get_layer('F').output)
    model_B = tf.keras.Model(inputs=model2.inputs,
                             outputs=model2.get_layer('P').output)
    i = len(Data_W) + len(Data_C) + len(Data_T)
    j = len(Data_inf)

    Fd = model_A.predict(X_test, verbose=False)[:, i:i + j]
    plt.plot(Fd.sum(axis=1), label="Dynamic-Unsupplied Gas")
    Fw = model_A.predict(X_test, verbose=False)[:, 0]
    plt.plot(Fw, label="Dynamic-Gas Supply")
    plt.plot(data.sum(axis=1), label="Total Demand")
    model2 = flow_model(path, fd, seeds=1, s=0)
    modelpath = 'model2_Col.h5'
    model2.load_weights(s2_path / modelpath)
    model_A = tf.keras.Model(inputs=model2.inputs,
                             outputs=model2.get_layer('F').output)
    model_B = tf.keras.Model(inputs=model2.inputs,
                             outputs=model2.get_layer('P').output)
    i = len(Data_W) + len(Data_C) + len(Data_T)
    j = len(Data_inf)
    Fd = model_A.predict(X_test, verbose=False)[:, i:i + j]
    plt.plot(Fd.sum(axis=1), label="Unsupplied Gas")
    plt.xlabel("Samples", fontsize=20)  # Adds label on x-axis
    plt.ylabel("Flows", fontsize=20)    # Adds label on y-axis
    plt.legend()           # Displays legend on the plot
    # Adjust here the size for the X axis tick labels
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()


def loss_val():
    """
    Visualizes and compares differences in performance metrics of various models using box plots.

    This function loads performance data from different models from the files Weymouth.xlsx, Balance.xlsx, and Costos.xlsx.
    Then, box plots are generated for each of these metrics, showing how the performance values are distributed among the different models.

    The plots display the Mean Absolute Percentage Error (MAPE) and costs, facilitating visual comparison between the models.
    A logarithmic scale is used to ease the visualization of data that may have a wide range of values.

    The function has no input parameters and does not return any value. The results are directly displayed as plots.
    """

    W = pd.read_csv(
        '/content/drive/Shareddrives/red_gas_col/Prueba/Data/Weymouth.xlsx')
    W.drop(['Unnamed: 0'], axis=1, inplace=True)
    B = pd.read_csv(
        '/content/drive/Shareddrives/red_gas_col/Prueba/Data/Balance.xlsx')
    B.drop(['Unnamed: 0'], axis=1, inplace=True)
    C = pd.read_csv(
        '/content/drive/Shareddrives/red_gas_col/Prueba/Data/Costos.xlsx')
    C.drop(['Unnamed: 0'], axis=1, inplace=True)
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    fontsize = 10
    rotation_angle = 45
    # Organizar los datos en la gráfica y establecer escala logarítmica
    B.boxplot(ax=axs[0], boxprops=dict(color='blue'), medianprops=dict(
        color='orangered'), whiskerprops=dict(color='blue'))
    axs[0].set_yscale('log')
    axs[0].set_ylabel('MAPE', fontsize=15)
    axs[0].set_xlabel('Loss Functions', fontsize=15)
    axs[0].tick_params(axis='x', labelsize=fontsize, rotation=rotation_angle)
    axs[0].tick_params(axis='y', labelsize=fontsize)
    # for label in axs[0, 0].get_xticklabels():
    #    label.set_rotation(rotation_angle)

    W.boxplot(ax=axs[1], boxprops=dict(color='blue'), medianprops=dict(
        color='orangered'), whiskerprops=dict(color='blue'))
    #axs[0, 1].set_title('Weymouth')
    axs[1].set_yscale('log')
    axs[1].set_ylabel('MAPE', fontsize=15)
    axs[1].set_xlabel('Loss Functions', fontsize=15)
    axs[1].tick_params(axis='x', labelsize=fontsize, rotation=rotation_angle)
    axs[1].tick_params(axis='y', labelsize=fontsize)
    # for label in axs[0, 1].get_xticklabels():

    C.boxplot(ax=axs[2], boxprops=dict(color='blue'), medianprops=dict(
        color='orangered'), whiskerprops=dict(color='blue'))
    #axs[1, 1].set_title('Costos')
    axs[2].set_yscale('symlog')
    axs[2].set_ylabel('Costs', fontsize=15)
    axs[2].set_xlabel('Loss Functions', fontsize=15)
    axs[2].tick_params(axis='x', labelsize=fontsize, rotation=rotation_angle)
    axs[2].tick_params(axis='y', labelsize=fontsize)


def bounded(s1='/content/drive/Shareddrives/red_gas_col/', s2='/content/drive/Shareddrives/red_gas_col/Prueba/Data/', s=1):
    """
     Generates box plots to visualize and compare performance metric differences across various models.

     This function loads performance data of different models from the Weymouth.xlsx, Balance.xlsx, and Costos.xlsx files.
     Then, box plots are generated for each of these metrics, showing how the performance values are distributed
     among the different models.

     The plots display the Mean Absolute Percentage Error (MAPE) and costs, facilitating visual comparison between the models.
     A logarithmic scale is used to ease the visualization of data that may have a wide range of values.

     Parameters
     ----------
     s1 : str, optional
         Path to the directory containing input data files. Default is '/content/drive/Shareddrives/red_gas_col/'.
     s2 : str, optional
         Path to the directory containing model data files. Default is '/content/drive/Shareddrives/red_gas_col/Prueba/Data/'.
     s : int, optional
         Determines the type of model. If 0, uses one set of models, if 1, uses another set. Default is 1.

     Returns
     -------
     None
         The results are directly displayed as plots.
     """
    s1_path = Path(s1)
    s2_path = Path(s2)

    X_test = load_data(s2_path / 'inputs_P.mat', 'inputs')
    y_test = load_data(s2_path / 'outputs_P.mat', 'outputs')
    if s == 0:
        modelpath = ['model2_8.h5', 'model2_9.h5', 'model2_10.h5', 'model2_11.h5',
                     'model2_12.h5', 'model2_13.h5', 'model2_14.h5', 'model2_15.h5', 'model2_Col.h5']
    elif s == 1:
        modelpath = ['model8.h5', 'model9.h5', 'model10.h5', 'model11.h5',
                     'model12.h5', 'model13.h5', 'model14.h5', 'model15.h5', 'modelCol.h5']
    filepath = ['ng_case8.xlsx', 'ng_case9.xlsx', 'ng_case10.xlsx', 'ng_case11.xlsx','ng_case12.xlsx', 'ng_case13.xlsx', 'ng_case14.xlsx', 'ng_case15.xlsx', 'ng_caseCol_18.xlsx']
    Weymouth = pd.DataFrame()
    Balance = pd.DataFrame()
    Costos = pd.DataFrame()
    PjPi = pd.DataFrame()
    im = 8
    path = s1_path / filepath[im]
    Data_inf = pd.read_excel(path, sheet_name='node.info')
    Data_D = pd.read_excel(path, sheet_name='node.dem')
    Data_Dc = pd.read_excel(path, sheet_name='node.demcost')
    Data_W = pd.read_excel(path, sheet_name='well')
    Data_T = pd.read_excel(path, sheet_name='pipe')
    Data_C = pd.read_excel(path, sheet_name='comp')
    Data_sto = pd.read_excel(path, sheet_name='sto')
    Cost = np.concatenate((Data_W['Cg'].values, Data_T['C_O'].values, Data_C['costc'], Data_Dc['al_Res'].values,
                           Data_Dc['al_Ind'].values, Data_Dc['al_Com'].values, Data_Dc[
                               'al_NGV'].values, Data_Dc['al_Ref'].values, Data_Dc['al_Pet'].values,
                           Data_sto['C_S+'].values - Data_sto['C_V'].values, -1 * (Data_sto['C_S-'] - Data_sto['C_V']).values, Data_sto['C_V'].values)).reshape(-1, 1)
    w = gen_w(Data_inf, Data_W, Data_T, Data_C, Data_Dc, Data_sto)
    y = np.array(y_test[im])
    fd = np.array(X_test[im]).max(axis=0)
    model2 = flow_model(path, fd, seeds=1, s=s)
    model2.load_weights(s2_path / modelpath[im])
    model_A = tf.keras.Model(inputs=model2.inputs,
                             outputs=model2.get_layer('F').output)
    model_B = tf.keras.Model(inputs=model2.inputs,
                             outputs=model2.get_layer('P').output)
    Fe = model_A.predict(np.array(X_test[im]), verbose=False)
    Pe = model_B.predict(np.array(X_test[im]), verbose=False)
    i = Data_T['fnode'].values.astype(int) - 1
    j = Data_T['tnode'].values.astype(int) - 1
    CK = Data_T['Kij'].values.reshape(-1,)
    nod = len(Data_inf)
    Fte = Fe[:, 1:1 + len(Data_T)]
    Fce = Fe[:, 1 + len(Data_T):1 + len(Data_T) + len(Data_C)]
    Fwe = Fe[:, 1:2]
    fig, axs = plt.subplots(2, 2, figsize=(29, 20))
    fontsize = 20
    rotation_angle = 45
    # Organizar los datos en la gráfica y establecer escala logarítmica
    axs[0, 0].boxplot(Fwe, boxprops=dict(color='blue'), medianprops=dict(
        color='orangered'), whiskerprops=dict(color='blue'))
    axs[0, 0].set_yscale('symlog')
    axs[0, 0].plot(1, 80, 'k*', markersize=15, label='Injection Limit')
    axs[0, 0].plot(1, 0, 'k*', markersize=15)
    axs[0, 0].legend()
    axs[0, 0].legend(fontsize=20)
    axs[0, 0].set_ylabel('Gas injection', fontsize=20)
    axs[0, 0].set_xlabel('Well node', fontsize=20)
    axs[0, 0].tick_params(axis='x', labelsize=fontsize,
                          rotation=rotation_angle)
    axs[0, 0].tick_params(axis='y', labelsize=fontsize)

    # for label in axs[0, 0].get_xticklabels():
    #    label.set_rotation(rotation_angle)

    Li = Data_T['Fg_min'].values
    Ls = Data_T['Fg_max'].values

    axs[0, 1].boxplot(Fte, boxprops=dict(color='blue'), medianprops=dict(
        color='orangered'), whiskerprops=dict(color='blue'))
    axs[0, 1].set_xticklabels(np.arange(2, 2 + len(Data_T)))
    #axs[0, 1].set_title('Weymouth')
    axs[0, 1].set_yscale('symlog')
    axs[0, 1].plot(np.arange(1, 1 + len(Data_T)), Ls,
                   'k*', markersize=15, label='Flow limit')
    axs[0, 1].plot(np.arange(1, 1 + len(Data_T)), Li, 'k*', markersize=15)
    axs[0, 1].legend()
    axs[0, 1].legend(fontsize=20)
    axs[0, 1].set_ylabel('Gas flow', fontsize=20)
    axs[0, 1].set_xlabel('Pipeline node', fontsize=20)
    axs[0, 1].tick_params(axis='x', labelsize=fontsize,
                          rotation=rotation_angle)
    axs[0, 1].tick_params(axis='y', labelsize=fontsize)
    # for label in axs[0, 1].get_xticklabels():

    Li = [0] * len(Data_C)
    Ls = Data_C['fmaxc'].values
    axs[1, 0].boxplot(Fce, boxprops=dict(color='blue'), medianprops=dict(
        color='orangered'), whiskerprops=dict(color='blue'))
    #axs[1, 1].set_title('Costos')
    axs[1, 0].set_xticklabels(
        np.arange(2 + len(Data_T), 2 + len(Data_T) + len(Data_C)))
    axs[1, 0].set_yscale('symlog')
    axs[1, 0].plot(np.arange(1, 4), Ls, 'k*',
                   markersize=15, label='Flow limit')
    axs[1, 0].plot(np.arange(1, 4), Li, 'k*', markersize=15)
    axs[1, 0].legend()
    axs[1, 0].legend(fontsize=20)
    axs[1, 0].set_ylabel('Gas flow', fontsize=20)
    axs[1, 0].set_xlabel('Compressor node', fontsize=20)
    axs[1, 0].tick_params(axis='x', labelsize=fontsize,
                          rotation=rotation_angle)
    axs[1, 0].tick_params(axis='y', labelsize=fontsize)
    Li = Data_inf['Pmin'].values
    Ls = Data_inf['Pmax'].values
    axs[1, 1].boxplot(Pe, boxprops=dict(color='blue'), medianprops=dict(
        color='orangered'), whiskerprops=dict(color='blue'))
    #axs[1, 1].set_title('Costos')
    #axs[1, 1].set_yscale('log')
    axs[1, 1].plot(np.arange(1, 19), Ls, 'k*',
                   markersize=15, label='Pressure limit')
    axs[1, 1].legend()
    axs[1, 1].legend(fontsize=20)
    axs[1, 1].set_ylabel('Pressure[psia]', fontsize=20)
    axs[1, 1].set_xlabel('Node', fontsize=20)
    axs[1, 1].tick_params(axis='x', labelsize=fontsize,
                          rotation=rotation_angle)
    axs[1, 1].tick_params(axis='y', labelsize=fontsize)

