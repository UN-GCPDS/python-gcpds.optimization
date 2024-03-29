"""
========================================================
Solver Applications in Optimization and Network Modeling
========================================================


The provided code constitutes a comprehensive framework for research and development in the fields of artificial intelligence and computational biology. Its architecture consists of specialized modules addressing key aspects of modeling, machine learning, and optimization, with an emphasis on code robustness and reusability.

From a technical perspective, the code includes:

Training Data Generation: Functionality for generating synthetic data and manipulating real datasets, with advanced options for controlled noise introduction and simulation of specific scenarios.

Machine Learning Model Development: Tools for constructing and customizing machine learning models, including deep neural network architectures and supervised and unsupervised learning algorithms.

Performance Evaluation: Capabilities for systematically evaluating model performance under various conditions, using statistical metrics and visualizations to analyze prediction quality and generalization ability.

Optimization Solver Comparison: Functionality for comparing different optimization methods used in solving specific problems, with tools for measuring convergence, computational efficiency, and scalability.

The underlying technical approach is based on the efficient implementation of algorithms and data structures, leveraging high-performance software libraries, and adopting software engineering practices such as modularity, encapsulation, and detailed documentation.




"""


import os
import warnings
from typing import Any, Callable, List, Tuple, Optional

import keras.backend as K
import numpy as np
import pandas as pd
import scipy.io as sio
import tensorflow as tf
import cvxpy as cp
from google.colab import drive
from keras import Model, backend as k, layers, regularizers
from keras.layers import (
    Activation, Concatenate, Conv1D, Conv2D, Dense, Dropout, Flatten, Input,
    Lambda, MaxPool1D, MaxPooling2D, ReLU, Reshape, BatchNormalization, Dot
)
from keras.models import Sequential
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import linprog
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import Callback
from tensorflow.python.ops.nn_ops import softmax
from IPython import display
from typing import List, Optional
# Metrics aliases for clarity
mse = mean_squared_error
mae = mean_absolute_error
mape = mean_absolute_percentage_error

# Suppress warnings
warnings.filterwarnings("ignore")


def add_noise(data: np.ndarray, snr: float, mu: float = 0.0) -> np.ndarray:
    """
    Generate noise to be added to a signal with a specified signal-to-noise ratio (SNR).

    Parameters
    ----------
    data : np.ndarray
        The original signal data.
    snr : float
        The desired signal-to-noise ratio in decibels.
    mu : float, optional
        The mean value for the noise generation, default is 0.0.

    Returns
    -------
    np.ndarray
        The generated noise array with the same shape as the input data.
    """

    mean_signal = abs(np.mean(data))
    signal_db = 10 * np.log10(mean_signal)
    noise_db = signal_db - snr
    noise_power = 10 ** (noise_db / 10)
    return np.random.normal(mu, np.sqrt(noise_power), data.shape)


def probability_vector(num_elements: int = 5) -> np.ndarray:
    """
    Generate an array of size num_elements with elements rounded to two decimal places
    such that the sum is 1.0.

    Parameters
    ----------
    num_elements : int, optional
        Number of elements in the generated array. Default is 5.

    Returns
    -------
    np.ndarray
        A 1-D array of shape (num_elements, 1) with elements summing to 1.0.
    """

    probabilities = np.empty(0)

    while probabilities.size != num_elements:

        random_probabilities = np.round(
            np.random.uniform(0.1, 1, num_elements), 2)

        if np.sum(random_probabilities) == 1.0:
            probabilities = random_probabilities

    return probabilities.reshape(num_elements, 1)


def random_matrix(num_ranges: int = 5, num_rows: int = 10) -> np.ndarray:
    """
    Generate a matrix with uniformly distributed random values in the range [i-1, i],
    where i iterates from 1 to num_ranges, as columns, and num_rows rows.

    Parameters
    ----------
    num_ranges : int, optional
        Number of different ranges to use for each column. Default is 5.
    num_rows : int, optional
        Number of rows in the generated matrix. Default is 10.

    Returns
    -------
    np.ndarray
        A (num_rows, num_ranges) matrix of uniformly distributed random values.
    """

    matrix = np.zeros((num_rows, num_ranges))
    for i in range(1, num_ranges + 1):
        matrix[:, i - 1] = np.random.uniform(i - 1, i, num_rows)
    return matrix


class SoftmaxWeightConstraint(tf.keras.constraints.Constraint):
    """Constraint class that applies a softmax to the weights."""

    def __init__(self, softmax_function: Callable[[tf.Tensor, int], tf.Tensor]):
        """
        Initializes the constraint using a given softmax function.

        Parameters
        ----------
        softmax_function : Callable[[tf.Tensor, int], tf.Tensor]
            The softmax function to be applied on the weights.
        """

        self.softmax_function = softmax_function

    def __call__(self, weights: tf.Tensor) -> tf.Tensor:
        """
        Applies softmax on the weights along the specified axis.

        Parameters
        ----------
        weights : tf.Tensor
            The tensor to which the softmax function will be applied.

        Returns
        -------
        tf.Tensor
            The tensor with softmax applied.
        """

        return self.softmax_function(weights, axis=0)

    def get_config(self) -> dict:
        """
        Gets the configuration of the constraint.

        Returns
        -------
        dict
            A dictionary containing the configuration of the constraint.
        """

        return {'softmax_function': self.softmax_function}


def minimize_l1_norm(y: np.ndarray, u: np.ndarray, s: int, solver: str = 'CPLEX') -> np.ndarray:
    """
    Solve the optimization problem to find vector z that minimizes the L1 norm
    between y and the product of matrix u and vector z, with constraints that z
    sums to 1 and each entry in z is between 0 and 1 inclusive.

    Parameters
    ----------
    y : np.ndarray
        The observed data vector (n by 1).
    u : np.ndarray
        The design matrix (n by s).
    s : int
        The length of the solution vector z.
    solver : str, optional
        The solver used for the optimization problem, default is 'CPLEX'.

    Returns
    -------
    np.ndarray
        The solution vector z with shape (s, 1).
    """

    z = cp.Variable((s, 1))
    objective = cp.Minimize(cp.norm(y - u @ z, p=1))
    constraints = [cp.sum(z) == 1, z >= 0, z <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver)
    Z_solution = z.value
    return Z_solution.reshape(s, 1)


class Evaluate:
    """
    A class to handle the evaluation of models with respect to their Mean Absolute Percentage Error (MAPE) 
    metrics on various noise levels, using different solvers for optimization problems.

    Attributes
    ----------
    result_path : str
        The path where results should be saved.
    model_columns : List[str]
        List of column names for the models.
    solvers : np.ndarray
        An array of solver names.
    signal_to_noise_levels : List[Optional[int]]
        List of signal-to-noise ratios to be evaluated, including `None` for the original signal.
    """


    def __init__(self, result_path: str = ''):
        """
        Initialize the Evaluate class with default parameters.

        Parameters
        ----------
        result_path : str, optional
            The path where results should be saved. Defaults to an empty string.
        """

        self.model_columns: List[str] = [
            'NN(-1)', 'S(-1)', 'NN(1)', 'S(1)',
            'NN(3)', 'S(3)', 'NN(5)', 'S(5)', 'NN(N)', 'S(N)'
        ]
        self.solvers: np.ndarray = np.array(
            ['CLARABEL', 'GUROBI', 'MOSEK', 'XPRESS', 'SCS'])
        self.signal_to_noise_levels: List[Optional[int]
            ] = [-1, 1, 3, 5, None]

    def create_custom_model(self, input_size: int = 5, num_layers: int = 5, dropout_rate: float = 0.1, regularization_factor: float = 1e-4, learning_rate: float = 1e-3, loss: str = 'huber') -> Model:
        """
        Create a custom Keras model with specified parameters and constraints.

        Parameters
        ----------
        input_size : int, optional
            The size of the input layer. Default is 5.
        num_layers : int, optional
            The number of neurons in each dense layer. Default is 5.
        dropout_rate : float, optional
            The dropout rate for regularization. Default is 0.1.
        regularization_factor : float, optional
            The regularization factor for L1/L2 regularization. Default is 1e-4.
        learning_rate : float, optional
            The learning rate for the optimizer. Default is 1e-3.
        loss : str, optional
            The loss function selector. Default is 'huber'.

        Returns
        -------
        Model
            A compiled Keras model.
        """


        input_tensor = Input(shape=(input_size,))
        loss = loss
        # Initialize the dense layer parameters
        dense_kwargs = {
            'activation': 'relu',
            'kernel_constraint': SoftmaxWeightConstraint(tf.nn.softmax)
        }

        # Build dense layers with constraints
        x = input_tensor
        for _ in range(num_layers):
            x = Dense(units=num_layers, **dense_kwargs)(x)
            x = Dropout(rate=dropout_rate)(x)

        # Output layer with softmax weight constraint and no bias
        output_tensor = Dense(
            units=1, activation=None, use_bias=False,
            kernel_constraint=SoftmaxWeightConstraint(tf.nn.softmax)
        )(x)

        # Create and compile the model
        model = Model(inputs=input_tensor, outputs=output_tensor)
        model.compile(
            loss = loss,
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        )

        return model

    def ModelEval(self, output_path: str = '') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Evaluate the model performance by measuring Mean Absolute Percentage Error (MAPE)
        on different levels of noise and save the results to Excel files.

        Parameters
        ----------
        output_path : str, optional
            The directory path where Excel files will be saved. Defaults to an empty string.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing two pandas DataFrames of MAPE for y and pi respectively.
        """


        MAPE_y = pd.DataFrame(columns=self.model_columns)
        MAPE_pi = pd.DataFrame(columns=self.model_columns)

        for j in range(len(self.signal_to_noise_levels)):
            MSE_y: List[float] = []
            MSE_pi: List[float] = []

            for i in range(self.num_samples):

                true_pic = self.PIC[i]
                u = self.U[i]
                y = self.Y[j, i]

                modeloMax = self.create_custom_model()

                history = modeloMax.fit(u[0:400], y[0:400], epochs=100, batch_size=32,
                                        verbose=False, validation_split=0.3, steps_per_epoch=2)

                y_re = modeloMax.predict(u[400:], verbose=False)
                pic_re = modeloMax.layers[-1].get_weights()[0]
                y = self.Y[-1, i]

                MSE_y.append(mape(y[400:], y_re))
                MSE_pi.append(mape(true_pic, pic_re))

            MAPE_y[self.model_columns[j * 2]] = MSE_y
            MAPE_pi[self.model_columns[j * 2]] = MSE_pi

        MAPE_y.to_excel(f'{output_path}Modely.xlsx', index=False)
        MAPE_pi.to_excel(f'{output_path}Modelpi.xlsx', index=False)
        history_loss = pd.DataFrame(history.history)
        history_loss.to_excel(f'{output_path}hystory_loss'+self.loss+'.xlsx', index=False)
      
        return MAPE_y, MAPE_pi

    
    
    def history_plot(self) -> pd.DataFrame:
        """
        Plot the training history for different loss functions.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the history of the model's loss and validation loss.
        """


        # Obtener los datos de entrada y salida
        u = self.U[-1]
        y = self.Y[-1, 0]

        # Inicializar una lista para almacenar el historial de pérdidas
        history_loss = []

        # Crear tres figuras independientes para cada función de pérdida
        fig, axs = plt.subplots(3, 1, figsize=(10, 18))

        # Iterar sobre diferentes funciones de pérdida
        for i, loss_function in enumerate(['mse', 'mae', 'huber']):
            # Crear un modelo con la función de pérdida actual
            modeloMax = self.create_custom_model(loss=loss_function)

            # Entrenar el modelo y obtener el historial de pérdidas
            history = modeloMax.fit(u[0:400], y[0:400], epochs=100, batch_size=32,
                                    verbose=False, validation_split=0.3, steps_per_epoch=2)

            # Convertir el historial de pérdidas a un DataFrame y agregarlo a la lista
            history_loss.append(pd.DataFrame(history.history))

            # Trazar la pérdida de entrenamiento y validación para la función de pérdida actual
            axs[i].plot(history_loss[-1]['loss'], label=loss_function)
            axs[i].plot(history_loss[-1]['val_loss'], label='val_' + loss_function)

            # Agregar leyendas y etiquetas de ejes para cada gráfico
            axs[i].legend()
            axs[i].set_xlabel('Epoch')
            axs[i].set_ylabel('Error')
            axs[i].set_title(f'Training and Validation Loss for {loss_function.upper()}')

        # Ajustar el espaciado entre subgráficos
        plt.tight_layout()

        # Mostrar las gráficas
        plt.show()

  

        
      
        

    def evaluate_solver(self, solver: str, MAPE_y: pd.DataFrame, MAPE_pi: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Evaluate the solver performance by comparing the predicted results against the true values
        and update the Mean Absolute Percentage Error (MAPE) dataframes for y and pi variables.

        Parameters
        ----------
        solver : str
            The name of the solver used for the optimization problem.
        MAPE_y : pd.DataFrame
            The DataFrame containing existing MAPE values for y.
        MAPE_pi : pd.DataFrame
            The DataFrame containing existing MAPE values for pi.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Updated MAPE dataframes for y and pi variables.
        """


        for j, noise_level in enumerate(self.signal_to_noise_levels):
            MSE_s1: List[float] = []
            MSE_s2: List[float] = []
            solver_index = np.where(self.solvers == solver)[0][0]
            for i in range(self.num_samples):
                true_pic = self.PIC[i]
                u = self.U[i]
                y = self.Y[-1][i]
                z = self.Z[j][solver_index][i]
                MSE_s1.append(mean_absolute_percentage_error(y[400:], u[400:] @ z))
                MSE_s2.append(mean_absolute_percentage_error(true_pic, z))
            MAPE_y[self.model_columns[j * 2 + 1]] = MSE_s1
            MAPE_pi[self.model_columns[j * 2 + 1]] = MSE_s2
        return MAPE_y, MAPE_pi

    def plot(self, solver: str, path: str = '') -> None:
        """
        Plot the Mean Absolute Percentage Error (MAPE) for model predictions using a specified solver.

        Parameters
        ----------
        solver : str
            The solver used for the optimization problem.
        path : str, optional
            The directory path where the data files are located. Defaults to an empty string.

        Raises
        ------
        FileNotFoundError
            If the path does not contain the expected files.
        """

        self.PIC = np.load(path + 'PIC.npy')
        self.U = np.load(path + 'U.npy')
        self.Y = np.load(path + 'Y.npy')
        self.Z = np.load(path + 'Z.npy')
        self.num_samples = self.U.shape[0]

        # Attempt to load existing MAPE results; calculate if not available
        try:
            MAPE_y = pd.read_excel(path + 'Modely.xlsx')
            MAPE_pi = pd.read_excel(path + 'Modelpi.xlsx')
        except FileNotFoundError:
            MAPE_y, MAPE_pi = self.ModelEval()

        # Evaluate using the specified solver
        MAPE_y, MAPE_pi = self.evaluate_solver(solver, MAPE_y, MAPE_pi)

        # Reverse the order of the columns for plotting
        MAPE_y = MAPE_y.loc[:, ::-1]
        MAPE_pi = MAPE_pi.loc[:, ::-1]

        # Set up plots
        fig = plt.figure(figsize=(19, 9))
        gs = GridSpec(nrows=2, ncols=2)
        
        # Plot MAPE for y
        ax0 = fig.add_subplot(gs[:, 0])
        ax0.set_yscale('log')
        ax0.set_ylabel('MAPE', fontsize=20)
        ax0.set_xlabel('Noise levels', fontsize=20)
        MAPE_y.boxplot(ax=ax0, boxprops=dict(color='blue'), medianprops=dict(color='orangered'), whiskerprops=dict(color='blue'))
        ax0.tick_params(axis='x', labelsize=20, rotation=45)
        ax0.tick_params(axis='y', labelsize=20)
        ax0.set_ylim([1e-12, 1e0])

        # Plot MAPE for pi
        ax1 = fig.add_subplot(gs[:, 1])
        ax1.set_yscale('log')
        ax1.set_ylabel('MAPE', fontsize=20)
        ax1.set_xlabel('Noise levels', fontsize=20)
        MAPE_pi.boxplot(ax=ax1, boxprops=dict(color='blue'), medianprops=dict(color='orangered'), whiskerprops=dict(color='blue'))
        ax1.tick_params(axis='x', labelsize=20, rotation=45)
        ax1.tick_params(axis='y', labelsize=20)
        ax1.set_ylim([1e-11, 1e1])
        
        # Display the plots
        plt.show()

    def data_generate(self, M: int = 10, N: int = 120, s: int = 5) -> None:
        """
        Generate data sets for the optimization problem with given parameters M, N, and s.

        Parameters
        ----------
        M : int, optional
            Number of samples to generate. Default is 10.
        N : int, optional
            The number of observations for each sample. Default is 120.
        s : int, optional
            The number of elements in the PIC vector. Default is 5.

        Side effects
        ------------
        Saves arrays to disk (`PIC.npy`, `Z.npy`, `U.npy`, `Y.npy`).
        Prints the shapes of the generated arrays to the console.
        """

        columns = ['CLARABEL', 'GUROBI', 'MOSEK', 'XPRESS', 'SCS']
        SNR = [-1, 1, 3, 5, None]
        PIC = np.zeros((M, s, 1))
        U = np.zeros((M, N, s))
        Y = np.zeros((len(SNR), M, N, 1))
        Z = np.zeros((len(SNR), len(columns), M, s, 1))
        for j in range(M):
            pic = probability_vector(s)
            u = random_matrix(s, N)
            for k, snr_level in enumerate(SNR):
                y = u @ pic
                if snr_level is not None:
                    y += add_noise(y, snr_level)
                U[j] = u
                PIC[j] = pic
                Y[k, j] = y
                for i, solver in enumerate(columns):
                    z = minimize_l1_norm(y, u, s, solver=solver)
                    Z[k, i, j] = z
        np.save('PIC.npy', PIC)
        np.save('Z.npy', Z)
        np.save('U.npy', U)
        np.save('Y.npy', Y)
        print(np.array(PIC).shape, np.array(Z).shape, np.array(U).shape, np.array(Y).shape)


class FlyEvaluate:
    """
    A class designed to evaluate a population growth model for fruit flies, and compare
    the performance of different solvers.
    """

    def __init__(self, N: int = 100, C: int = 500, t: int = 1) -> None:
        """
        Initialize FlyEvaluate with default parameters.

        Parameters
        ----------
        N : int, optional
            Number of simulation runs. Default is 100.
        C : int, optional
            Number of cycles in the simulation. Default is 500.
        t : int, optional
            Time lag in the model. Default is 1.
        """

        self.C = C
        self.N = N
        self.t = t
        self.solver = np.array([
            'CLARABEL', 'GUROBI', 'MOSEK', 'XPRESS', 'SCS'
        ])

    def parameters(self) -> Tuple[float, float, float, float, float, float, float]:
        """
        Generate parameters for the growth model using random normal distributions.

        Returns
        -------
        Tuple[float, float, float, float, float, float, float]
            A tuple of parameters (P, d, N0, od, op, e1, e2).
        """

        P = np.random.normal(2, 2**2)
        d = np.random.normal(-1.8, 0.4**2)
        N0 = np.random.normal(6, 0.5**2)
        od = np.random.normal(-0.75, 1**2)
        op = np.random.normal(-0.5, 1**2)
        e1 = 1
        e2 = 1
        return P, d, N0, od, op, e1, e2

    def fly_data(
        self, P: float, N0: float, d: float, e2: float, e1: float, C: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data for the fly population growth and corresponding weight matrices.

        Parameters
        ----------
        P : float
            Growth rate.
        N0 : float
            Initial population.
        d : float
            Death rate.
        e2 : float
            Error term for the death rate.
        e1 : float
            Error term for the growth rate.
        C : int
            Number of cycles.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of N (fly population) and W (weights) both as numpy arrays.
        """

        N: List[float] = [0] * self.t
        N.append(N0)
        W: List[List[float]] = []

        for i in range(self.t, C - 1):
            N.append(P * N[i - self.t] * np.exp(-N[i - self.t] / N0) * e1 + N[i] * np.exp(d * e2))
            aux = [0.0] * (self.t + 1)
            aux[0] = np.exp(d * e2)
            aux[-1] = P * np.exp(-N[i - self.t] / N0) * e1
            W.append(aux)

        

        return np.array(N), np.array(W)

    def generate(self) -> None:
        """
        Generate and save data for fly population growth, as well as the optimization
        problems for different solvers.
        """
        N=[];W=[];Y=[];X=[]
        Z=np.ones((self.solver.shape[0],self.N,self.C-1,self.t+1))*1000
        for j in range(self.N):
            #Z.append([])
            p,d,n0,_,_,e1,e2=self.parameters()
            n,w=self.Flydata(p,n0,d,e2,e1,self.C)
            suma=0
            y=[]
            xx=[]
            for i in range(self.t,self.C-1):
                #Z[j].append([])
                x=[]
                for k in range(i,i-self.t-1,-1):
                    suma=suma+1
                    x.append(n[k])
                x=np.array([x])
                Ps=x.T@x
                Qs=-2*(n[i+1]*x).T
                y.append(n[i+1])
                r0=n[i+1]*n[i+1]
                for s in range(len(self.solver)):
                    #Z[j][i-self.t].append([])
                    z=cp.Variable(self.t+1)
                    #obj=cp.Minimize(0.5*cp.quad_form(z,Ps)+Qs.T@z+r0)
                    #prob=cp.Problem(obj)
                    #z=cp.Variable((s,1))
                    obj=cp.Minimize(cp.norm(y-x@z,p=2))
                    prob=cp.Problem(obj)
                    #prob.solve(solver=solver)
                    try:
                        prob.solve(solver=self.solver[s])
                        #Z[s][j][i]=np.array(z.value)
                        Z[s][j][i-self.t]=z.value
                    except:
                        Z[s][j][i-self.t]=np.array([0,0])
                    #print(Z[j][i-self.t][s])
                xx.append(x[0])
            W.append(w)
            Y.append(y)
            X.append(xx)
        W=np.array(W)
        Z=np.array(Z)[:, :,:-1,:]
        Y=np.array(Y)
        X=np.array(X)
        np.save('FW.npy',W)
        np.save('FZ.npy',Z)
        np.save('FY.npy',Y)
        np.save('FX.npy',X)
        print(W.shape,Z.shape,Y.shape,X.shape)

    def model_op(self, loss: str ='huber') -> Model:
        """
        Create a Keras model with a set of dense layers and custom regularization.

        Parameters:
        - loss (str): The loss function selector. Defaults to 'huber'.

        Returns:
        - Model: A compiled Keras model.
        """

        input_tensor = Input(shape=(2,))
        regularization_strength = 1e-1
        learning_rate = 1e-3
        layer = input_tensor
        for i in range(1, 5):
            layer = Dense(
                32, 
                activation='selu', 
                kernel_regularizer=regularizers.L1L2(
                    l1=regularization_strength, 
                    l2=regularization_strength
                ),
                name=f'Dense_{i}'
            )(layer)
            
        weights = Dense(
            2, 
            activation='selu', 
            kernel_regularizer=regularizers.L1L2(
                l1=regularization_strength, 
                l2=regularization_strength
            ), 
            name='W'
        )(layer)
        
        output_tensor = tf.reduce_sum(tf.multiply(weights, input_tensor), axis=-1)
        output_tensor = Reshape((1,))(output_tensor)
        model = Model(inputs=input_tensor, outputs=output_tensor)
        model.compile(loss = loss, optimizer=tf.keras.optimizers.Adam(learning_rate))
        return model

    def model_eval(self, path: str = '') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Evaluate the model and compute Mean Absolute Percentage Error (MAPE) for predictions.

        Parameters:
        - path (str): The directory path to load and save data files. Defaults to an empty string.

        Returns:
        - Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two pandas DataFrames with MAPE for predictions and weights.
        """

        columns = ['Red', 'Solver']
        mape_y = pd.DataFrame(columns=columns)
        mape_w = pd.DataFrame(columns=columns)
        fy = np.load(path + 'FY.npy', allow_pickle=True)
        fx = np.load(path + 'FX.npy', allow_pickle=True)
        fw = np.load(path + 'FW.npy', allow_pickle=True)
        mse_r1 = []
        mse_r2 = []

        for j in range(fx.shape[0]):
            model_max = self.model_op()
            history = model_max.fit(
                fx[j, 0:400], fy[j][0:400], epochs=400,
                batch_size=32, verbose=False, validation_split=0.3
            )
            ye = model_max.predict(fx[j, 400:], verbose=False)
            mse_r2.append(mape(fy[j, 400:], ye))
            weights_model = tf.keras.Model(
                inputs=model_max.inputs, outputs=model_max.get_layer('W').output
            )
            mse_r1.append(
                mape(fw[j, 400:], weights_model.predict(fx[j, 400:], verbose=False))
            )

        mape_w[columns[0]] = mse_r1
        mape_y[columns[0]] = mse_r2
        mape_y.to_excel(path + 'Modelfly_y.xlsx', index=False)
        mape_w.to_excel(path + 'Modelfly_w.xlsx', index=False)
        return mape_w, mape_y

    def evaluate_solver(
        self,
        solver: str,
        MAPE_y: pd.DataFrame,
        MAPE_pi: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Evaluate the solver performance and update the Mean Absolute Percentage Error (MAPE) dataframes.

        Parameters
        ----------
        solver : str
            The name of the solver used for the optimization problem.
        MAPE_y : pd.DataFrame
            DataFrame containing MAPE values for y.
        MAPE_pi : pd.DataFrame
            DataFrame containing MAPE values for pi.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Updated MAPE dataframes for y and pi variables.
        """

        columns = ['NN', 'Solver']
        fz = np.load(path + 'FZ.npy')
        fz = np.nan_to_num(fz, nan=1000)
        fy = np.load(path + 'FY.npy')
        fx = np.load(path + 'FX.npy')
        fw = np.load(path + 'FW.npy')
        mape_w.drop('Solver', axis=1, inplace=True)
        mape_y.drop('Solver', axis=1, inplace=True)
        solver_names = self.solver

        for k in range(len(solver_names)):
            mse_s1 = []
            mse_s2 = []
            for j in range(fx.shape[0]):
                ys = []
                for i in range(fz[0, j, 400:].shape[0]):
                    ys.append(fx[j, 400 + i]@fz[k, j, 400 + i])
                    
                
                mse_s2.append(mape(fy[j, 400:], ys))
                mse_s1.append(mape(fw[j, 400:], fz[k, j, 400:]))
            
            mape_w[solver_names[k]] = mse_s1
            mape_y[solver_names[k]] = mse_s2
        
        return mape_w, mape_y

    def plot_results(self, path: str = '') -> None:
        """
        Plot the Mean Absolute Percentage Error (MAPE) results obtained from the evaluations of the model and solver.

        Parameters
        ----------
        path : str, optional
            The directory path from which the data files are to be loaded. Default is an empty string.

        Returns
        -------
        None
        """

        try:
            mape_y = pd.read_excel(path + 'Modelfly_y.xlsx')
            mape_w = pd.read_excel(path + 'Modelfly_w.xlsx')
        except FileNotFoundError:
            mape_y, mape_w = self.model_eval(path)
            
        mape_y, mape_w = self.solver_eval(mape_y, mape_w, path)
        fig = plt.figure(figsize=(19, 9))
        gs = GridSpec(nrows=2, ncols=2)
        ax0 = fig.add_subplot(gs[:, 0])
        ax0.set_yscale('log')
        ax0.set_ylabel('MAPE', fontsize=20)
        ax0.set_xlabel('Noise levels', fontsize=20)
        mape_y.boxplot(ax=ax0, boxprops=dict(color='blue'), medianprops=dict(color='orangered'), whiskerprops=dict(color='blue'))
        ax0.tick_params(axis='x', labelsize=20, rotation=45)
        ax0.tick_params(axis='y', labelsize=20)
        ax0.set_ylim([1e-14, 1e+16])

        ax1 = fig.add_subplot(gs[:, 1])
        ax1.set_yscale('log')
        ax1.set_ylabel('MAPE', fontsize=20)
        ax1.set_xlabel('Noise levels', fontsize=20)
        mape_w.boxplot(ax=ax1, boxprops=dict(color='blue'), medianprops=dict(color='orangered'), whiskerprops=dict(color='blue'))
        ax1.tick_params(axis='x', labelsize=20, rotation=45)
        ax1.tick_params(axis='y', labelsize=20)
        ax1.set_ylim([1e-22, 1e+18])

        plt.show()

    def history_plot_loss(self, path: str = '') -> pd.DataFrame:
        """
        Plot the history of loss and validation loss for different loss functions.

        Parameters
        ----------
        path : str, optional
            The directory path where the data files are to be loaded from. Defaults to an empty string.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the history of the model's loss and validation loss.
        """

        fy = np.load(path + 'FY.npy', allow_pickle=True)
        fx = np.load(path + 'FX.npy', allow_pickle=True)
        history_loss = []
        fig, axs = plt.subplots(1, 2, figsize=(20, 20))

        for loss_name in ['mse', 'mae', 'huber']:
            model_max = self.model_op(loss = loss_name)
            history = model_max.fit(
                fx[0, 0:400], fy[0][0:400], epochs=400,
                batch_size=32, verbose=False, validation_split=0.3
            )
            history_loss = pd.DataFrame(history.history)

            axs[0].plot(history_loss['loss'], label=loss_name)
            axs[1].plot(history_loss['val_loss'], label='val_' + loss_name)

        axs[0].legend()
        axs[1].legend()
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Error')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Error')

        plt.show()

        return history_loss
