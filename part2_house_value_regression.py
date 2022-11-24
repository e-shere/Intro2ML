import csv
import torch
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import *
from sklearn.model_selection import *


class Network(nn.Module):

    def __init__(self, input_size, hidden_layers, inb):
        super(Network, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, inb))
        layers.append(nn.ReLU())
        for i in range(hidden_layers):
            layers.append(nn.Linear(inb, inb))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(inb, 1))
        self.base = nn.Sequential(*layers)

    def forward(self, features):
        return self.base.forward(features)


class Regressor(BaseEstimator):

    def __init__(self,
                 x,
                 nb_epoch=10,
                 batch_size=10,
                 hidden_layers=3,
                 neurons=13,
                 learning_rate=0.001):
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        self.lb = preprocessing.LabelBinarizer()
        self.x_fit = preprocessing.StandardScaler()
        self.y_fit = preprocessing.StandardScaler()

        X, _ = self._preprocessor(x, training=True)
        self.x = x
        self.input_size = X.shape[1]
        self.net = Network(self.input_size, hidden_layers, neurons)
        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=learning_rate)
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers
        self.neurons = neurons
        self.learning_rate = learning_rate
        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y=None, training=False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        x = x.copy()
        values = {
            "longitude": x["longitude"].mean(),
            "latitude": x["latitude"].mean(),
            "housing_median_age": x["housing_median_age"].mean(),
            "total_rooms": x["total_rooms"].mean(),
            "total_bedrooms": x["total_bedrooms"].mean(),
            "population": x["population"].mean(),
            "households": x["households"].mean(),
            "median_income": x["median_income"].mean(),
            "ocean_proximity": x["ocean_proximity"].mode(),
        }

        x.fillna(value=values, inplace=True)
        x_col = x.iloc[:, :-1]

        if training:
            if isinstance(y, pd.DataFrame):
                self.y_fit.fit(y)
            self.x_fit.fit(x_col)
            proximity = self.lb.fit(x['ocean_proximity'])

        proximity = self.lb.transform(x['ocean_proximity'])
        x_col = self.x_fit.transform(x_col)
        frame_with_labels = np.concatenate((x_col, proximity), axis=1)

        x_tensor = torch.tensor(frame_with_labels.astype(np.float32))
        if isinstance(y, pd.DataFrame):
            y_col = self.y_fit.transform(y)
            y_tensor = torch.tensor(y_col.astype(np.float32))

        return x_tensor, (y_tensor if isinstance(y, pd.DataFrame) else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, target = self._preprocessor(x, y=y, training=True)

        batch_number = int(len(X) / self.batch_size)

        for _ in range(self.nb_epoch):
            for batch in range(batch_number):
                batch_X = X[batch * self.batch_size : (batch + 1) * self.batch_size,]
                batch_target = target[batch * self.batch_size : (batch + 1) * self.batch_size,]
                input = self.net(batch_X)
                mse_loss = nn.MSELoss()
                loss = mse_loss(input, batch_target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training=False)
        self.net.eval()
        output = self.net(X)
        return self.y_fit.inverse_transform(output.detach().numpy())

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Don't need to preprocess data here as it is done in predict()
        y_hat = self.predict(x)
        return mean_squared_error(y, y_hat, squared=False)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model):
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def RegressorHyperParameterSearch(data, output_label):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    param_grid = [{
        "nb_epoch": [1000],
        "batch_size": range(50, 200, 10),
        "hidden_layers": range(5, 20, 2),
        "neurons": range(1, 150, 5),
        "learning_rate": [0.001,0.01,0.1]
        }]

    # TODO: add cross validation
    search = RandomizedSearchCV(estimator=Regressor(x_train),
                          param_distributions=param_grid,
                          n_jobs=-1,
                          verbose=4,
                          cv=3,
                          scoring="neg_root_mean_squared_error")

    result = search.fit(x_train, y_train)
    return result.best_params_

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv")

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1)
    # Training
    # This example trains on the whole available dataset.
    # You probably want to separate some held-out data
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))
        
if __name__ == "__main__":
    example_main()
