import torch
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.metrics import *

class Network(nn.Module):
    def __init__(self,input_size,):
        super(Network, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(input_size,input_size),
            nn.ReLU(),
            nn.Linear(input_size,input_size),    
            nn.ReLU(),
            nn.Linear(input_size,1)
        )
    def forward(self, features):
        return self.base.forward(features)



class Regressor():

    def __init__(self, x, nb_epoch = 10, batch_size = 10):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
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

        # Replace this code with your own
        self.lb = preprocessing.LabelBinarizer()
        self.x_fit = preprocessing.StandardScaler()
        self.y_fit = preprocessing.StandardScaler()
        
        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.learning_rate = 0.001
        self.net = Network(self.input_size)  
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.output_size = 1
        self.nb_epoch = nb_epoch 
        self.batch_size = batch_size
        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
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
        values = {"longitude": 0, "latitude": 0, "housing_median_age": 0, "total_rooms": 0, "total_bedrooms": 0, "population": 0, "households": 0, "median_income": 0, "ocean_proximity": "INLAND"}
        x.fillna(value=values, inplace=True)
        x_col = x.iloc[:,:-1]
        
        if training:
            if isinstance(y, pd.DataFrame):
                self.y_fit.fit(y)
            self.x_fit.fit(x_col)
            proximity = self.lb.fit(x['ocean_proximity'])
        
        proximity = self.lb.transform(x['ocean_proximity'])
        x_col = self.x_fit.transform(x_col)
        frame_with_labels = np.concatenate((x_col,proximity),axis=1)

        x_tensor = torch.tensor(frame_with_labels.astype(np.float32))    
        if isinstance(y, pd.DataFrame):
            y_col = self.y_fit.transform(y)
            y_tensor = torch.tensor(y_col.astype(np.float32))
            
        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None
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

        X, target = self._preprocessor(x, y = y, training = True) # Do not forget

        batch_number = int(len(X) / self.batch_size)    

        for _ in range(self.nb_epoch):
            for batch in range(batch_number):
                batch_X, batch_target = X[batch * self.batch_size : (batch + 1) * self.batch_size, ], target[batch * self.batch_size : (batch + 1) * self.batch_size, ]
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

        X, _ = self._preprocessor(x, training = False) # Do not forget
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

        # X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        # self.net.eval()
        # output = self.net(X).detach().numpy()
        # #reg = LinearRegression().fit(self.y_fit.inverse_transform(output), self.y_fit.inverse_transform(Y))
        # print(self.y_fit.inverse_transform(Y))
        # print(self.y_fit.inverse_transform(output))
        # print(mean_squared_error(Y, output))
        # return mean_squared_error(self.y_fit.inverse_transform(Y), self.y_fit.inverse_transform(output), squared=False)
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



def RegressorHyperParameterSearch(): 
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

    return  # Return the chosen hyper parameters

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

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 10)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    #Test
    # test_data = pd.read_csv("test.csv")
    # x_test = test_data.loc[:, data.columns != output_label]
    # out = regressor.predict(x_test)
    # y_test = test_data.loc[:, [output_label]]
    # print(out)
    # print(y_test)
    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()

