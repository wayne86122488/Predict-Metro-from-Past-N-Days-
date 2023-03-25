import numpy as np
from pandas_ods_reader import read_ods
import random 

# Define sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the neural network
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights randomly
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.weights3 = 130000
        self.hidden_size = hidden_size
    
    def forward(self, X):
        # Propagate inputs through the network
        self.hidden = sigmoid(np.dot(X, self.weights1))
        self.output = sigmoid(np.dot(self.hidden, self.weights2))
        return self.output*self.weights3
    
    def backward(self, X, y, learning_rate):
        # Compute derivatives of the loss function
        d_weights1 = 2*(y - self.output*self.weights3)*self.weights3*self.output*(1-self.output) * np.tensordot(X, self.hidden*(1-self.hidden), axes=0 )
        d_weights2 = 2*(y - self.output*self.weights3) * self.weights3 * self.output*(1-self.output)*self.hidden
        d_weights3 = -2*(y-self.output*self.weights3)*self.output
        
        # Update weights
        self.weights1 += learning_rate * d_weights1
        self.weights2 += learning_rate * d_weights2.reshape((self.hidden_size,1))
        self.weights3 += learning_rate * d_weights3
    
    def train(self, X, y, learning_rate):
        output = self.forward(X)
        self.backward(X, y, learning_rate)
    
    def predict(self, X):
        # Propagate inputs through the network and return the predicted value
        return self.forward(X)
    
# Test the neural network with sample date
training_data = []
for year in range(2018,2023,1):
    for month in range(1,10,1):
        path = './data/'+str(year)+'0'+str(month)+'_cht.ods'
        sheet_index = 1
        df = read_ods(path , sheet_index)
        data = df['台北車站'].to_numpy()
        training_data = np.concatenate((training_data,data))
    for month in range(10,13,1):
        path = './data/'+str(year)+str(month)+'_cht.ods'
        sheet_index = 1
        df = read_ods(path , sheet_index)
        data = df['台北車站'].to_numpy()
        training_data = np.concatenate((training_data,data))
        
# Initialize the neural network
nn = NeuralNetwork(5, 5, 1)

# Train the neural network
for i in range(1,20000,1):
    k = random.randint(0,1817)
    X = training_data[k:k+5]
    y = training_data[k+6]
    nn.train(X, y, 0.00002)
print(nn.weights1)
print(nn.weights2)
print(nn.weights3)

# Predict the output for a new input
new_X = np.array([[183655, 141609, 136743, 137928, 174365]])
print(nn.predict(new_X)) # should print a positive number

