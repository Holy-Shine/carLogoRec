import numpy as np
import matplotlib.pyplot as plt
from datasets_utils import load_test_data, load_train_data

np.random.seed(1)
def nn_model(input_zise,hid_size = 80, output_size=5):
    W1 = np.random.randn(hid_size,input_zise)*0.01
    W2 = np.random.randn(output_size, hid_size)*0.01
    b1 = np.zeros((hid_size, 1))
    b2 = np.zeros((output_size,1))
    model_params = {
        'W1':W1,
        'W2':W2,
        'b1':b1,
        'b2':b2
    }
    return model_params

def label2onehot(label_org,n_class):
    """
    [0,1,2...]->[[0,0,0],[0,1,0],[0,0,1]...]^T

    :param label_org:
    :return:
    """
    Y = np.zeros(shape=(n_class, label_org.shape[1]))
    for i in range(label_org.shape[1]):
        Y[int(label_org[0][i])][i]=1.0
    return Y

def sigmoid(z):
    return 1/(1+np.exp(-z))

def forward_propagate(params, X, Y):
    '''
    前向传播

    :param params:
    :param X:
    :param Y:
    :return:
    '''
    W1 = params['W1']
    W2 = params['W2']
    b1 = params['b1']
    b2 = params['b2']
    Z1 = np.dot(W1, X)+b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1)+b2
    A2 = sigmoid(Z2)

    cost = 1/2*np.sum(np.square(A2-Y))
    cache={
        'Z1':Z1,
        'Z2':Z2,
        'A1':A1,
        'A2':A2
    }

def optimize(X, Y, model_params, n_epochs=100, learning_rate=0.1):

    costs = []

    for i in range(n_epochs):
        cost, cache = forward_propagate(model_params, X, Y)

if __name__=='__main__':
    n_class = 5
    train_X_org, train_Y_org = load_train_data()
    test_X_org, test_Y_org = load_test_data()

    # normalization train&test X
    train_X, test_X = train_X_org/255, test_X_org/255

    # prune label to one-hot
    train_Y = label2onehot(train_Y_org,n_class)
    test_Y = label2onehot(test_Y_org,n_class)

    # build model
    model_params = nn_model(1)
    costs, params = optimize(train_X, train_Y, model_params, n_epochs = 100, learning_rate = 0.01 )