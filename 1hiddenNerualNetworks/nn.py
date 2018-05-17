import numpy as np
import pickle
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
    前向传播,使用MSE

    '''
    W1 = params['W1']
    W2 = params['W2']
    b1 = params['b1']
    b2 = params['b2']
    Z1 = np.dot(W1, X)+b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1)+b2
    A2 = sigmoid(Z2)

    cost = 1/2*np.sum(np.square(A2-Y))
    cache={
        'Z1':Z1,
        'Z2':Z2,
        'A1':A1,
        'A2':A2
    }
    return cost, cache

def backward_propagate(cache, params, X, Y):
    '''
    反向传播

    '''
    m = X.shape[1]
    W1 = params['W1']
    W2 = params['W2']
    Z1,A1,Z2,A2 = cache['Z1'],cache['A1'],cache['Z2'],cache['A2']
    dA2 = A2 - Y
    dZ2 = dA2*A2*(1-A2)
    dW2 = 1/m*np.dot(dZ2,A1.T)
    db2 = 1/m * np.sum(dZ2,axis=1,keepdims=True)
    dZ1 = np.dot(W2.T,dZ2)*(1-np.square(A1))
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }
    return grads


def optimize(X, Y, model_params, n_epochs=100, learning_rate=0.1):

    costs = []

    for i in range(n_epochs):
        cost, cache = forward_propagate(model_params, X, Y)
        grads = backward_propagate(cache,model_params,X, Y)
        model_params['W1'] -= learning_rate * grads['dW1']
        model_params['W2'] -= learning_rate * grads['dW2']
        model_params['b1'] -= learning_rate * grads['db1']
        model_params['b2'] -= learning_rate * grads['db2']
        print("epoch %d: cost=%f"%(i,cost))
        costs.append(cost)
    return costs,model_params

def evaluate(params, test_X, test_Y):
    W1 = params['W1']
    W2 = params['W2']
    b1 = params['b1']
    b2 = params['b2']
    prediction = sigmoid(np.dot(W2,np.tanh(np.dot(W1,test_X)+b1))+b2)
    pre_label = prediction.argmax(axis=0).reshape(test_Y.shape[0],test_Y.shape[1])

    wrong = np.count_nonzero(pre_label-test_Y)
    print('accuracy: %.2f%%'%((500-wrong)/5))

if __name__=='__main__':
    n_class = 5
    train_X_org, train_Y_org = load_train_data()
    test_X_org, test_Y_org = load_test_data()

    # normalization train&test X
    train_X, test_X = train_X_org/255, test_X_org/255

    # prune label to one-hot
    train_Y = label2onehot(train_Y_org,n_class)
    #test_Y = label2onehot(test_Y_org,n_class)

    # build model
    model_params = nn_model(input_zise=train_X.shape[0])
    costs, params = optimize(train_X, train_Y, model_params, n_epochs = 200, learning_rate = 0.1 )
    pickle.dump(params, open('nn_params.data','wb'))
    params = pickle.load(open('nn_params.data','rb'))
    evaluate(params, test_X, test_Y_org)