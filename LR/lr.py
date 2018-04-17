import numpy as np
import matplotlib.pyplot as plt
import pylab
import scipy
from PIL import Image
from datasets_utils import load_train_data,load_test_data

## one-vs-rest

def initial_parameters(dim):
    '''
    initial Weight and Bias

    :param dim: dimmention of feature
    :return: W,b
    '''
    w = np.zeros((dim,1))
    b = 0
    return w,b

def sigmoid(x):
    s=1/(1+np.exp(-x))
    return s

def propagate(w,b,X,Y):
    A = sigmoid(np.dot(w.T,X)+b)
    m = X.shape[1]
    cost = -1/m*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))

    dw = 1/m * np.dot(X, (A-Y).T)
    db = 1/m * np.sum(A-Y)
    grads = {
        'dw':dw,
        'db':db
    }
    assert (dw.shape==w.shape)
    assert (db.dtype==float)
    cost = np.squeeze(cost)
    assert (cost.shape==())
    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w,b,X,Y)
        dw = grads['dw']
        db = grads['db']
        w = w - learning_rate*dw
        b = b - learning_rate*db

        if i%100==0:
            costs.append(cost)

        if print_cost and i%100==0:
            print("cost after iteration %i: %f"%(i,cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs

def lr_model(train_X, train_Y, num_iterations=1000, learning_rate=0.01,print_cost=False):
    m = train_X.shape[1]
    dim = train_X.shape[0]
    w,b = initial_parameters(dim)
    parameters, grads, costs = optimize(w,b, train_X, train_Y,num_iterations, learning_rate, print_cost)
    return parameters
def show_pic(index):
    '''
    show image sample

    :param index: show sample by its index
    :return:
    '''
    plt.imshow(train_X[:,index].reshape((45,45,3)))
    pylab.show()


def predict(w,b, X):
    m=X.shape[1]
    w = w.reshape(X.shape[0],1)

    A = sigmoid(np.dot(w.T,X)+b)
    Y_prediction = np.round(A)
    return Y_prediction

def estimate(test_X, test_Y, params):
    w = params['w']
    b = params['b']
    Y_prediction = predict(w,b,test_X)
    print("test acc:{} %".format(100-np.mean(np.abs(Y_prediction-test_Y))*100))

if __name__=='__main__':
    #print("\n".join([" "*(n-i)+"*"*(2*i-1) for i in range(1,n+1)]+[" "*i+"*"*((n-i)*2-1) for i in range(1, n)]))
    train_X,train_Y=load_train_data(0)
    train_X = train_X/255
    params=lr_model(train_X, train_Y,print_cost=True)
    test_X, test_Y = load_test_data(0)
    test_X/=255
    estimate(test_X, test_Y, params)