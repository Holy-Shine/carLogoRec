import numpy as np
import matplotlib.pyplot as plt
import pylab
import scipy
from PIL import Image
from datasets_utils import load_train_data,load_test_data, load_test_data_by_pos

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
    print(Y_prediction)
    return Y_prediction

def estimate(test_X, test_Y, params):
    m = test_X.shape[1]
    Y_prediction = np.zeros((1, m))
    Y_results = []  # 5 x 1 x m
    for param in params:
        w = param['w']
        b = param['b']
        Y_results.append(predict(w,b,test_X))
    for n in range(m):
        max_p = 0
        max_i = 0
        for index in range(5):
            if Y_results[index][0][n]>max_p:
                max_p=Y_results[index][0][n]
                max_i=index
        Y_prediction[0,n]=max_i
    wrong = list(list(map(lambda x,y:x==y, Y_prediction, test_Y))[0]).count(0)

    print("test acc:{} %".format(100-wrong/m*100))

if __name__=='__main__':
    ## one-vs-rest  n classiffer
    params=[]
    for positive in range(5):
        train_X, train_Y = load_train_data(positive)
        train_X = train_X / 255
        print("model:%d"%positive)
        params.append(lr_model(train_X, train_Y,print_cost=True))
    test_X, test_Y = load_test_data()
    test_X/=255
    estimate(test_X, test_Y, params)