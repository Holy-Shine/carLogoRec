import numpy as np
import matplotlib.pyplot as plt
from datasets_utils import load_test_data, load_train_data


if __name__=='__main__':
    train_X_org, train_Y_org = load_train_data()
    test_X_org, test_Y_org = load_test_data()
    