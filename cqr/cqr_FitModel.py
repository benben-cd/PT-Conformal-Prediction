import random
import warnings
import numpy as np
warnings.filterwarnings('ignore')
from cqrfile.torch_models import *
from cqrfile import helper
from nonconformist.nc import RegressorNc
from nonconformist.nc import QuantileRegErrFunc
from nonconformist.icp import *
from datasets import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

dataset_names = ['meps_19', 'meps_20', 'meps_21', 'bike', 'blog_data', 'bio', 'facebook_1', 'facebook_2', 'concrete', 'star']
for name in dataset_names:
    for i in range(5):
        seed = i

        random_state_train_test = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # desired miscoverage error
        alpha = 0.1

        # desired quanitile levels
        quantiles = [0.05, 0.95]

        # used to determine the size of test set
        test_ratio = 0.2

        # name of dataset
        dataset_base_path = "./datasets/"
        dataset_name = name

        # load the dataset
        X, y = datasets.GetDataset(dataset_name, dataset_base_path)

        # divide the dataset into test and train based on the test_ratio parameter
        x_train, x_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=test_ratio,
                                                            random_state=random_state_train_test)

        # reshape the data
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)

        # compute input dimensions
        n_train = x_train.shape[0]
        in_shape = x_train.shape[1]

        # display basic information
        print("Dataset: %s" % (dataset_name))
        print("Dimensions: train set (n=%d, p=%d) ; test set (n=%d, p=%d)" %
              (x_train.shape[0], x_train.shape[1], x_test.shape[0], x_test.shape[1]))

        # divide the data into proper training set and calibration set
        idx = np.random.permutation(n_train)
        n_half = int(np.floor(n_train / 2))
        idx_train, idx_cal = idx[:n_half], idx[n_half:2 * n_half]

        # zero mean and unit variance scaling
        scalerX = StandardScaler()
        scalerX = scalerX.fit(x_train[idx_train])

        # scale
        x_train = scalerX.transform(x_train)
        x_test = scalerX.transform(x_test)

        # scale the labels by dividing each by the mean absolute response
        mean_y_train = np.mean(np.abs(y_train[idx_train]))
        y_train = np.squeeze(y_train) / mean_y_train
        y_test = np.squeeze(y_test) / mean_y_train

        #####################################################
        # Neural network parameters
        # (See AllQNet_RegressorAdapter class in helper.py)
        #####################################################

        # pytorch's optimizer object
        nn_learn_func = torch.optim.Adam

        # number of epochs
        epochs = 1000

        # learning rate
        lr = 0.0005

        # mini-batch size
        batch_size = 64

        # hidden dimension of the network
        hidden_size = 64

        # dropout regularization rate
        dropout = 0.1

        # weight decay regularization
        wd = 1e-6
        # ratio of held-out data, used in cross-validation
        cv_test_ratio = 0.05

        # seed for splitting the data in cross-validation.
        # Also used as the seed in quantile random forests function
        cv_random_state = 1

        quantile_estimator = helper.AllQNet_RegressorAdapter(model=None,
                                                             fit_params=None,
                                                             in_shape=in_shape,
                                                             hidden_size=hidden_size,
                                                             quantiles=quantiles,
                                                             learn_func=nn_learn_func,
                                                             epochs=epochs,
                                                             batch_size=batch_size,
                                                             dropout=dropout,
                                                             lr=lr,
                                                             wd=wd,
                                                             test_ratio=cv_test_ratio,
                                                             random_state=cv_random_state,
                                                             use_rearrangement=False)

        # define a CQR object, computes the absolute residual error of points
        # located outside the estimated quantile neural network band
        nc = RegressorNc(quantile_estimator, QuantileRegErrFunc())
        icp = IcpRegressor(nc, condition=None)
        icp.fit(x_train[idx_train, :], y_train[idx_train], dataset_name=dataset_name, seed=seed)
