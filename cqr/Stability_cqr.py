import random
import csv
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

# baseline parameter
alpha = 0.1
p = 0.95
bias = 1.0
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

dataset_names = ['meps_19', 'meps_20', 'meps_21', 'bike', 'blog_data', 'bio', 'facebook_1', 'facebook_2', 'concrete', 'star']
bias_list = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2]

csv_file = 'std_results.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Dataset', 'stability_VCP', 'stability_VCP_std', 'stability_PCP', 'stability_PCP_std'])

    for i in range(len(dataset_names)):
        # desired quanitile levels
        quantiles = [0.05, 0.95]

        # used to determine the size of test set
        test_ratio = 0.2

        # name of dataset
        dataset_base_path = "./datasets/"
        dataset_name = dataset_names[i]
        bias = bias_list[i]

        # load the dataset
        X, y = datasets.GetDataset(dataset_name, dataset_base_path)

        std_list_VCP = []
        std_list_PCP = []

        for k in range(5):
            seed = k

            random_state_train_test = seed
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            
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
                                                                random_state=seed,
                                                                use_rearrangement=False)

            # define a CQR object, computes the absolute residual error of points
            # located outside the estimated quantile neural network band
            nc = RegressorNc(quantile_estimator, QuantileRegErrFunc())
            icp = IcpRegressor(nc, condition=None)

            #load the model
            model = all_q_model(in_shape=in_shape, hidden_size=hidden_size, dropout=dropout, quantiles=quantiles)
            model.to(device)
            model.load_state_dict(torch.load(f'cqr_model/cqr_{dataset_name}_{seed}.pt', map_location=device))

            # calibration
            icp.calibrate(x_train[idx_cal, :], y_train[idx_cal], model=model, bias=bias)
            # prediction_VCP
            predictions_VCP = icp.predict(x_test, significance=alpha, model=model, bias=bias)
            y_lower_VCP = predictions_VCP[:, 0]
            y_upper_VCP = predictions_VCP[:, 1]
            length_VCP = y_upper_VCP - y_lower_VCP
            std_VCP = 0.0
            std_list_VCP.append(std_VCP)

            # prediction_PCP
            alpha_prime = 1 - (1 - alpha) / p
            predictions_PCP = icp.predict(x_test, significance=alpha_prime, model=model, bias=bias)
            y_lower_PCP = predictions_PCP[:, 0]
            y_upper_PCP = predictions_PCP[:, 1]
            std_list = []
            for l in range(len(y_lower_PCP)):
                std_x_list = []
                for j in range(50):
                    flag = random.random()
                    if flag > p:
                        length_x_PCP = 0.0
                    else:
                        length_x_PCP = y_upper_PCP[l] - y_lower_PCP[l]
                    std_x_list.append(length_x_PCP)
                std_x = np.std(std_x_list) / np.sqrt(50)
                std_list.append(std_x)
            std_PCP = np.mean(std_list)
            std_list_PCP.append(std_PCP)

        stability_VCP = np.mean(std_list_VCP)
        stability_PCP = np.mean(std_list_PCP)
        stability_VCP_std = np.std(std_list_VCP) / np.sqrt(5)
        stability_PCP_std = np.std(std_list_PCP) / np.sqrt(5)
        writer.writerow([dataset_name, stability_VCP, stability_VCP_std, stability_PCP, stability_PCP_std])
