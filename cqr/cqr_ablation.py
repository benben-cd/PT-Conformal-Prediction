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
p = 0.96
bias = 1.0

alpha_list = [0.05, 0.1, 0.15, 0.2]
p_list = [0.96, 0.97, 0.98, 0.99, 1]
dataset_names = ['meps_19', 'meps_20', 'meps_21', 'blog_data', 'bio', 'facebook_1', 'facebook_2', 'concrete', 'star']

for dataset_name in dataset_names:
    if dataset_name == 'meps_19' or dataset_name == 'meps_20' or dataset_name == 'meps_21' or dataset_name == 'blog_data' or dataset_name == 'bio' or dataset_name == 'facebook_1' or dataset_name == 'facebook_2':
        bias_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    elif dataset_name == 'concrete' or dataset_name == 'star':
        bias_list = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    alpha_p_array_VCP = np.zeros((5, len(alpha_list), len(p_list)))
    alpha_bias_array_VCP = np.zeros((5, len(alpha_list), len(bias_list)))
    alpha_p_array_PCP = np.zeros((5, len(alpha_list), len(p_list)))
    alpha_bias_array_PCP = np.zeros((5, len(alpha_list), len(bias_list)))

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

        for j in range(len(alpha_list)):
            for k in range(len(p_list)):
                # calibration
                icp.calibrate(x_train[idx_cal, :], y_train[idx_cal], model=model, bias=bias)
                # prediction_VCP
                predictions_VCP = icp.predict(x_test, significance=alpha_list[j], model=model, bias=bias)
                y_lower_VCP = predictions_VCP[:, 0]
                y_upper_VCP = predictions_VCP[:, 1]
                coverage_VCP, length_VCP = helper.compute_coverage(y_test,
                                                                        y_lower_VCP,
                                                                        y_upper_VCP,
                                                                        alpha,
                                                                        "CQR Neural Net")
                alpha_p_array_VCP[i, j, k] = length_VCP

                # prediction_PCP
                alpha_prime = 1 - (1 - alpha_list[j]) / p_list[k]
                predictions_PCP = icp.predict(x_test, significance=alpha_prime, model=model, bias=bias)
                y_lower_PCP = predictions_PCP[:, 0]
                y_upper_PCP = predictions_PCP[:, 1]
                for l in range(len(y_lower_PCP)):
                    flag = random.random()
                    if flag > p_list[k]:
                        y_lower_PCP[l] = 0
                        y_upper_PCP[l] = 0
                coverage_PCP, length_PCP = helper.compute_coverage(y_test,
                                                                        y_lower_PCP,
                                                                        y_upper_PCP,
                                                                        alpha_prime,
                                                                        "CQR Neural Net")
                alpha_p_array_PCP[i, j, k] = length_PCP

            for k in range(len(bias_list)):
                # calibration
                icp.calibrate(x_train[idx_cal, :], y_train[idx_cal], model=model, bias=bias_list[k])
                # prediction_VCP
                predictions_VCP = icp.predict(x_test, significance=alpha_list[j], model=model, bias=bias_list[k])
                y_lower_VCP = predictions_VCP[:, 0]
                y_upper_VCP = predictions_VCP[:, 1]
                coverage_VCP, length_VCP = helper.compute_coverage(y_test,
                                                                y_lower_VCP,
                                                                y_upper_VCP,
                                                                alpha,
                                                                "CQR Neural Net")
                alpha_bias_array_VCP[i, j, k] = length_VCP

                # prediction_PCP
                alpha_prime = 1 - (1 - alpha_list[j]) / p
                predictions_PCP = icp.predict(x_test, significance=alpha_prime, model=model, bias=bias_list[k])
                y_lower_PCP = predictions_PCP[:, 0]
                y_upper_PCP = predictions_PCP[:, 1]
                for l in range(len(y_lower_PCP)):
                    flag = random.random()
                    if flag > p:
                        y_lower_PCP[l] = 0
                        y_upper_PCP[l] = 0
                coverage_PCP, length_PCP = helper.compute_coverage(y_test,
                                                                y_lower_PCP,
                                                                y_upper_PCP,
                                                                alpha_prime,
                                                                "CQR Neural Net")
                alpha_bias_array_PCP[i, j, k] = length_PCP


    with open(f'cqr_ablation_result/ablation_study_{dataset_name}_p.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(p_list)
        for row in np.mean(alpha_p_array_VCP, axis=0):
            writer.writerow(row)

        for row in np.mean(alpha_p_array_PCP, axis=0):
            writer.writerow(row)

    print(f'Results saved to ablation_study_{dataset_name}_p.csv')

    with open(f'cqr_ablation_result/ablation_study_{dataset_name}_bias.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(bias_list)
        for row in np.mean(alpha_bias_array_VCP, axis=0):
            writer.writerow(row)

        for row in np.mean(alpha_bias_array_PCP, axis=0):
            writer.writerow(row)

    print(f'Results saved to ablation_study_{dataset_name}_bias.csv')

