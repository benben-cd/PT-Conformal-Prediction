import random
import csv
import os
from sklearn.preprocessing import StandardScaler
from importlib import reload

import datasets.datasets
reload(datasets.datasets)
from datasets import datasets

import ordinary_Regmodel
reload(ordinary_Regmodel)
from ordinary_Regmodel import *

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

def compute_std(model, x_train, y_train, x_test, y_test, idx_cal, alpha=0.1, p=1, method='VCP', bias=0):
    num_cali = x_train[idx_cal].shape[0]
    level = np.ceil((num_cali + 1) * (1 - alpha)) / num_cali
    alpha_prime = 1 - (1 - alpha) / p
    level_prime = np.ceil((num_cali + 1) * (1 - alpha_prime)) / num_cali

    model.eval()
    y_cal_hat = model(torch.from_numpy(x_train[idx_cal]).float().to(device).requires_grad_(False)).cpu().detach().numpy()
    res_cal = np.abs(y_train[idx_cal] - y_cal_hat - bias).flatten()

    if method == 'VCP':
        std = 0
    elif method == 'PCP':
        std_list = []
        for i in range(y_test.shape[0]):
            std_x_list = []
            for j in range(50):
                flag = random.random()
                if flag < p:
                    qhat = np.percentile(np.sort(res_cal), level_prime * 100)
                else:
                    qhat = 0
                std_x_list.append(2*qhat)
            std_x = np.std(std_x_list) / np.sqrt(50)
            std_list.append(std_x)
        std = np.mean(std_list)

    return std

os.makedirs('interval_stability_results', exist_ok=True)
csv_file = 'interval_stability_results/table3_interval_stability_vcp_ptvcp.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Dataset', 'stability_VCP', 'stability_VCP_std', 'stability_PT_VCP', 'stability_PT_VCP_std'])

    alpha = 0.1
    p = 0.95
    bias_list = [20, 20, 20, 10, 20, 10, 10, 10, 5, 5]
    dataset_names = ['meps_19', 'meps_20', 'meps_21', 'bike', 'blog_data', 'bio', 'facebook_1', 'facebook_2', 'concrete', 'star']
    for i in range(len(dataset_names)):
        # set parameters
        lr = 5e-4
        batch_size = 64
        wd = 1e-6
        max_epochs = 1000
        dropout = 0.1
        hidden_size = 64
        optimizer_params = {
            'lr': lr,
            'weight_decay': wd
        }

        # get data
        ## name of dataset
        dataset_base_path = "./datasets/"
        dataset_name = dataset_names[i]
        bias = bias_list[i]

        ## load the data
        X, y = datasets.GetDataset(dataset_name, dataset_base_path)

        std_list_VCP = []
        std_list_PCP = []

        ## set seed
        for j in range(5):
            # set seed
            seed = j
            random_state_train_test = seed
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            ## divide the dataset into test and train based on the test_ratio parameter
            test_ratio = 0.2
            x_train, x_test, y_train, y_test = train_test_split(X,
                                                                y,
                                                                test_size=test_ratio,
                                                                random_state=random_state_train_test)
            ## reshape the data
            x_train = np.asarray(x_train)
            y_train = np.asarray(y_train)
            x_test = np.asarray(x_test)
            y_test = np.asarray(y_test)

            ## compute input dimensions
            n_train = x_train.shape[0]
            in_shape = x_train.shape[1]

            ## display basic information
            print("Dataset: %s" % (dataset_name))
            print("Dimensions: train set (n=%d, p=%d) ; test set (n=%d, p=%d)" %
                    (x_train.shape[0], x_train.shape[1], x_test.shape[0], x_test.shape[1]))

            ## divide the data into proper training set and calibration set
            idx = np.random.permutation(n_train)
            n_half = int(np.floor(n_train / 2))
            idx_train, idx_cal = idx[:n_half], idx[n_half:2 * n_half]

            ## zero mean and unit variance scaling
            scalerX = StandardScaler()
            scalerX = scalerX.fit(x_train[idx_train])

            ## scale
            x_train = scalerX.transform(x_train)
            x_test = scalerX.transform(x_test)

            ## scale the labels by dividing each by the mean absolute response
            mean_y_train = np.mean(np.abs(y_train[idx_train]))
            y_train = np.squeeze(y_train) / mean_y_train
            y_train = y_train.reshape(-1, 1)
            y_test = np.squeeze(y_test) / mean_y_train
            y_test = y_test.reshape(-1, 1)

            ## load the model
            model = mse_model(in_shape=in_shape, hidden_size=hidden_size, dropout=dropout)
            model.to(device)
            model.load_state_dict(torch.load(f'model/{dataset_name}_{seed}.pt', map_location=device))

            std_VCP = compute_std(model, x_train, y_train, x_test, y_test, idx_cal, alpha, p, 'VCP', bias)
            std_PCP = compute_std(model, x_train, y_train, x_test, y_test, idx_cal, alpha, p, 'PCP', bias)
            
            std_list_VCP.append(std_VCP)
            std_list_PCP.append(std_PCP)


        stability_VCP = np.mean(std_list_VCP)
        stability_PT_VCP = np.mean(std_list_PCP)
        stability_VCP_std = np.std(std_list_VCP) / np.sqrt(5)
        stability_PT_VCP_std = np.std(std_list_PCP) / np.sqrt(5)
        writer.writerow([dataset_name, stability_VCP, stability_VCP_std, stability_PT_VCP, stability_PT_VCP_std])
