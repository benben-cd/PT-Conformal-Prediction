import random
import csv
from sklearn.preprocessing import StandardScaler
from datasets import datasets
from ordinary_Regmodel import *

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


def compute_len_cvg(model, x_train, y_train, x_test, y_test, idx_cal, alpha=0.1, p=1, method='VCP', bias=0, perturbation_rate=0.01):
    num_cali = x_train[idx_cal].shape[0]
    level = np.ceil((num_cali + 1) * (1 - alpha)) / num_cali
    alpha_prime = 1 - (1 - alpha) / p
    level_prime = np.ceil((num_cali + 1) * (1 - alpha_prime)) / num_cali

    model.eval()
    y_cal_hat = model(torch.from_numpy(x_train[idx_cal]).float().to(device).requires_grad_(False)).cpu().detach().numpy()
    res_cal = np.abs(y_train[idx_cal] - y_cal_hat - bias).flatten()
    y_test_hat = model(torch.from_numpy(x_test).float().to(device).requires_grad_(False)).cpu().detach().numpy()
    res_test = np.abs(y_test - y_test_hat - bias).flatten()

    if method == 'VCP':
        qhat = np.percentile(np.sort(res_cal), level * 100)
        length = 2 * qhat
        cvg = np.sum(res_test < qhat) / y_test.shape[0]
    elif method == 'PCP':
        len_list = []
        cvg_list = []
        for i in range(y_test.shape[0]):
            flag = random.random()
            if flag < p:
                qhat = np.percentile(np.sort(res_cal), level_prime * 100)
            else:
                qhat = 0
            len_list.append(2 * qhat)
            cvg_list.append(res_test[i] < qhat)
        length = np.mean(len_list)
        cvg = np.sum(cvg_list) / y_test.shape[0]
    elif method == 'PT_perturbation':
        len_list = []
        cvg_list = []
        q = np.percentile(np.sort(res_cal), level * 100)
        q_prime = np.percentile(np.sort(res_cal), level_prime * 100)

        for i in range(y_test.shape[0]):
            flag = random.random()
            if flag < p:
                len_list.append(2 * q_prime)
                cvg_list.append(res_test[i] < q_prime)
            else:
                len_list.append(2 * q * perturbation_rate)
                cvg_list.append(res_test[i] < q * perturbation_rate)
        length = np.mean(len_list)
        cvg = np.sum(cvg_list) / y_test.shape[0]

    return length, cvg

# baseline parameter
alpha = 0.1
p = 0.96
bias = 10
perturbation_rate = 0.01
dataset_names = ['meps_19', 'meps_20', 'meps_21', 'bike', 'blog_data', 'bio', 'facebook_1', 'facebook_2', 'concrete', 'star']
alpha_list = [0.05, 0.1, 0.15, 0.2]
p_list = [0.96, 0.97, 0.98, 0.99, 1]
alpha_p_array_VCP = np.zeros((5, len(alpha_list), len(p_list)))
alpha_p_array_PT_perturbation = np.zeros((5, len(alpha_list), len(p_list)))
alpha_p_len_array_VCP = np.zeros((5, len(alpha_list), len(p_list)))
alpha_p_len_array_PT_perturbation = np.zeros((5, len(alpha_list), len(p_list)))

for dataset_name in dataset_names:
    if dataset_name == 'meps_19' or dataset_name == 'meps_20' or dataset_name == 'meps_21' or dataset_name == 'blog_data':
        bias = 20
    elif dataset_name == 'bio' or dataset_name == 'facebook_1' or dataset_name == 'facebook_2' or dataset_name == 'bike':
        bias = 10
    elif dataset_name == 'concrete' or dataset_name == 'star':
        bias = 5
    for i in range(5):
        # set seed
        seed = i
        random_state_train_test = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
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

        ## load the data
        X, y = datasets.GetDataset(dataset_name, dataset_base_path)

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

        #load the model
        model = mse_model(in_shape=in_shape, hidden_size=hidden_size, dropout=dropout)
        model.to(device)
        model.load_state_dict(torch.load(f'model/{dataset_name}_{seed}.pt', map_location=device))

        for j in range(len(alpha_list)):
            for k in range(len(p_list)):
                len_VCP, cvg_VCP = compute_len_cvg(model, x_train, y_train, x_test, y_test, idx_cal, alpha_list[j], p_list[k], 'VCP', bias)
                len_PT_perturbation, cvg_PT_perturbation = compute_len_cvg(model, x_train, y_train, x_test, y_test, idx_cal, alpha_list[j], p_list[k], 'PT_perturbation', bias, perturbation_rate)
                alpha_p_array_VCP[i, j, k] = cvg_VCP
                alpha_p_array_PT_perturbation[i, j, k] = cvg_PT_perturbation
                alpha_p_len_array_VCP[i, j, k] = len_VCP
                alpha_p_len_array_PT_perturbation[i, j, k] = len_PT_perturbation

    with open(f'pt_result/cvg_{dataset_name}_p_PT_perturbation.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(p_list)
        
        for row in np.mean(alpha_p_array_VCP, axis=0):
            writer.writerow(row)

        for row in np.mean(alpha_p_array_PT_perturbation, axis=0):
            writer.writerow(row)

        for row in np.std(alpha_p_array_VCP, axis=0):
            writer.writerow(row)

        for row in np.std(alpha_p_array_PT_perturbation, axis=0):
            writer.writerow(row)

    with open(f'pt_result/len_{dataset_name}_p_PT_perturbation.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(p_list)

        for row in np.mean(alpha_p_len_array_VCP, axis=0):
            writer.writerow(row)

        for row in np.mean(alpha_p_len_array_PT_perturbation, axis=0):
            writer.writerow(row)

        for row in np.std(alpha_p_len_array_VCP, axis=0):
            writer.writerow(row)

        for row in np.std(alpha_p_len_array_PT_perturbation, axis=0):
            writer.writerow(row)

    print(f'Results saved to pt_result/cvg_{dataset_name}_p_PT_perturbation.csv')
    print(f'Results saved to pt_result/len_{dataset_name}_p_PT_perturbation.csv')