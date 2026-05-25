import random
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler
from datasets.datasets_groupcoverage import *

from importlib import reload
import ordinary_Regmodel
reload(ordinary_Regmodel)
from ordinary_Regmodel import *

# define function to compute the group coverage
def compute_group_cvg(model, x_train, y_train, x_test, y_test, idx_cal, column_names, name, alpha=0.1, p=1.0, bias=0,
                      method='VCP'):
    assert name in column_names, 'we do not have the name in the column_names'
    df_xtest = pd.DataFrame(x_test)
    df_xtest.columns = column_names
    df_xtest_name = df_xtest.groupby(name)
    group_coverage = np.zeros(len(df_xtest_name.size()))

    num_cali = x_train[idx_cal].shape[0]
    level = np.ceil((num_cali + 1) * (1 - alpha)) / num_cali
    alpha_prime = 1 - (1 - alpha) / p
    level_prime = np.ceil((num_cali + 1) * (1 - alpha_prime)) / num_cali

    model.eval()
    y_cal_hat = model(
        torch.from_numpy(x_train[idx_cal]).float().to(device).requires_grad_(False)).cpu().detach().numpy()
    res_cal = np.abs(y_train[idx_cal] - y_cal_hat - bias).flatten()
    df_xtest['y_test_hat'] = model(
        torch.from_numpy(x_test).float().to(device).requires_grad_(False)).cpu().detach().numpy()
    df_xtest['y_test'] = y_test
    df_xtest['res_test'] = np.abs(df_xtest['y_test'] - df_xtest['y_test_hat'] - bias)

    if method == 'VCP':
        qhat = np.percentile(np.sort(res_cal), level * 100)
        df_xtest['in_the_range'] = np.where(df_xtest['res_test'] < qhat, 1, 0)

        group_sum = df_xtest.groupby(name)['in_the_range'].sum()
        group_count = df_xtest.groupby(name)['in_the_range'].count()
        group_coverage = group_sum / group_count

    elif method == 'PCP':
        qhat = np.percentile(np.sort(res_cal), level_prime * 100)
        df_xtest['in_the_range'] = np.where(df_xtest['res_test'] < qhat, 1, 0)
        for i in range(len(df_xtest['in_the_range'])):
            flag = random.random()
            if flag >= p:
                df_xtest['in_the_range'][i] = 0

        group_sum = df_xtest.groupby(name)['in_the_range'].sum()
        group_count = df_xtest.groupby(name)['in_the_range'].count()
        group_coverage = group_sum / group_count

    return group_coverage.to_numpy()

# set the device
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


datasets = ['bike']
alpha_list = [0.05, 0.10, 0.15, 0.20]
p_list = [0.96, 0.97, 0.98, 0.99, 1.0]


for dataset_name in datasets:
    if dataset_name == 'meps_19':
        group_names = ['SEX=1', 'MARRY=1', 'REGION=1']
        bias = 20
    elif dataset_name == 'meps_20':
        group_names = ['FTSTU=1', 'ACTDTY=1', 'HONRDC=1']
        bias = 20
    elif dataset_name == 'meps_21':
        group_names = ['RTHLTH=1', 'MNHLTH=1', 'HIBPDX=1']
        bias = 20
    elif dataset_name == 'star':
        group_names = ['gender', 'stark', 'school1']
        bias = 5
    elif dataset_name == 'bike':
        group_names = ['day', 'month', 'year']
        bias = 10

    for group_name in group_names:
        alpha_p_array_VCP = np.zeros((len(alpha_list), len(p_list)))
        alpha_p_array_PCP = np.zeros((len(alpha_list), len(p_list)))
        for i in range(len(alpha_list)):
            for j in range(len(p_list)):
                group_coverage_list_VCP = []
                group_coverage_list_PCP = []
                for k in range(5):
                    # set seed
                    seed = k
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
                    alpha = alpha_list[i]
                    p = p_list[j]

                    # get data
                    ## name of dataset
                    dataset_base_path = "./datasets/"

                    ## load the data
                    X, y = GetDataset(dataset_name, dataset_base_path)
                    
                    ## get the column names
                    column_names = X.columns.tolist()

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

                    ## get the model
                    model = mse_model(in_shape=in_shape, hidden_size=hidden_size, dropout=dropout)
                    model.to(device)
                    model.load_state_dict(torch.load(f'model/{dataset_name}_{seed}.pt', map_location=device))

                    ## compute group coverage
                    group_coverage_VCP = compute_group_cvg(model, x_train, y_train, x_test, y_test, idx_cal, column_names, group_name, alpha, p, bias,
                                                        method='VCP')
                    group_coverage_PCP = compute_group_cvg(model, x_train, y_train, x_test, y_test, idx_cal, column_names, group_name, alpha, p, bias,
                                                        method='PCP')
                    group_coverage_list_VCP.append(group_coverage_VCP.tolist())
                    group_coverage_list_PCP.append(group_coverage_PCP.tolist())

                    gc_array_VCP = np.mean(np.array(group_coverage_list_VCP), axis=0)
                    gc_VCP = np.min(gc_array_VCP)
                    alpha_p_array_VCP[i, j] = gc_VCP
                    gc_array_PCP = np.mean(np.array(group_coverage_list_PCP), axis=0)
                    gc_PCP = np.min(gc_array_PCP)
                    alpha_p_array_PCP[i, j] = gc_PCP
                    # gc_std_VCP = np.std(np.array(group_coverage_list_VCP), axis=0)
                    # gc_std_PCP = np.std(np.array(group_coverage_list_PCP), axis=0)


        with open(f'group_coverage_result/gc_{dataset_name}_{group_name}.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(p_list)
            for row in alpha_p_array_VCP:
                writer.writerow(row)

            for row in alpha_p_array_PCP:
                writer.writerow(row)

            print(f'group_coverage_result/gc_{dataset_name}_{group_name}.csv')
