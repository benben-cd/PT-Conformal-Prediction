import numpy as np
import random
import csv
from sklearn.linear_model import LinearRegression

# define function to compute length and coverage
def compute_len_cvg(model, x_train, y_train, x_test, y_test, idx_cal, alpha=0.1, p=1.0, method='VCP'):
    num_cali = x_train[idx_cal].shape[0]
    level = np.ceil((num_cali + 1) * (1 - alpha)) / num_cali
    alpha_prime = 1 - (1 - alpha) / p
    level_prime = np.ceil((num_cali + 1) * (1 - alpha_prime)) / num_cali

    y_cal_hat = model.predict(x_train[idx_cal])
    res_cal = np.abs(y_train[idx_cal] - y_cal_hat).flatten()
    y_test_hat = model.predict(x_test)
    res_test = np.abs(y_test - y_test_hat).flatten()

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

    return length, cvg

# set standard parameter
n = 2000
test_ratio = 0.2
alpha = 0.1
p = 0.96
bias = 20

alpha_list = [0.05, 0.1, 0.15, 0.2]
p_list = [0.96, 0.97, 0.98, 0.99, 1]
bias_list = [0, 2, 4, 6, 8, 10, 12]

len_list_VCP = []
cvg_list_VCP = []
len_list_PCP = []
cvg_list_PCP = []

alpha_p_len_VCP = np.zeros((5, len(alpha_list), len(p_list)))
alpha_p_len_PCP = np.zeros((5, len(alpha_list), len(p_list)))
alpha_p_cvg_VCP = np.zeros((5, len(alpha_list), len(p_list)))
alpha_p_cvg_PCP = np.zeros((5, len(alpha_list), len(p_list)))
# alpha_bias_len_VCP = np.zeros((5, len(alpha_list), len(bias_list)))
# alpha_bias_len_PCP = np.zeros((5, len(alpha_list), len(bias_list)))

for i in range(5):
    # set random seed
    seed = i
    random.seed(seed)
    np.random.seed(seed)

    # generate data
    x1 = np.random.normal(0, 1, (n, 1))
    x2 = np.random.normal(0, 1, (n, 1))
    eps_pos = np.random.normal(bias, 1, (int(n / 2), 1))
    eps_neg = np.random.normal(-bias, 1, (int(n / 2), 1))
    eps = np.vstack((eps_pos, eps_neg))

    np.random.shuffle(eps)
    np.random.shuffle(x1)
    np.random.shuffle(x2)

    beta1 = 1
    beta2 = 1
    y = beta1 * x1 + beta2 * x2 + eps

    # get training set and calibration set
    x1_train = x1[:int(n * (1 - test_ratio))]
    x2_train = x2[:int(n * (1 - test_ratio))]
    y_train = y[:int(n * (1 - test_ratio))]
    n_train = x1_train.shape[0]
    idx = np.random.permutation(n_train)
    n_half = int(np.floor(n_train / 2))
    idx_train, idx_cal = idx[:n_half], idx[n_half:2 * n_half]

    # get test set
    x1_test = x1[int(n * (1 - test_ratio)):]
    x2_test = x2[int(n * (1 - test_ratio)):]
    y_test = y[int(n * (1 - test_ratio)):]

    x_train = np.column_stack((x1_train, x2_train))
    x_test = np.column_stack((x1_test, x2_test))

    # fit the model
    model_linear = LinearRegression(fit_intercept=True)
    model_linear.fit(x_train[idx_train], y_train[idx_train])

    # compute length and coverage for each method
    for j in range(len(alpha_list)):
        for k in range(len(p_list)):
            len_VCP, cvg_VCP = compute_len_cvg(model_linear, x_train, y_train, x_test, y_test, idx_cal, alpha_list[j], p_list[k], 'VCP')
            len_PCP, cvg_PCP = compute_len_cvg(model_linear, x_train, y_train, x_test, y_test, idx_cal, alpha_list[j], p_list[k], 'PCP')
            alpha_p_len_VCP[i, j, k] = len_VCP
            alpha_p_len_PCP[i, j, k] = len_PCP
            alpha_p_cvg_VCP[i, j, k] = cvg_VCP
            alpha_p_cvg_PCP[i, j, k] = cvg_PCP

with open(f'ablation_study_simulation_p.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(p_list)
    
    for row in np.mean(alpha_p_cvg_VCP, axis=0):
        writer.writerow(row)
    for row in np.std(alpha_p_cvg_VCP, axis=0):
        writer.writerow(row)
    for row in np.mean(alpha_p_len_VCP, axis=0):
        writer.writerow(row)
    for row in np.std(alpha_p_len_VCP, axis=0):
        writer.writerow(row)
    for row in np.mean(alpha_p_cvg_PCP, axis=0):
        writer.writerow(row)
    for row in np.std(alpha_p_cvg_PCP, axis=0):
        writer.writerow(row)
    for row in np.mean(alpha_p_len_PCP, axis=0):
        writer.writerow(row)
    for row in np.std(alpha_p_len_PCP, axis=0): 
        writer.writerow(row)