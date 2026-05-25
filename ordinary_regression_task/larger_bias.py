import random
import csv
import os
from sklearn.preprocessing import StandardScaler
from datasets import datasets
from ordinary_Regmodel import *

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


def compute_len_cvg(model, x_train, y_train, x_test, y_test, idx_cal, alpha=0.1, p=1, method='VCP', bias=0, perturbation_rate=0.001):
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
perturbation_rate = 0.001

dataset_names = [
    'meps_19', 'meps_20', 'meps_21', 'bike', 'blog_data',
    'bio', 'facebook_1', 'facebook_2', 'concrete', 'star'
]

# Use larger bias than the standard setting to magnify the length gap.
bias_map = {
    'meps_19': 80,
    'meps_20': 80,
    'meps_21': 80,
    'blog_data': 80,
    'bio': 40,
    'facebook_1': 40,
    'facebook_2': 40,
    'bike': 40,
    'concrete': 20,
    'star': 20,
}

result_dir = 'larger_bias'
os.makedirs(result_dir, exist_ok=True)

summary_rows = []

for dataset_name in dataset_names:
    bias = bias_map[dataset_name]

    cvg_vcp_seed = np.zeros(5)
    len_vcp_seed = np.zeros(5)
    cvg_pcp_seed = np.zeros(5)
    len_pcp_seed = np.zeros(5)

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

        len_vcp, cvg_vcp = compute_len_cvg(
            model, x_train, y_train, x_test, y_test, idx_cal,
            alpha, p, 'VCP', bias
        )
        len_pcp, cvg_pcp = compute_len_cvg(
            model, x_train, y_train, x_test, y_test, idx_cal,
            alpha, p, 'PCP', bias
        )

        cvg_vcp_seed[i] = cvg_vcp
        len_vcp_seed[i] = len_vcp
        cvg_pcp_seed[i] = cvg_pcp
        len_pcp_seed[i] = len_pcp

    summary_rows.append([
        dataset_name,
        bias,
        np.mean(cvg_vcp_seed), np.std(cvg_vcp_seed),
        np.mean(len_vcp_seed), np.std(len_vcp_seed),
        np.mean(cvg_pcp_seed), np.std(cvg_pcp_seed),
        np.mean(len_pcp_seed), np.std(len_pcp_seed),
    ])

summary_csv = f'{result_dir}/table2_larger_bias_alpha0.1_p0.96.csv'
with open(summary_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        'dataset', 'bias',
        'vcp_coverage_mean', 'vcp_coverage_std',
        'vcp_length_mean', 'vcp_length_std',
        'pcp_coverage_mean', 'pcp_coverage_std',
        'pcp_length_mean', 'pcp_length_std'
    ])
    for row in summary_rows:
        writer.writerow(row)

name_map = {
    'meps_19': 'MEPS-19',
    'meps_20': 'MEPS-20',
    'meps_21': 'MEPS-21',
    'bike': 'BIKE',
    'blog_data': 'BLOG-DATA',
    'bio': 'BIO',
    'facebook_1': 'FACEBOOK-1',
    'facebook_2': 'FACEBOOK-2',
    'concrete': 'CONCRETE',
    'star': 'STAR',
}

ordered_dataset = [
    'meps_19', 'meps_20', 'meps_21', 'bike', 'blog_data',
    'bio', 'facebook_1', 'facebook_2', 'concrete', 'star'
]

row_map = {row[0]: row for row in summary_rows}
md_lines = []
md_lines.append('# Table 2 Style Results (larger bias, p = 0.96)')
md_lines.append('')
md_lines.append('**Caption.** Comparison of performance between VCP and PCP in regression tasks across different datasets at fixed `alpha = 0.1` and `p = 0.96`, using a larger bias setting to amplify the interval-length contrast. Values are reported as `mean +/- std` over 5 random seeds. Bold marks the smaller value in the two length columns for each dataset.')
md_lines.append('')
md_lines.append('| Dataset | Bias | VCP Coverage | VCP Length | PCP Coverage | PCP Length |')
md_lines.append('|---|---:|---:|---:|---:|---:|')

for ds in ordered_dataset:
    row = row_map[ds]
    _, bias_value, vcp_cvg_mean, vcp_cvg_std, vcp_len_mean, vcp_len_std, pcp_cvg_mean, pcp_cvg_std, pcp_len_mean, pcp_len_std = row

    vcp_len_text = f'{vcp_len_mean:.2f} +/- {vcp_len_std:.2f}'
    pcp_len_text = f'{pcp_len_mean:.2f} +/- {pcp_len_std:.2f}'
    if vcp_len_mean <= pcp_len_mean:
        vcp_len_text = f'**{vcp_len_text}**'
    else:
        pcp_len_text = f'**{pcp_len_text}**'

    md_lines.append(
        f'| {name_map[ds]} | {int(bias_value)} | '
        f'{vcp_cvg_mean:.3f} +/- {vcp_cvg_std:.3f} | {vcp_len_text} | '
        f'{pcp_cvg_mean:.3f} +/- {pcp_cvg_std:.3f} | {pcp_len_text} |'
    )

summary_md = f'{result_dir}/table2_larger_bias_alpha0.1_p0.96.md'
with open(summary_md, mode='w', newline='', encoding='utf-8') as file:
    file.write('\n'.join(md_lines) + '\n')

print(f'Results saved to {summary_csv}')
print(f'Results saved to {summary_md}')