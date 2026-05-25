import os
import random
from sklearn.preprocessing import StandardScaler
from datasets import datasets
from ordinary_Regmodel import *

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

dataset_names = ['meps_19', 'meps_20', 'meps_21', 'bike', 'blog_data', 'bio', 'facebook_1', 'facebook_2', 'concrete', 'star']
os.makedirs('model', exist_ok=True)
for name in dataset_names:
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
        dataset_name = name

        ## load the data
        X, y = datasets.GetDataset(dataset_name, dataset_base_path)

        # Three-stage split:
        # 1) D_total -> D_train_all (80%) + D_test (20%)
        # 2) D_train_all -> D_remain (80%) + D_res (20%)
        # 3) D_remain -> D_train (80%) + D_cal (20%)
        test_ratio = 0.2
        x_train_all, x_test, y_train_all, y_test = train_test_split(
            X,
            y,
            test_size=test_ratio,
            random_state=random_state_train_test
        )
        x_remain, x_res, y_remain, y_res = train_test_split(
            x_train_all,
            y_train_all,
            test_size=0.2,
            random_state=random_state_train_test + 1
        )
        x_train, x_cal, y_train, y_cal = train_test_split(
            x_remain,
            y_remain,
            test_size=0.2,
            random_state=random_state_train_test + 2
        )

        ## reshape the data
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        x_cal = np.asarray(x_cal)
        y_cal = np.asarray(y_cal)
        x_res = np.asarray(x_res)
        y_res = np.asarray(y_res)
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)

        ## compute input dimensions
        in_shape = x_train.shape[1]

        ## display basic information
        print("Dataset: %s" % (dataset_name))
        print(
            "Split sizes: D_train=%d, D_cal=%d, D_res=%d, D_test=%d" %
            (x_train.shape[0], x_cal.shape[0], x_res.shape[0], x_test.shape[0])
        )

        ## zero mean and unit variance scaling (fit on D_train only)
        scalerX = StandardScaler()
        scalerX = scalerX.fit(x_train)

        ## scale all sets with the same scaler
        x_train = scalerX.transform(x_train)
        x_cal = scalerX.transform(x_cal)
        x_res = scalerX.transform(x_res)
        x_test = scalerX.transform(x_test)

        ## scale labels by the mean absolute response of D_train
        mean_y_train = np.mean(np.abs(y_train))
        y_train = np.squeeze(y_train) / mean_y_train
        y_train = y_train.reshape(-1, 1)
        y_cal = np.squeeze(y_cal) / mean_y_train
        y_cal = y_cal.reshape(-1, 1)
        y_res = np.squeeze(y_res) / mean_y_train
        y_res = y_res.reshape(-1, 1)
        y_test = np.squeeze(y_test) / mean_y_train
        y_test = y_test.reshape(-1, 1)

        MyNeuralNet = mse_model(in_shape=in_shape, hidden_size=hidden_size, dropout=dropout)
        MyLearnerOptimized = LearnerOptimized(model=MyNeuralNet, optimizer_class=torch.optim.Adam, loss_func=nn.MSELoss(),
                                              device=device, optimizer_params=optimizer_params)
        MyLearnerOptimized.fit(x=x_train, y=y_train, epochs=max_epochs, batch_size=batch_size)

        torch.save(MyNeuralNet.state_dict(), f'model/{dataset_name}_{seed}.pt')
