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

        MyNeuralNet = mse_model(in_shape=in_shape, hidden_size=hidden_size, dropout=dropout)
        MyLearnerOptimized = LearnerOptimized(model=MyNeuralNet, optimizer_class=torch.optim.Adam, loss_func=nn.MSELoss(),
                                              device=device, optimizer_params=optimizer_params)
        MyLearnerOptimized.fit(x=x_train[idx_train], y=y_train[idx_train], epochs=max_epochs, batch_size=batch_size)

        torch.save(MyNeuralNet.state_dict(), f'model/{dataset_name}_{seed}.pt')
