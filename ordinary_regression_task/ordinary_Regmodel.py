import torch
import torch.nn as nn
import sys
import copy
import numpy as np
from sklearn.model_selection import train_test_split

def epoch_internal_train(model, loss_func, x_train, y_train, batch_size, optimizer, cnt=0, best_cnt=np.inf):
    """ Sweep over the data and update the model's parameters

    Parameters
    ----------

    model : class of neural net model
    loss_func : class of loss function
    x_train : pytorch tensor n training features, each of dimension p (nXp)
    batch_size : integer, size of the mini-batch
    optimizer : class of SGD solver
    cnt : integer, counting the gradient steps
    best_cnt: integer, stop the training if current cnt > best_cnt

    Returns
    -------

    epoch_loss : mean loss value
    cnt : integer, cumulative number of gradient steps

    """

    model.train()
    shuffle_idx = np.arange(x_train.shape[0])
    np.random.shuffle(shuffle_idx)
    x_train = x_train[shuffle_idx]
    y_train = y_train[shuffle_idx]
    epoch_losses = []
    for idx in range(0, x_train.shape[0], batch_size):
        cnt = cnt + 1
        optimizer.zero_grad()
        batch_x = x_train[idx : min(idx + batch_size, x_train.shape[0]),:]
        batch_y = y_train[idx : min(idx + batch_size, y_train.shape[0])]
        preds = model(batch_x)
        loss = loss_func(preds, batch_y)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.cpu().detach().numpy())

        if cnt >= best_cnt:
            break

    epoch_loss = np.mean(epoch_losses)

    return epoch_loss, cnt

class mse_model(nn.Module):
    """ Conditional mean estimator, formulated as neural net
    """

    def __init__(self,
                 in_shape,
                 hidden_size=64,
                 dropout=0.1):
        """ Initialization

        Parameters
        ----------

        in_shape : integer, input signal dimension (p)
        hidden_size : integer, hidden layer dimension
        dropout : float, dropout rate

        """

        super().__init__()
        self.in_shape = in_shape
        self.out_shape = 1
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.build_model()
        self.init_weights()

    def build_model(self):
        """ Construct the network
        """
        self.base_model = nn.Sequential(
            nn.Linear(self.in_shape, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1),
        )

    def init_weights(self):
        """ Initialize the network parameters
        """
        for m in self.base_model:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """ Run forward pass
        """
        return self.base_model(x)


# Define the training procedure
class LearnerOptimized:
    """ Fit a neural network (conditional mean) to training data
    """
    def __init__(self, model, optimizer_class, loss_func, device='cpu', test_ratio=0.2, random_state=0, optimizer_params=None):
        """ Initialization

        Parameters
        ----------

        model : class of neural network model
        optimizer_class : class of SGD optimizer (e.g. Adam)
        loss_func : loss to minimize
        device : string, "cuda:0" or "cpu"
        test_ratio : float, test size used in cross-validation (CV)
        random_state : int, seed to be used in CV when splitting to train-test

        """
        self.model = model.to(device)
        self.optimizer_class = optimizer_class
        if optimizer_params is None:
            self.optimizer_params = {}
        else:
            self.optimizer_params = optimizer_params
        self.optimizer = optimizer_class(self.model.parameters(), **self.optimizer_params)
        self.loss_func = loss_func.to(device)
        self.device = device
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.loss_history = []
        self.test_loss_history = []
        self.full_loss_history = []

    def fit(self, x, y, epochs, batch_size, verbose=False):
        """ Fit the model to data

        Parameters
        ----------

        x : numpy array, containing the training features (nXp)
        y : numpy array, containing the training labels (n)
        epochs : integer, maximal number of epochs
        batch_size : integer, mini-batch size for SGD

        """

        sys.stdout.flush()
        model = copy.deepcopy(self.model)
        model = model.to(self.device)
        optimizer = self.optimizer_class(model.parameters(), **self.optimizer_params)
        best_epoch = epochs

        x_train, xx, y_train, yy = train_test_split(x, y, test_size=self.test_ratio,random_state=self.random_state)

        x_train = torch.from_numpy(x_train).float().to(self.device).requires_grad_(False)
        xx = torch.from_numpy(xx).float().to(self.device).requires_grad_(False)
        y_train = torch.from_numpy(y_train).float().to(self.device).requires_grad_(False)
        yy = torch.from_numpy(yy).float().to(self.device).requires_grad_(False)

        best_cnt = 1e10
        best_test_epoch_loss = 1e10

        cnt = 0
        for e in range(epochs):
            epoch_loss, cnt = epoch_internal_train(model, self.loss_func, x_train, y_train, batch_size, optimizer, cnt)
            self.loss_history.append(epoch_loss)

            # test
            model.eval()
            preds = model(xx)
            test_epoch_loss = self.loss_func(preds, yy).cpu().detach().numpy()

            self.test_loss_history.append(test_epoch_loss)

            if (test_epoch_loss <= best_test_epoch_loss):
                best_test_epoch_loss = test_epoch_loss
                best_epoch = e
                best_cnt = cnt

            if (e+1) % 100 == 0 and verbose:
                print("CV: Epoch {}: Train {}, Test {}, Best epoch {}, Best loss {}".format(e+1, epoch_loss, test_epoch_loss, best_epoch, best_test_epoch_loss))
                sys.stdout.flush()

        # use all the data to train the model, for best_cnt steps
        x = torch.from_numpy(x).float().to(self.device).requires_grad_(False)
        y = torch.from_numpy(y).float().to(self.device).requires_grad_(False)

        cnt = 0
        for e in range(best_epoch+1):
            if cnt > best_cnt:
                break

            epoch_loss, cnt = epoch_internal_train(self.model, self.loss_func, x, y, batch_size, self.optimizer, cnt, best_cnt)
            self.full_loss_history.append(epoch_loss)

            if (e+1) % 100 == 0 and verbose:
                print("Full: Epoch {}: {}, cnt {}".format(e+1, epoch_loss, cnt))
                sys.stdout.flush()

    def predict(self, x):
        """ Estimate the label given the features

        Parameters
        ----------
        x : numpy array of training features (nXp)

        Returns
        -------
        ret_val : numpy array of predicted labels (n)

        """
        self.model.eval()
        ret_val = self.model(torch.from_numpy(x).float().to(self.device).requires_grad_(False)).cpu().detach().numpy()
        return ret_val
