import enum
import sys

sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    fn = nn.Sequential(nn.Linear(in_features=dim, out_features=hidden_dim),
                       norm(dim=hidden_dim), nn.ReLU(), nn.Dropout(drop_prob),
                       nn.Linear(in_features=hidden_dim, out_features=dim),
                       norm(dim=dim))
    block = nn.Sequential(nn.Residual(fn), nn.ReLU())
    return block
    ### END YOUR SOLUTION


def MLPResNet(dim,
              hidden_dim=100,
              num_blocks=3,
              num_classes=10,
              norm=nn.BatchNorm1d,
              drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    # blocks = nn.Sequential()
    resnet = nn.Sequential(
        nn.Linear(in_features=dim, out_features=hidden_dim), nn.ReLU(), *[
            ResidualBlock(dim=hidden_dim,
                          hidden_dim=hidden_dim // 2,
                          norm=norm,
                          drop_prob=drop_prob) for _ in range(num_blocks)
        ], nn.Linear(in_features=hidden_dim, out_features=num_classes))
    return resnet
    ### END YOUR SOLUTION


# TODO bug about computing accuracy
def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_fn = nn.SoftmaxLoss()
    loss_list = []
    acc_list = []
    if opt:
        model.train()
        for i, batch in enumerate(dataloader):
            x, y = batch[0], batch[1]
            x = x.reshape((x.shape[0], -1))
            y_hat = model(x)
            logit = np.argmax(y_hat.cached_data, axis=1)
            loss = loss_fn(y_hat, y)
            loss_list.append(loss.cached_data)
            correct = np.sum(logit == y.cached_data)
            acc_list.append(correct / x.shape[0])
            opt.reset_grad()
            loss.backward()
            opt.step()
        n_sample = 0
        correct = 0
        for i, batch in enumerate(dataloader):
            x, y = batch[0], batch[1]
            n_sample += x.shape[0]
            x = x.reshape((x.shape[0], -1))
            y_hat = model(x)
            logit = np.argmax(y_hat.cached_data, axis=1)
            correct += np.sum(logit == y.cached_data)
        acc, loss = correct / n_sample, np.mean(loss_list)
        # acc, loss = np.mean(acc_list), np.mean(loss_list)
        return np.array([acc, loss])
    else:
        model.eval()
        n_sample = 0
        correct = 0
        for i, batch in enumerate(dataloader):
            x, y = batch[0], batch[1]
            n = x.shape[0]
            n_sample += n
            x = x.reshape((n, -1))
            y_hat = model(x)
            logit = np.argmax(y_hat.cached_data, axis=1)
            loss = loss_fn(y_hat, y)
            loss_list.append(loss.cached_data * n)
            counter = np.sum(logit == y.cached_data)
            correct += counter

        acc, loss = correct / n_sample, np.sum(loss_list) / n_sample
        return np.array([acc, loss])

    ### END YOUR SOLUTION


def train_mnist(batch_size=100,
                epochs=10,
                optimizer=ndl.optim.Adam,
                lr=0.001,
                weight_decay=0.001,
                hidden_dim=100,
                data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
