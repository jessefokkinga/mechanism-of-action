import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss
import numpy as np


class simple_neural_net(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size, dropout_rate = 0.2):
        super(simple_neural_net, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.leaky_relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        return x


class transfer_learning_neural_net(nn.Module):
    def __init__(self, num_features, num_targets, dropout_rates = [0.5, 0.35, 0.3, 0.25],
                 hidden_size = [1500, 1250, 1000, 750]):
        super(transfer_learning_neural_net, self).__init__()
        self.dropout_rates = dropout_rates
        self.hidden_size = hidden_size

        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.Linear(num_features, hidden_size[0])

        self.batch_norm2 = nn.BatchNorm1d(hidden_size[0])
        self.dropout2 = nn.Dropout(dropout_rates[0])
        self.dense2 = nn.Linear(hidden_size[0], hidden_size[1])

        self.batch_norm3 = nn.BatchNorm1d(hidden_size[1])
        self.dropout3 = nn.Dropout(dropout_rates[1])
        self.dense3 = nn.Linear(hidden_size[1], hidden_size[2])

        self.batch_norm4 = nn.BatchNorm1d(hidden_size[2])
        self.dropout4 = nn.Dropout(dropout_rates[2])
        self.dense4 = nn.Linear(hidden_size[2], hidden_size[3])

        self.batch_norm5 = nn.BatchNorm1d(hidden_size[3])
        self.dropout5 = nn.Dropout(dropout_rates[3])
        self.dense5 = nn.utils.weight_norm(nn.Linear(hidden_size[3], num_targets))

    def forward(self, x):
        x = self.batch_norm1(x)
        x = F.leaky_relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = F.leaky_relu(self.dense3(x))

        x = self.batch_norm4(x)
        x = self.dropout4(x)
        x = F.leaky_relu(self.dense4(x))

        x = self.batch_norm5(x)
        x = self.dropout5(x)
        x = self.dense5(x)
        return x


class TabularDataset:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return (self.X.shape[0])

    def __getitem__(self, i):
        X_i = torch.tensor(self.X[i, :], dtype=torch.float)
        y_i = torch.tensor(self.y[i, :], dtype=torch.float)

        return X_i, y_i


class TabularDatasetTest:

    def __init__(self, X):
        self.X = X

    def __len__(self):
        return (self.X.shape[0])

    def __getitem__(self, i):
        X_i = torch.tensor(self.X[i, :], dtype=torch.float)
        return X_i


def valid_func(model, loss_func, dataloader, device):
    model.eval()

    valid_loss = 0
    valid_preds = []

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = loss_func(outputs, labels)

        valid_loss += loss.item()
        valid_preds.append(outputs.sigmoid().detach().cpu().numpy())

    valid_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)

    return valid_loss, valid_preds


def train_func(model, optimizer, scheduler, loss_func, dataloader, device):
    train_loss = 0

    model.train()
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()

    train_loss /= len(dataloader)

    return train_loss


def inference_func(model, dataloader, device):
    model.eval()
    preds = []

    for data in dataloader:
        inputs = data.to(device)

        with torch.no_grad():
            outputs = model(inputs)

        preds.append(outputs.sigmoid().detach().cpu().numpy())

    preds = np.concatenate(preds)

    return preds


class TransferLearningScheduler:
    def __init__(self, epochs, device, model):
        self.epochs = epochs
        self.model = model
        self.device  = device
        self.epochs_per_step = 0
        self.frozen_layers = []

    def copy_without_last_layer(self, model, num_features, num_targets, num_targets_new):
        self.frozen_layers = []

        model_new = model(num_features, num_targets)
        model_new.load_state_dict(model.state_dict())

        for name, param in model_new.named_parameters():
            layer_index = name.split('.')[0][-1]

            if layer_index == 5:
                continue

            param.requires_grad = False

            if layer_index not in self.frozen_layers:
                self.frozen_layers.append(layer_index)

        self.epochs_per_step = self.epochs // len(self.frozen_layers)

        model_new.batch_norm5 = nn.BatchNorm1d(model_new.hidden_size[3])
        model_new.dropout5 = nn.Dropout(model_new.dropout_value[3])
        model_new.dense5 = nn.utils.weight_norm(nn.Linear(model_new.hidden_size[-1], num_targets_new))
        model_new.to(self.device)
        return model_new

    def step(self, epoch, model):
        if len(self.frozen_layers) == 0:
            return

        if epoch % self.epochs_per_step == 0:
            last_frozen_index = self.frozen_layers[-1]

            for name, param in model.named_parameters():
                layer_index = name.split('.')[0][-1]

                if layer_index == last_frozen_index:
                    param.requires_grad = True

            del self.frozen_layers[-1]


class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
            self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss