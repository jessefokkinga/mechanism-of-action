from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_tabnet.tab_model import TabNetRegressor
import numpy as np
import pandas as pd

# Run pre-processing script prior to training procedure
from models import simple_neural_net, transfer_learning_neural_net,TabularDataset, TransferLearningScheduler, \
    train_func, valid_func, SmoothBCEwLogits
from processing import perform_preprocessing

# Run preprocessing script and merge scored with non-scored targets
train_df, targets, targets_nonscored, _ = perform_preprocessing()
all_targets = all_targets = pd.concat([targets,targets_nonscored], axis = 1)

# Define parameters for during training procedure
epochs = 24
batch_size = 128
lr = 1e-3
early_stopping_steps = 10
pct_start = 0.1
early_stop = True
in_size = len(train_df.columns)
out_size = len(targets.columns)
out_size_full = len(targets.columns) + len(targets_nonscored.columns)
hidden_size = 1500
number_of_splits = 10
seeds = [0,42,1337,666,7]
device = ('cuda' if torch.cuda.is_available() else 'cpu')
weight_decay = {'all_targets': 1e-5, 'transfer_learn': 3e-6, 'standard': 1e-5}
max_lr = {'all_targets': 1e-2, 'transfer_learn': 3e-3,'standard': 1e-2}
div_factor = {'all_targets': 1e3, 'transfer_learn': 1e2, 'standard':1e3}

tabnet_params = dict(
    n_d = 32,
    n_a = 32,
    n_steps = 1,
    gamma = 1.3,
    lambda_sparse = 0,
    optimizer_fn = optim.Adam,
    optimizer_params = dict(lr = 2e-2, weight_decay = 1e-5),
    mask_type = "entmax",
    scheduler_params = dict(
        mode = "min", patience = 5, min_lr = 1e-5, factor = 0.9),
    scheduler_fn = ReduceLROnPlateau,
    verbose = 10
)


def train_model(model, tag_name, X_train, X_valid, y_train, y_valid, fine_tune_scheduler=None):

    train_ds = TabularDataset(X_train, y_train) #
    valid_ds = TabularDataset(X_valid, y_valid)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay[tag_name])
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                              steps_per_epoch=len(train_dl),
                                              pct_start=pct_start,
                                              div_factor=div_factor[tag_name],
                                              max_lr=max_lr[tag_name],
                                              epochs=epochs)

    loss_fn = nn.BCEWithLogitsLoss()
    loss_tr = SmoothBCEwLogits(smoothing=0.001)

    best_loss = np.inf
    step = 0

    for epoch in range(epochs):
        if fine_tune_scheduler is not None:
            fine_tune_scheduler.step(epoch, model)

        train_loss = train_func(model, optimizer, scheduler, loss_tr, train_dl, device)
        valid_loss, valid_preds = valid_func(model, loss_fn, valid_dl, device)
        print(
            f"SEED: {seed}, FOLD: {fold_nb}, {tag_name}, EPOCH: {epoch}, train_loss: {train_loss:.6f}, valid_loss: {valid_loss:.6f}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), f"trained_models/neural_net_{tag_name}_fold_{fold_nb}_{seed}.pth")

        elif early_stop:
            step += 1
            if step >= early_stopping_steps:
                break


for seed in seeds:
    mskf = MultilabelStratifiedKFold(n_splits=number_of_splits, random_state=seed, shuffle=True)
    
    for fold_nb, (train_idx, val_idx) in enumerate(mskf.split(train_df, targets)):
        print("FOLDS: ", fold_nb)

        # Prepare data
        X_train, y_train = train_df.values[train_idx, :], targets.values[train_idx, :]
        X_val, y_val = train_df.values[val_idx, :], targets.values[val_idx, :]
        y_train_all_targets, y_val_all_targets = all_targets.values[val_idx, :], all_targets.values[val_idx, :]

        # Train TabNet model
        model = TabNetRegressor(**tabnet_params)
        model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_val, y_val)],
            eval_name=["val"],
            eval_metric=["LogitsLogLoss"],
            max_epochs=200,
            patience=20,
            batch_size=1024,
            virtual_batch_size=32,
            num_workers=1,
            drop_last=False,
            loss_fn=F.binary_cross_entropy_with_logits
        )

        model.save_model("trained_models/tabnet_" + f"fold_{fold_nb}_{seed}")
        
        # Train simple feedforward neural network model
        model = simple_neural_net(num_features=in_size, num_targets=out_size,
                           hidden_size=hidden_size)
        model.to(device)

        train_model(model, 'standard', X_train, X_val, y_train, y_val, fine_tune_scheduler=None)

        # Train model that we will use for transfer learning 
        model = transfer_learning_neural_net(in_size, out_size_full)
        model.to(device)

        # Train on scored + nonscored targets
        train_model(model, 'all_targets', X_train, X_val, y_train_all_targets,
                    y_val_all_targets,  fine_tune_scheduler=None)
        
        # Load the pretrained model with the best loss
        pretrained_model = transfer_learning_neural_net(in_size, out_size_full)
        pretrained_model.load_state_dict(torch.load(f"neural_net_all_targets_{fold_nb}_{seed}.pth"))
        pretrained_model.to(device)

        # Copy model without the top layer
        fine_tune_scheduler = TransferLearningScheduler(epochs)
        final_model = fine_tune_scheduler.copy_without_last_layer(pretrained_model, in_size, out_size_full,
                                                           out_size)

        # Fine-tune the model on scored targets only
        train_model(final_model, 'transfer_learn', X_train, X_val, y_train, y_val, fine_tune_scheduler)

