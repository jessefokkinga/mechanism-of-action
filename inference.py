from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_tabnet.tab_model import TabNetRegressor
import numpy as np

# Run pre-processing script prior to training procedure
from models import simple_neural_net, transfer_learning_neural_net,TabularDataset, TransferLearningScheduler, \
    train_func, valid_func, SmoothBCEwLogits, TabularDatasetTest, inference_func
from processing import perform_preprocessing, prepare_submission

# Run preprocessing script and merge scored with non-scored targets
train_df, targets, targets_nonscored, test_df = perform_preprocessing()
X_test = test_df.values

# Define parameters for during training procedure
epochs = 24,
batch_size = 128,
lr = 1e-3,
early_stopping_steps = 10,
pct_start = 0.1
early_stop = True,
in_size = len(train_df.columns),
out_size = len(targets.columns),
out_size_full = len(targets.columns) + len(targets_nonscored.columns),
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

test_preds_tabnet = []
test_preds_nn = []
test_preds_nn_transfer_learn = []

for seed in seeds:
    mskf = MultilabelStratifiedKFold(n_splits=number_of_splits, random_state=seed, shuffle=True)
    for fold_nb, (train_idx, val_idx) in enumerate(mskf.split(train_df, targets)):
        print("FOLDS: ", fold_nb)

        model = TabNetRegressor(**tabnet_params)
        model.load_model("trained_models/tabnet_" + f"fold_{fold_nb}_{seed}.zip")
        preds_test = model.predict(X_test)
        preds_test = 1 / (1 + np.exp(-preds_test))

        test_preds_tabnet.append(preds_test)

        test_ds = TabularDatasetTest(X_test)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model = simple_neural_net(num_features=in_size, num_targets=out_size,
                                 hidden_size=hidden_size)

        if device == "cpu":
            model.load_state_dict(torch.load(f"trained_models/neural_net_standard_fold_{fold_nb}_{seed}.pth",
                                             map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(f"trained_models/neural_net_standard_fold_{fold_nb}_{seed}.pth"))

        model.to(device)

        preds_test = np.zeros((test_df.shape[0], out_size))
        preds_test = inference_func(model, test_dl, device)

        test_preds_nn.append(preds_test)

        model = transfer_learning_neural_net(num_features=in_size, num_targets=out_size,
                                 hidden_size=hidden_size)

        if device == "cpu":
            model.load_state_dict(torch.load(f"trained_models/neural_net_transfer_learn_fold_{fold_nb}_{seed}.pth",
                                             map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(f"trained_models/neural_net_transfer_learn_fold_{fold_nb}_{seed}.pth"))

        model.to(device)

        preds_test = np.zeros((test_df.shape[0], out_size))
        preds_test = inference_func(model, test_dl, device)

        test_preds_nn_transfer_learn.append(preds_test)

# Convert lists of predictions into numpy arrays
test_preds_tabnet = np.stack(test_preds_tabnet)
test_preds_nn  = np.stack(test_preds_nn)
test_preds_nn_transfer_learn = np.stack(test_preds_nn_transfer_learn)

# Prepare predictions for the test data by creating a submission csv
predictions = [test_preds_tabnet, test_preds_nn, test_preds_nn_transfer_learn]
submission = prepare_submission(data_path= "/data", predictions = predictions, weights = [1/3,1/3,1/3])

# Write submission csv to folder (this will be submitted to Kaggle)
submission.to_csv("submission.csv", index = None)
