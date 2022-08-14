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

# Define parameters relevant during inference
batch_size = 128
in_size = len(train_df.columns)
out_size = len(targets.columns)
hidden_size = 1500
number_of_splits = 10
seeds = [0,42,1337,666,7]
device = ('cuda' if torch.cuda.is_available() else 'cpu')

test_preds_tabnet = []
test_preds_nn = []
test_preds_nn_transfer_learn = []

for seed in seeds:
    mskf = MultilabelStratifiedKFold(n_splits=number_of_splits, random_state=seed, shuffle=True)
    for fold_nb, (train_idx, val_idx) in enumerate(mskf.split(train_df, targets)):
        print("FOLDS: ", fold_nb)

        # Generate TabNet predictions
        model = TabNetRegressor()
        model.load_model("trained_models/tabnet_" + f"fold_{fold_nb}_{seed}.zip")
        preds_test = model.predict(X_test)
        preds_test = 1 / (1 + np.exp(-preds_test))

        test_preds_tabnet.append(preds_test)

        # Load datasets via PyTorch dataloader
        test_ds = TabularDatasetTest(X_test)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # Generate predictions for our simple neural network
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

        # Generate predictions from our transfer learning model
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
submission = prepare_submission(data_path= "data/", predictions = predictions, weights = [1/3,1/3,1/3])

# Write submission csv to folder (this will be submitted to Kaggle)
submission.to_csv("submission.csv", index = None)
