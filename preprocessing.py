import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA


def perform_preprocessing(variance_threshold = 0.75, n_pcomps_genes = 200, n_pcomps_cells = 200, data_path = "data/"):

    # Import data
    train = pd.read_csv(data_path + "train_features.csv")
    targets = pd.read_csv(data_path + "train_targets_scored.csv")
    targets_nonscored = pd.read_csv(data_path + "train_targets_nonscored.csv")
    test = pd.read_csv(data_path + "test_features.csv")

    # Remove control group
    train = train[train["cp_type"] != "ctl_vehicle"]
    test = test[test["cp_type"] != "ctl_vehicle"]
    targets = targets.iloc[train.index]
    targets_nonscored = targets_nonscored.iloc[train.index]
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    targets.reset_index(drop=True, inplace=True)
    targets_nonscored.reset_index(drop=True, inplace=True)

    # Remove columns with very low variance
    cols_numeric = [feat for feat in list(train.columns) if feat not in ["sig_id", "cp_type", "cp_time", "cp_dose"]]
    mask = train[cols_numeric].var() >= variance_threshold
    train = train[['sig_id', 'cp_type', 'cp_time', 'cp_dose'] + list(train[cols_numeric].columns[mask])]
    test = test[['sig_id', 'cp_type', 'cp_time', 'cp_dose']   + list(train[cols_numeric].columns[mask])]
    data_all = pd.concat([train, test], ignore_index = True)

    cols_numeric = [feat for feat in list(data_all.columns) if feat not in ["sig_id", "cp_type", "cp_time", "cp_dose"]]
    gene_columns = [col for col in data_all.columns if col.startswith("g-")]
    cell_columns = [col for col in data_all.columns if col.startswith("c-")]

    # Scale data
    scaler = QuantileTransformer(n_quantiles=100,random_state=0, output_distribution="normal").fit(train[cols_numeric])
    data_all[cols_numeric] = scaler.transform(data_all[cols_numeric])

    # Perform principal component analysis
    pca_genes_transformer = PCA(n_components=n_pcomps_genes,
                                random_state=42).fit(train[gene_columns])
    pca_genes = pca_genes_transformer.transform(data_all[gene_columns])

    pca_cells_transformer = PCA(n_components=n_pcomps_cells,
                                random_state=42).fit(train[cell_columns])
    pca_cells = pca_cells_transformer.transform(data_all[cell_columns])

    pca_genes = pd.DataFrame(pca_genes, columns=[f"pca_g-{i}" for i in range(n_pcomps_genes)])
    pca_cells = pd.DataFrame(pca_cells, columns=[f"pca_c-{i}" for i in range(n_pcomps_cells)])
    data_all = pd.concat([data_all, pca_genes, pca_cells], axis=1)

    data_all = pd.get_dummies(data_all, columns = ["cp_time", "cp_dose"])

    # Add row-wise descriptive statistics about distribution of the gene features
    for stats in ["sum", "mean", "std", "kurt", "skew"]:
        data_all["g_" + stats] = getattr(data_all[gene_columns], stats)(axis = 1)
        data_all["c_" + stats] = getattr(data_all[cell_columns], stats)(axis = 1)
        data_all["gc_" + stats] = getattr(data_all[gene_columns + cell_columns], stats)(axis = 1)

    train_df = data_all[: train.shape[0]]
    train_df.reset_index(drop = True, inplace = True)
    test_df = data_all[train_df.shape[0]: ]
    test_df.reset_index(drop = True, inplace = True)

    train_df.drop(["sig_id", "cp_type"], axis = 1, inplace = True)
    test_df.drop(["sig_id", "cp_type"], axis = 1, inplace = True)
    targets.drop("sig_id", axis = 1, inplace = True)
    targets_nonscored.drop("sig_id", axis = 1, inplace = True)

    return train_df, targets, targets_nonscored, test_df


def prepare_submission(predictions, data_path="data/", weights = None):
    if not weights:
        weights = [1/3,1/3,1/3]

    # Load test and submission data as raw inputs
    submission = pd.read_csv(data_path + "sample_submission.csv")
    test = pd.read_csv(data_path + "test_features.csv")

    # Save the names of all targets, remove rows with control vehicle
    target_names = [col for col in submission.columns if col not in ["sig_id"]]
    sig_id = test[test["cp_type"] != "ctl_vehicle"].sig_id.reset_index(drop=True)

    # blend the predictions
    blended_predictions = predictions[0].mean(axis=0) * weights[0] + predictions[1].mean(axis=0) * weights[1] \
                          + predictions[2].mean(axis=0) * weights[2]

    # Convert to dataframe, set column names and add identifier
    blended_predictions = pd.DataFrame(blended_predictions, columns=target_names)
    blended_predictions["sig_id"] = sig_id

    submission = pd.merge(test[["sig_id"]], blended_predictions, on="sig_id", how="left")
    submission.fillna(0, inplace=True)
    submission.loc[test["cp_type"] == 0, submission.columns[1:]] = 0
    return submission
