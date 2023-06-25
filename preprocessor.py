import torch
import numpy as np
import pandas as pd

biased_features = ["srcip", "sport", "dstip", "dsport", "stcpb", "dtcpb", "Stime", "Ltime"]
onehot_features = ["proto", "state", "service"]


def insert_feature_name(raw_file_path, feature_name_path, output_path):
    with open(feature_name_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    feature_names = [line.strip() for line in lines]
    feature_line = ",".join(feature_names) + "\n"
    with open(output_path, 'w', encoding='utf-8') as out_f:
        out_f.write(feature_line)
        with open(raw_file_path, 'r', encoding='utf-8') as in_f:
            for line in in_f:
                out_f.write(line)


def data_preprocessing(dataset_path, data_dir):
    dataset = pd.read_csv(dataset_path)

    # Drop biased columns
    print(f"Number of columns: {len(dataset.columns)}")
    dataset.drop(biased_features, axis=1, inplace=True, errors="ignore")
    print(f"Number of columns after dropping: {len(dataset.columns)}")

    # One-hot encoding
    onehot_df = dataset[onehot_features]
    numerical_df = dataset.drop(onehot_features, axis=1)
    dfs = []
    for col_name in onehot_df.columns:
        print(f"Found {len(onehot_df[col_name].unique())} categories in {col_name}")
        print(onehot_df[col_name].unique())
        encoded_df = pd.get_dummies(onehot_df[col_name])
        dfs.append(encoded_df)
    dfs.append(numerical_df)
    encoded_dataset = pd.concat(dfs, axis=1)
    print(f"Number of columns after encoding: {len(encoded_dataset.columns)}")

    # Replace NaN values in 'attack_cat' column with "none"
    encoded_dataset["attack_cat"] = encoded_dataset["attack_cat"].replace(np.nan, 'None')

    # Train/eval/test split
    encoded_dataset = encoded_dataset.sample(frac=1).reset_index(drop=True)
    train_prop = 0.5
    eval_prop = 0.15
    test_prop = 0.35
    train_num = int(len(encoded_dataset) * train_prop)
    eval_num = int(len(encoded_dataset) * eval_prop)
    test_num = len(encoded_dataset) - train_num - eval_num
    print("Splitting")
    print(f"train_num: {train_num}\t eval_num: {eval_num}\t test_num: {test_num}")
    train_set = encoded_dataset.iloc[:train_num]
    eval_set = encoded_dataset.iloc[train_num:train_num + eval_num]
    test_set = encoded_dataset.iloc[train_num + eval_num:]

    print("Saving datasets...")
    train_set.to_csv(data_dir + "train_set.csv", index=False)
    eval_set.to_csv(data_dir + "eval_set.csv", index=False)
    test_set.to_csv(data_dir + "test_set.csv", index=False)

    # Compute min-max scalar
    data = encoded_dataset.iloc[:, :-2].to_numpy(dtype=np.float32)
    x_max = data.max(axis=0).reshape((1, -1))
    x_min = data.min(axis=0).reshape((1, -1))
    print(f"Scalar dim: {x_max.shape}")
    scalar = np.concatenate([x_max, x_min], axis=0)
    np.save(data_dir + "minmax_scalar.npy", scalar)

if __name__ == "__main__":
    data_dir = "./data/"
    raw_file_path = "./data/UNSW-NB15_1.csv"
    feature_path = "./data/feature_names.txt"
    dataset_path = "./data/dataset_with_feature_names.csv"

    insert_feature_name(raw_file_path, feature_path, dataset_path)
    data_preprocessing(dataset_path, data_dir)

    # Convert train_set.csv, eval_set.csv, and test_set.csv to torch tensors
    train_set = pd.read_csv(data_dir + "train_set.csv")
    eval_set = pd.read_csv(data_dir + "eval_set.csv")
    test_set = pd.read_csv(data_dir + "test_set.csv")

    train_tensor = torch.tensor(train_set.values, dtype=torch.float32)
    eval_tensor = torch.tensor(eval_set.values, dtype=torch.float32)
    test_tensor = torch.tensor(test_set.values, dtype=torch.float32)


