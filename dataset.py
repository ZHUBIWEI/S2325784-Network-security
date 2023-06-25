import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class NB15Dataset(Dataset):
    def __init__(self, file_path, scalar_path, scaling_method, benign_only=True):
        self.data = pd.read_csv(file_path)
        self.X, self.y, self.attacks = self._process_data(scalar_path, scaling_method, benign_only)
    
    def set_indices(self, indices):
        # Set the indices for the dataset...
        self.indices = indices

    def _process_data(self, scalar_path, scaling_method, benign_only):
        X = self.data.iloc[:, :-2].to_numpy(dtype=np.float32)
        attacks = self.data["attack_cat"].to_numpy(dtype=object)
        y = self.data["Label"].to_numpy(dtype=np.float32)
        print(f"benign flows: {len(y) - y.sum()}\t attack flows: {y.sum()}")

        scalar = np.load(scalar_path)
        if scaling_method == "minmax":
            x_max = scalar[0]
            x_min = scalar[1]
            X = (X - x_min) / (x_max - x_min + 1e-8)

        if benign_only:
            benign_idx = y == 0
            X = X[benign_idx]
            attacks = attacks[benign_idx]
            y = y[benign_idx]

        return X, y, attacks

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index], self.attacks[index]


if __name__ == "__main__":
    dataset = DataLoader(
        NB15Dataset("./data/eval_set.csv", "./data/minmax_scalar.npy", scaling_method="minmax", benign_only=True),
        shuffle=True, batch_size=64, drop_last=True
    )
    for X, y, label in dataset:
        print(X.dtype)
        print(y.dtype)
        break
