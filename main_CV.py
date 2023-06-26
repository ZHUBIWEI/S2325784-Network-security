import argparse
from Body import Body
import json
from utils import AttributeAccessibleDict
from sklearn.model_selection import KFold

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import Timer
from models import VAE, AutoEncoder
from dataset import NB15Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, precision_recall_curve


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Anomaly Detection Exp")
    parser.add_argument("--config_file", type=str, default=None)
    args = parser.parse_args()
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    config = AttributeAccessibleDict(config)

    # Load Exp class
    exp = Body(args=config)

    # k-fold Cross-validation
    # Control the number of fold
    k = 5
    
    kfold = KFold(n_splits=k, shuffle=True)
    fold = 1
    for train_indices, val_indices in kfold.split(exp.train_loader.dataset):
        print(f"Fold: {fold}")
        exp.train_loader.dataset.set_indices(train_indices)
        exp.val_loader.dataset.set_indices(val_indices)
        fold += 1

        exp.model, exp.optimizer = exp.init_model_optimizer()  # Initialize model and optimizer for each fold

        exp.train_losses = {
            "loss": [],
            "CE": [],
            "KLD": []
        }
        exp.test_losses = {
            "loss": [],
            "CE": [],
            "KLD": []
        }

        best_f1 = 0
        best_threshold = 0
        for epoch in range(1, exp.epoch_num + 1):
            losses = exp.train(epoch)
            exp.log_losses(losses, train=True)
            losses = exp.test(validate=True)
            exp.log_losses(losses, train=False)
            recon_errors, labels = exp.get_recon_errors_and_labels(validate=True)
            epoch_f1, threshold = exp.compute_best_f1(recon_errors, labels, return_threshold=True)
            print(f"F1 score: {epoch_f1:.4f} at: {threshold:.4f}")
            if epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_threshold = threshold
                # Save model
                if torch.cuda.device_count() > 1:
                    torch.save(exp.model.module.state_dict(), f"{exp.save_dir}{exp.model_name}_fold{fold}.pth")
                else:
                    torch.save(exp.model.state_dict(), f"{exp.save_dir}{exp.model_name}_fold{fold}.pth")

        # Load the best model
        exp.model.load_state_dict(torch.load(f"{exp.save_dir}{exp.model_name}_fold{fold}.pth"))
        recon_errors, labels = exp.get_recon_errors_and_labels(validate=False)
        exp.compute_all_metrics(recon_errors, labels, best_threshold)
        exp.visualize_losses()


