import argparse
from Experiment import Experiment
import json
from utils import AttributeAccessibleDict
import numpy as np
import pandas as pd

from models import VAE, AutoEncoder
import lime
import torch
import torch.nn as nn
from dataset import NB15Dataset
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

def load_dataset(train_path, test_path, scalar_path, scaling_method):
    train_set = NB15Dataset(train_path, scalar_path, scaling_method, benign_only=False)
    test_set = NB15Dataset(test_path, scalar_path, scaling_method, benign_only=False)
    feature = pd.read_csv(train_path)
    feature_name = feature.columns
    return train_set.X, train_set.y, test_set.X, test_set.y,feature_name


if __name__ == '__main__':


    # configuration
    train_path = "./data/train_set.csv"
    test_path = "./data/test_set.csv"
    scalar_path = "./data/minmax_scalar.npy"
    scaling_method = "minmax"

    # load data
    train_X, train_y, test_X, test_y, feature_name = load_dataset(train_path, test_path, scalar_path, scaling_method)



    # Load Exp class
    #exp = Experiment(args=config)
    #exp.start_experiment()

    pth_dir = "./models/"
    # 加载训练好的模型参数
    #model_ae.load_state_dict(torch.load(pth_dir +'ae.pth'))
    #model_vae.load_state_dict(torch.load('vae.pth'))
    clf = torch.load(pth_dir +'ae.pth')
    model_vae = torch.load(pth_dir +'vae.pth')
    # Lime解释
    explainer = LimeTabularExplainer(train_X, feature_names=feature_name, class_names=["Normal", "Attack"])

    # Explaination
    test_instance_index = 0
    exp = explainer.explain_instance(test_X[test_instance_index], clf.predict_proba, num_features=10)
    
    # Visualization
    fig = exp.as_pyplot_figure()
    
    # Save as jpg
    plt.tight_layout()
    plt.savefig("./plots/" +"lime_logistic.jpg", format='jpg', dpi=300)



