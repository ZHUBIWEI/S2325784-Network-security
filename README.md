### Dependencies
```
pytorch
numpy
matplotlib
tqdm
scikit-learn
pandas
lime
```
### Dowload the UNSW-NB15 and put the dataset into the 'data' folder
We will use the UNSW-NB15 Dataset (Moustafa & Slay, 2015) or the other dataset as you want
- Moustafa, N. & Slay, J. "UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)." Military Communications and Information Systems Conference (MilCIS), 2015. IEEE, 2015

### How to run AE and VAE
Step 1:
Preprocess the dataset by run the file preprocessor.py directly
or can use the command
```shell
python preprocessor.py
```

The config files of AE and VAE is saved under `./configs`.
Replace `<ROOT_DIR>` in each config file with the actual path of this
directory and then run
```shell
python main.py --config_file ./configs/ae_exp.json
```
```shell
python main.py --config_file ./configs/vae_exp.json
```
If do have GPU, we can choose larger `batch_size`, else using smaller one,
e.g., 32 or 64, to avoid large memory consumption.

If you want to test different k-fold cross-validation, 
you can change the variable 'k' in main.py and run the above command
Or comment out the k-fold part, you can simply run the AE and VAE models

### How to run other anomaly detection methods.
In order to have a more comfortable and real-time results,

I decide to use jupyter notebook to show this part. 

You can see a folder name as 'lime_other_anomaly_detection_methods'.

After running the preprocessor, three datasets will be generated, namely: 'train_set', 'test_set' and 'eval_set'.

You need to put these three datasets into the folder 'lime_other_anomaly_detection_methods'.

Finally, you can run the Other anomaly decetion approaches.ipynb to perform other anomaly detection methods and their feature important

### How to use 'lime' to explain the AE and VAE models
Firstly, you should guarantee the datasets 'train_set', 'test_set' and 'eval_set' appear after running preprocessor in the folder of 'lime_other_anomaly_detection_methods'.

Secondly, after running the AE and VAE models, you can see the best model are stored in the folder 'models', you should take the ae.pth and vae.pth into the 'lime_other_anomaly_detection_methods'.

Finally, you can run the 'AE_lime' and 'VAE_lime'

Reference:
Munhouiani. PyTorch Implementation of GEE: A Gradient-based Explainable Variational Autoen-
coder for Network Anomaly Detection. [Online], March 2023. https://github.com/munhouiani/GEE.git.
