### Dependencies
```
pytorch
numpy
matplotlib
tqdm
scikit-learn
pandas
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
If you want to try k-fold cross-validation, you can change the 'main_CV' to 'main'
and run the above command

### How to run other anomaly detection methods.
In order to have a more comfortable and real-time results,
I decide to use jupyter notebook to show this part. 
You can see a folder in the .zip name as 'lime_other_anomaly_detection_methods'


Reference:
Munhouiani. PyTorch Implementation of GEE: A Gradient-based Explainable Variational Autoen-
coder for Network Anomaly Detection. [Online], March 2023. https://github.com/munhouiani/GEE.git.
