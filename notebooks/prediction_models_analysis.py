# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Prediction Models Analysis
#

# The main goal in this notebook is to analyse the models with different configuration scenarios.

# ### General imports

# +
import math
import pymssql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import OrderedDict

import warnings
warnings.filterwarnings('ignore')
# -

import sys
print(sys.executable)

import sys
sys.path.append('..')

# +
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split

from sklearn.utils import resample

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    mean_absolute_error,
    median_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
    f1_score,
)
# -

from xgboost import XGBClassifier
from xgboost import XGBRegressor

# # Data Fetching

# The data is fetched the VFS DWH, from a single SQL table.
# In order to avoid overloading the database, the data was exported to a file in `parquet` format.

parquet_filename = "../datasets/data.parquet"

df = pd.read_parquet(parquet_filename)
# df = df.head(10000)

# # Data Processing

# Different function for transforming the columns:
# * datetime columns
# * numerical
# * categorical

label_column = "<< column >>"

# Separate `label_column` from the rest of the columns

x = df.loc[:, df.columns != label_column]
y = df[label_column].copy()

# #### X transform

# +
# data transformation
# -

# #### Y transform

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# ### Split data into train and validation
#

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)


# # Models

# Useful functions.

# +
def conf_matrix(y_val, y_pred):
    cm = confusion_matrix(y_val, y_pred)
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cm, annot=True, ax = ax, linewidths=.5, fmt='g')
    
def classification_metrics(y_val, y_pred, confusion_matrix=False, classification_report=False):
    if confusion_matrix:
        conf_matrix(y_val, y_pred)
        
    if classification_report:
        print(classification_report(y_val, y_pred))
    
    #print(
    #    f"ROC_AUC Score: {roc_auc_score(y_val, y_pred, average=None)}"
    #)
    print(
        f"Accuracy: {accuracy_score(y_val, y_pred)}"
    )
    print(
        f"F1 Score: {f1_score(y_val, y_pred)}"
    )


# -

# ## Evaluate DecisionTree Model for interpretability

# Since the ensemble of trees methods are quite difficult to interpret, one solution is to analyse a simpler model, and based on that
# to get a glimpse on how the learning process is working.
#
# In this work, the based ensemble method is decicion tree, so it will be used for analysis, because it's quite easy to visualise and
# interpret.

# ### Decicion Tree

from sklearn.tree import DecisionTreeClassifier
from dtreeviz.trees import *
from sklearn import tree

model = DecisionTreeClassifier()

model.fit(x_train, y_train)

# ## XGBoost

conf = {
    'objective': 'binary:logistic',
    'use_label_encoder': True,
    'base_score': 0.5,
    'booster': 'gbtree',
    'colsample_bylevel': 1,
    'colsample_bynode': 1,
    'colsample_bytree': 1,
    'gamma': 0,
    'gpu_id': -1,
    'importance_type': 'gain',
    'interaction_constraints': '',
    'learning_rate': 0.200000012,
    'max_delta_step': 0,
    'max_depth': 10,
    'min_child_weight': 1,
    'missing': None,
    'monotone_constraints': '()',
    'n_estimators': 100,
    'n_jobs': 12,
    'num_parallel_tree': 1,
    'random_state': 0,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'scale_pos_weight': 1,
    'subsample': 1,
    'tree_method': 'exact',
    'validate_parameters': 1,
    'verbosity': None,
}

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.fit_transform(y_val)

# +
model = XGBClassifier(**conf)

model.fit(
    x_train,
    np.squeeze(y_train),
    eval_set=[
        (x_val, y_val),
    ],
    eval_metric="auc",
    verbose=False,
)
# -

# ## NN - Pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable

# +
BATCH_SIZE = 64
EPOCH = 200

x_train_p = Variable(torch.tensor(x_train.values))
#y_train_p =  Variable(torch.nn.functional.one_hot(torch.tensor(y_train.values)))
y_train_p =  Variable(torch.tensor(y_train.values))
y_train_p = torch.reshape(y_train_p, (y_train_p.shape[0], 1))

torch_dataset = Data.TensorDataset(nn.functional.normalize(x_train_p.float()), y_train_p)

loader = Data.DataLoader(
    dataset=torch_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, num_workers=2,
)
# -

input_layer_dim = x_train_p.shape[1]
output_layer_dim = y_train_p.shape[1]

# +
net = nn.Sequential(
        nn.Linear(input_layer_dim, 200),
        nn.LeakyReLU(),
        nn.Linear(200, 100),
        nn.LeakyReLU(),
        nn.Linear(100, 1),
    )

optimizer = optim.Adam(net.parameters(), lr=0.01)
loss_func = nn.BCEWithLogitsLoss()
# -

for epoch in range(EPOCH):
    if epoch % 10 == 0:
        print(epoch)
    for step, (batch_x, batch_y) in enumerate(loader):
        
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        prediction = net(b_x.float())
        
        loss = loss_func(prediction, b_y.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

x_val_p = Variable(nn.functional.normalize(torch.tensor(x_val.values).float()))
#y_val_p =  Variable(torch.nn.functional.one_hot(torch.tensor(y_val.values)))
y_val_p =  Variable(torch.tensor(y_val.values))

pred = net(x_val_p) 

y_pred = torch.sigmoid(pred)

torch.round(y_pred).squeeze()

t = Variable(torch.Tensor([0.5]))  # threshold
pred = ((y_pred > t).float() * 1).int()

torch.round(
    (
        torch.true_divide(
        (
            pred.squeeze() == y_val_p).sum(),
            y_val_p.shape[0]
        )
    ) * 100
)
