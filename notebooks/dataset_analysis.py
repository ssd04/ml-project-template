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

# # Data Analysis
#
# Analyse the dataset. This is suitable if it is `tabular data`.

# ## General imports

# +
import pymssql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')
# -

import sys
sys.path.append('..')

# ## Fetch Data

# For temporarly use, the dataset has been stored as a `parquet` file, in order to avoid overloading the database.

parquet_filename = "./datasets/data.parquet"

df = pd.read_parquet(parquet_filename)
# df = df.head(10000)

# ## Data Overview

df

df.info()

# ### Class Distribution

y = df['<< class label name >>'].astype(int)

y.value_counts(normalize=True) * 100

sns.distplot(y)

# ## Columns
#
# Get a quick understanding of the columns and see if any of them could be useful.
# Showing the distribution.

# ### << column name >>
#
# Determine if the customer (the debtor) is a company or person

df["<< column name >>"].value_counts()

df["<< column name >>"].value_counts().hist()

df["<< column name >>"].isnull().value_counts()

df["<< column name >>"].isnull().sum() * 100 / len(df["<< column name >>"])

# ### Groupby
#
# Lets investigate multiple columns together.

df_gb = df[['<< column 1 >>', '<< column 2 >>']].groupby(by='<< column 1 >>').sum()
df_gb.sort_values(by='<< column 2 >>')

# ### Training data
#
# This will create a data set that has the labels we want. This is what we are going to use for training the model.

train = df[df['<< column label >>'].notnull()]

# ### Test data
#
# Check to see which values have no closure date, and should also have no closureid as well.
# In production this is the data that we will be using to make predictions on. However for training this kind of data is not useful for us as we don't have the "labels". 

df[df['<< label column >>'].isnull()]

test = df[df['<< label column >>'].isnull()]

# ### Missing/NULL Values
#
# Check how many null values are in the dataset

df.isnull().sum()

# ### Class relations

df.hist(figsize = (10, 10));

g = sns.PairGrid(df)
g.map(sns.scatterplot)

# ## Pandas Profiling
#
# Generates profile reports from the dataset, automatically using `pandas_profiling` library.

from pandas_profiling import ProfileReport

profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)

# # ! Important note, it will use quite a lot of resources.

profile

# ## Great Expectations Analysis

# The library can be used to setup thresholds when it comes to null values in the dataset.
#
# It is not in place at the application level, right now.

import great_expectations as ge

ge_df = ge.from_pandas(df)

ge_df.get_expectation_suite()

ge_df

ge_df.expect_column_values_to_not_be_null(column="<< column name >>")
