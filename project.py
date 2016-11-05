import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

#print("Train data dimensions: ", train_data.shape)
#print("Test data dimensions: ", test_data.shape)

train_data.info()
test_data.info()

print("Number of missing values",train_data.isnull().sum().sum())

# setting pandas env variables to display max rows and columns
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows',1000)

# describing statistics of continuous variables
train_data.describe()

# describing statistics of categorical variables
train_data.describe(include = ['object'])

plt.figure(figsize=(13,9))
sns.distplot(np.log1p(train_data["loss"]))