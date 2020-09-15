# Import Module

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

train = pd.read_csv('/kaggle/input/mnist-datasets/train.csv')
test  = pd.read_csv('/kaggle/input/mnist-datasets/test.csv')
# submission = pd.read_csv('/kaggle/input/mnist-datasets/submission.csv')

# 필요한 부분 indexing
train_img = train.iloc[:,3:].to_numpy().reshape(-1,28,28,1)
train_digit = train['digit']
train_letter = train['letter']
test_img = test.iloc[:,2:].to_numpy().reshape(-1,28,28,1)
test_letter = test['letter']

# array 정규화
train_img_norm = train_img / 255.0
test_img_norm = test_img / 255.0

features = train_img_norm
labels = train_digit.to_numpy()

# test_set 저장
X_test = test_img_norm

from sklearn.model_selection import train_test_split

validation_split = 0.1

X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size = validation_split, random_state = 1004)

