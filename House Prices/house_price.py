#!/usr/bin/env python
# coding=utf-8

# Import

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline

#bring in the six packs
df_train = pd.read_csv('./train.csv')

print df_train.columns