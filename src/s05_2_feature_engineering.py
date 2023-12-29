#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import statsmodels.api as sm
import sys
from scipy import stats
from scipy.special import boxcox1p, logit
from scipy.stats import norm, skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures


sys.path.append(os.path.join('..', 'src'))


import importlib
import utils
importlib.reload(utils)

import params
importlib.reload(params)
from params import ProjectParameters

# # define paths and capture data

inputs = os.path.join('..', 'data', '02_intermediate')
outputs = os.path.join('..', 'data', '03_processed')
reports = os.path.join('..', 'data', '06_reporting')

# # set project parameters

numerical_cols = ProjectParameters().numerical_cols

# # plotting

def get_distribution(column, save_file):
    #Check the new distribution 
    sns.distplot(column, fit=norm);

    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(column)

    # plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                loc='best')
    plt.ylabel('Frequency')
    plt.title('estimate distribution')
    plt.savefig(os.path.join(reports,save_file+'.jpg'), bbox_inches = "tight")
    plt.close()
    
    fig = plt.figure()
    res = stats.probplot(column, plot=plt)
    plt.savefig(os.path.join(reports,save_file+'_regline.jpg'), bbox_inches = "tight")
    plt.close()
    
# # transformations

# build polynomials
def build_polynomials(df, cols, method = 'k_degree', degrees=2, testing = False):  
    print('number of columns before building polynomials:', df.shape[1])
    if method == 'simple_square':
        poly = df[cols]**2
        poly.columns = [c+'_power2' for c in df[cols].columns]
        df = pd.concat([df, poly], axis=1)
    elif method == 'k_degree':
        poly = PolynomialFeatures(2, include_bias = False)
        transformed = poly.fit_transform(df[cols])
        expanded_cols = poly.get_feature_names(df.columns)
        df = pd.DataFrame(transformed, columns = expanded_cols)
    print('number of columns after building polynomials:', df.shape[1])
    
    return df

# change skewness
def get_skewness(df):
    cols = ProjectParameters().numerical_cols
    df = df[cols]
    numeric_features = df.dtypes[(df.dtypes != "object") & (df.dtypes != "bool")].index.to_list()
    skewed_features = df[numeric_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'skew' :skewed_features})

    return skewness

def treat_skewness(df, df_name, bc_lambda = 0, testing=False):
    skewness = get_skewness(df)

    skewness = skewness[skewness['skew'] > 0.75]

    if testing:
        print("There are {} skewed features prone to Box Cox transform on {} data".format(skewness.shape[0], df_name))
        print(skewness.head(10))
    
    skewed_features = skewness.index

    for feature in skewed_features:
        df[feature] = boxcox1p(df[feature], bc_lambda)

    if testing:
        skewness = get_skewness(df)
        skewness = skewness[skewness['skew'] > 0.75]
        print("Done transformation to deal with skewness".format(skewness.shape[0]))
        print(skewness.head(10))

    return df

# label transformation is only applicable for regression problems
def transform_label(df, transformation = None, testing=False):
    print('transform_label function activated')
    df = df.copy()
    if transformation == 'log':
        df = np.log1p(df)
    elif transformation == 'logit':
        max_x = max(df)+1
        min_x = min(df)-1
        df = (df - min_x) / (max_x - min_x)
        df = logit(df)
    return df
