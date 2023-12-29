import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imb_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer as Imputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.path.append(os.path.join('..', 'src'))
from s04_encoding import ordinal_encode, one_hot_encode
# from s04_1_feature_engineering import build_polynomials, transform_label, boxcox_on_variables
from s05_2_feature_engineering import build_polynomials, transform_label, treat_skewness
from params import ProjectParameters


def get_model_params(classifier, scoring):
    if scoring == 'neg_mean_squared_error':
        classifier.best_score_ = -classifier.best_score_
    return classifier.best_params_, classifier.best_score_

### model training steps ###
def timer(t_init):
    t = datetime.now() - t_init
    t = t.seconds + t.microseconds*10**-6
    return t

def measure_prediction_time(classifier, df, n_iterations = 10):
    start_time = datetime.now()
    classifier.predict(df.iloc[:n_iterations, :])
    prediction_time = round(timer(start_time)/n_iterations,9)
    return prediction_time

def apply_ml_model(X_train_set, y_train_set, cols, model, parameters, scoring=None, 
                   do_build_polynomals=False, do_transform_label=False, 
                   do_treat_skewness=False,
                   imputation=None, scaler=None,
                   smote=False, testing=False, print_pipeline=False):
    '''
    default is: (X_train_set, y_train_set, cols, model, parameters, scoring=None, imputation=None, scaler=None, smote=False, print_pipeline=False)
    '''
    start_time = datetime.now()
    print('test type:', testing)

    # if encoding == 'one-hot':
    #     set_name = 'X_train_oh'
    #     if treat_collinearity:
    #         reports = os.path.join('..', 'data', '06_reporting')
    #         collinear_vars = pd.read_csv(os.path.join(reports, 'collinear_vars.csv')).iloc[:, 0].to_list()
    #         for c in collinear_vars:
    #             cols.remove(c)
    # elif encoding == 'ordinal':
    #     set_name = 'X_train'

    # X_train_set = data_dict[set_name]
    # print(cols, X_train_set.columns.to_list())
    X_train_set = X_train_set[cols]
    if do_build_polynomals: 
        X_train_set = build_polynomials(X_train_set, ProjectParameters().numerical_cols, testing=testing)
#     if do_transform_label:
#         y_train_set = transform_label(y_train_set, do_transform_label, testing=testing)
    if do_treat_skewness:
        X_train_set = treat_skewness(X_train_set, set_name, testing=testing)
    if testing:
        print(X_train_set.shape, X_train_set.shape)
        
    #define steps, defined classes were commented but were still built
    steps = []
    if imputation: steps.append(('imputation', imputation))
    if scaler: steps.append(('scaler', scaler))
    if smote: steps.append(('sampling', SMOTE(random_state=42)))
        
    steps.append(('model', model))
    
    #default pipeline uses sklearn, whereas imb_pipeline refers to the one from imblearn package for smote implementation
    if smote: 
        pipeline = imb_pipeline(steps)
    else: 
        pipeline = Pipeline(steps)
        
    # Instantiate the GridSearchCV object: cv
    clf_cv = GridSearchCV(pipeline, parameters, scoring=scoring, cv=5)

    # Fit to the training set
    clf_cv.fit(X_train_set, y_train_set)

    # Compute metrics    
    train_time = round(timer(start_time),9)
    prediction_time = measure_prediction_time(clf_cv, X_train_set)
    
    return clf_cv, train_time, prediction_time
    
def save_model_parameters(folder, filename, alg):
    file = filename+'.txt'
    report_file = os.path.join(folder, file)
    with open(report_file, 'w+', encoding='utf-8') as f: 
        text = f.write(str(alg))

def save_model_metrics(folder, filename, outputs):
    file = filename+'.json'
    output_file = os.path.join(folder, file)
    json_txt = json.dumps(outputs, indent=4)
    with open(output_file, 'w') as file:
        file.write(json_txt)


### model selection steps ###
import operator
def plot_scores(results):
    tup_results = sorted(results.items(), key=operator.itemgetter(1))

    N = len(results)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.40       # the width of the bars

    fig = plt.figure(figsize=(8,2))
    ax = fig.add_subplot(111)
    rects = ax.bar(ind+0.5, list(zip(*tup_results))[1], width,)
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 
                1.005*height, 
                '{0:.4f}'.format(height), 
                ha='center', 
                va='bottom',)

    ax.set_ylabel('Score')
    ax.set_xlabel('ML model')
    ax.set_ylim(ymin=0.0,ymax = 1)
    ax.set_title("Score Comparison")
    ax.set_xticks(ind + width/2.)
    ax.set_xticklabels(list(zip(*tup_results))[0], rotation=22)

    plt.show()