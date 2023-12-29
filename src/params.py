# import json
# import numpy as np
# import pandas as pd
# import os
# import sys
# from datetime import datetime

class ProjectParameters:

    def __init__(self):
        # self.target_type = 'binary'
        # if self.target_type == 'regression':
        #     self.scoring = 'neg_mean_squared_error'
        # elif self.target_type == 'binary': # some options are: accuracy, f1, recall, neg_log_loss ## https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        #     self.scoring = 'f1'

        self.sandbox_mode = True

    
        self.numerical_cols = ['preset_1', 'preset_2', 'temperature', 'pressure', 'vibrationx', 'vibrationy', 'vibrationz', 'frequency', 'preset_comp']
        # self.collinear_vars = ['white_frac', 'male_frac', 'latitude', 'longitude', 'young']

