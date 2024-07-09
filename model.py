import time
import pandas as pd
import numpy as np
from numpy import array
import pickle
import os
from os import path
import sklearn
from sklearn.metrics import roc_auc_score, make_scorer, recall_score, accuracy_score, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate


from hpsklearn import HyperoptEstimator, any_preprocessing
from hpsklearn import random_forest_classifier, gradient_boosting_classifier, svc, xgboost_classification, k_neighbors_classifier
from hyperopt import tpe, hp

#from hpsklearn import HyperoptEstimator, any_classifier
#from hpsklearn.components import gradient_boosting, svc, xgboost, k_neighbors_classifier
#from hyperopt import tpe

import argparse
from data_preprocessing import data_preprocessing

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='Caco2_permeability')
    parser.add_argument('--validation', action='store_true', default=False)
    parser.add_argument('--prediction', action='store_true', default=False)

    args  =parser.parse_args()
    
    db_names = os.listdir('./best_models/')
    db_names = [name[:-4] for name in db_names ]
    features = eval(open(f'./features.txt', 'r').read())

    if args.prediction :#w/o_bioclass
        print('prediction mode')
        test_data = pd.read_csv(args.file_name)
        test_data['bioclass']=1
        scaled_data = data_preprocessing(test_data)

        print('size of data set: ', len(test_data))
        for name in db_names:
            print(f'predictive model for {name}')
            learner_model = pickle.load(open(f'best_models/{name}.pkl', 'rb'))
            predicted = learner_model.predict(scaled_data[features].values)
            test_data[f'{name}']=learner_model.predict(scaled_data[features].values)
        test_data.to_csv('admet_output.csv', index=False)
        print('save output in admet_output.csv')

    elif args.validation : #w_bioclass
        print('validation mode')
        test_data = pd.read_csv(args.file_name)
        scaled_data = data_preprocessing(test_data)

        print('size of data set: ', len(test_data))
        for name in db_names:
            print(f'predictive model for {name}')
            learner_model = pickle.load(open(f'best_models/{name}.pkl', 'rb'))
            print(type(learner_model))

            predicted = learner_model.predict(scaled_data[features].values)
            tn, fp, fn, tp = confusion_matrix(scaled_data['bioclass'], predicted).ravel()
            print('Se:', tp/(tp+fn))
            print('Sp:', tn/(tn+fp))
            print('acc:', (tp+tn)/(tp+fp+tn+fn))
            print('mcc:', ((tp*tn)-(fp*fn))/((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5 )

            try:
                predicted_proba = learner_model.predict_proba(scaled_data[features].values)[:,1]
            except:
                predicted_proba = learner_model.decision_function(scaled_data[features].values)
            print('auc:', roc_auc_score(scaled_data['bioclass'], predicted_proba) )
    else:
        print('choose mode [validation, prediction]')
