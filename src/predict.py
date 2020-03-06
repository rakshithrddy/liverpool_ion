import os
import numpy as np
import xgboost as xgb
import gc
from . import dispatcher
import joblib


class Predict:
    def __init__(self):
        self.params = dispatcher.other_parameters
        self.split = self.params['splits']
        self.X_test = joblib.load('../inputs/feature_engineered/test.pkl')
        self.predictions = None
        if not os.path.exists(f'../outputs/predictions'):
            os.mkdir("../outputs/predictions")

    def pred_lgb(self):
        try:
            fold = 1
            model_list = os.listdir('../outputs/lgb_model')
            for i in model_list:
                model = joblib.load(f'../outputs/lgb_model/{i}')
                print(f'====== Fold {fold:0.0f} of {self.split} ======')
                preds = model.predict(self.X_test, num_iteration=model.best_iteration).astype(np.float16)
                if fold == 1:
                    self.predictions = preds
                else:
                    self.predictions += preds
                fold += 1
            self.predictions /= self.split
            print('dumping predictions for model_lgb into outputs')
            joblib.dump(self.predictions, '../outputs/predictions/pred_lgb.pkl')
            gc.collect()
            return
        except Exception as e:
            raise e

    def pred_xgb(self):
        try:
            fold = 1
            model_list = os.listdir('../outputs/xgb_model')
            for i in model_list:
                model = joblib.load(f'../outputs/xgb_model/{i}')
                print(f'====== Fold {fold:0.0f} of {self.split} ======')
                preds = model.predict(xgb.DMatrix(self.X_test), ntree_limit=model.best_ntree_limit).astype(np.float16)
                if fold == 1:
                    self.predictions = preds
                else:
                    self.predictions += preds
                fold += 1
            self.predictions /= self.split
            print('dumping predictions for model_xgb into outputs')
            joblib.dump(self.predictions, '../outputs/predictions/pred_xgb.pkl')
            gc.collect()
            return
        except Exception as e:
            raise e

    def pred_cat(self):
        try:
            fold = 1
            model_list = os.listdir('../outputs/cat_model')
            for i in model_list:
                model = joblib.load(f"../outputs/cat_model/{i}")
                print(f'====== Fold {fold:0.0f} of {self.split} ======')
                preds = model.predict(self.X_test).astype(np.float16)
                if fold == 1:
                    self.predictions = preds
                else:
                    self.predictions += preds
                fold += 1
            self.predictions /= self.split
            print('dumping predictions for model_cat into outputs')
            joblib.dump(self.predictions, '../outputs/predictions/pred_cat.pkl')
            gc.collect()
            return
        except Exception as e:
            raise e