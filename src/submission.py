import joblib
import pandas as pd
from . import dispatcher
import numpy as np
import os
import gc


class Submission:
    def __init__(self):
        self.filename = 'submission.csv'
        self.params = dispatcher.other_parameters
        self.lgb_flag = self.params['lgb_flag']
        self.xgb_flag = self.params['xgb_flag']
        self.cat_flag = self.params['cat_flag']
        self.target_columns = self.params['target_columns']
        self.importance_factor = dispatcher.importance_factor
        self.lgb_factor = self.importance_factor['lgb_factor']
        self.xgb_factor = self.importance_factor['xgb_factor']
        self.cat_factor = self.importance_factor['cat_factor']
        self.preds_lgb = self.preds_xgb = self.preds_cat = 0
        if not os.path.exists('../outputs'):
            os.mkdir('../outputs')

    def submit(self):
        try:
            if self.lgb_flag:
                self.preds_lgb = joblib.load('../outputs/predictions/pred_lgb.pkl')
            if self.xgb_flag:
                self.preds_xgb = joblib.load('../outputs/predictions/pred_xgb.pkl')
            if self.cat_flag:
                self.preds_cat = joblib.load('../outputs/predictions/pred_cat.pkl')
        except Exception as e:
            raise e

        if self.lgb_factor + self.cat_factor + self.xgb_factor == 1:
            final_preds = self.lgb_factor * self.preds_lgb + \
                          self.xgb_factor * self.preds_xgb + \
                          self.cat_factor * self.preds_cat
            submission = pd.read_csv("../inputs/sample_submission.csv")
            submission[[self.target_columns]] = np.round(np.clip(final_preds, 0, 10)).astype(int)
            print('dumping the final submissions into output')
            submission.to_csv('../outputs/submission.csv', index=False, float_format='%.4f')
        else:
            raise print('Factor rations are not equal to 1')


if __name__ == '__main__':
    submission_obj = Submission()
    submission_obj.submit()
    gc.collect()
