import os
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import dispatcher

warnings.simplefilter(action='ignore', category=FutureWarning)


class FeatImportance:
    def __init__(self):
        self.params = dispatcher.other_parameters
        self.target_columns = self.params['target_columns']
        self.train = joblib.load('../inputs/feature_engineered/train.pkl')
        self.feature_columns = [c for c in self.train if c not in dispatcher.non_feature_columns]
        self.num_of_feats = len(self.feature_columns)
        self.lgb_flag = self.params['lgb_flag']
        self.xgb_flag = self.params['xgb_flag']
        self.cat_flag = self.params['cat_flag']
        if not os.path.exists('../outputs/extras'):
            os.mkdir('../outputs/extras')

    def lgb(self):
        model = joblib.load('../outputs/lgb_model/model1.pkl')
        feature_imp = pd.DataFrame({'Value': model.feature_importance(), 'Feature': self.feature_columns})
        plt.figure(figsize=(40, self.num_of_feats))
        sns.set(font_scale=5)
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                         ascending=False)[0:self.num_of_feats])
        plt.title('LightGBM Features (avg over folds)')
        plt.tight_layout()
        print("saving the feat importance image file for lgb model")
        plt.savefig('../outputs/extras/lgb_model_model1.png')

    def xgb(self):
        model = joblib.load('../outputs/xgb_model/model1.pkl')
        feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': self.feature_columns})
        plt.figure(figsize=(40, self.num_of_feats))
        sns.set(font_scale=5)
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                         ascending=False)[0:self.num_of_feats])
        plt.title('LightGBM Features (avg over folds)')
        plt.tight_layout()
        print("saving the feat importance image file for xgb model")
        plt.savefig('../outputs/extras/xgb_model_model1.png')

    def cat(self):
        model = joblib.load('../outputs/cat_model/model1.pkl')
        feature_imp = pd.DataFrame({'Value': model.get_feature_importance(), 'Feature': self.feature_columns})
        plt.figure(figsize=(40, self.num_of_feats))
        sns.set(font_scale=5)
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                         ascending=False)[0:self.num_of_feats])
        plt.title('LightGBM Features (avg over folds)')
        plt.tight_layout()
        print("saving the feat importance image file for cat model")
        plt.savefig('../outputs/extras/cat_model_model1.png')

    def main(self):
        if self.lgb_flag:
            self.lgb()
        if self.xgb_flag:
            self.xgb()
        if self.cat_flag:
            self.cat()


if __name__ == '__main__':
    importance_object = FeatImportance()
    importance_object.main()