import xgboost as xgb
from sklearn.model_selection import KFold
from catboost import Pool, CatBoostRegressor
import lightgbm as lgb
from . import metrics
import joblib
import gc
import os


class Models:
    def __init__(self, train,
                 feature_columns,
                 target_columns,
                 splits,
                 random_state,
                 num_boost_round,
                 early_stopping_rounds,
                 verbose_eval,
                 group_by,
                 shuffle
                 ):
        self.cols = feature_columns
        self.TARGET = target_columns
        self.split = splits
        self.random_state = random_state
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval
        self.group_by = group_by
        self.X = train[self.cols]
        self.y = train[self.TARGET].values
        self.groups = train[self.group_by]
        self.kfold = KFold(n_splits=self.split, shuffle=shuffle, random_state=self.random_state)

    def lgb(self, params):
        if not os.path.exists('../outputs/lgb_model'):
            os.mkdir('../outputs/lgb_model')
        fold = 1
        for tr_idx, val_idx in self.kfold.split(self.X, self.y, groups=self.groups):
            print(f'====== Fold {fold:0.0f} of {self.split} ======')
            X_tr, X_val = self.X.iloc[tr_idx], self.X.iloc[val_idx]
            y_tr, y_val = self.y[tr_idx], self.y[val_idx]
            train_set = lgb.Dataset(X_tr, y_tr)
            val_set = lgb.Dataset(X_val, y_val)
            del X_tr, X_val, y_tr, y_val
            model = lgb.train(params=params,
                              train_set=train_set,
                              num_boost_round=self.num_boost_round,
                              early_stopping_rounds=self.early_stopping_rounds,
                              valid_sets=[train_set, val_set],
                              verbose_eval=self.verbose_eval,
                              feval=metrics.MacroF1Metric)
            print(f"dumping model{self.split} for lgb")
            joblib.dump(model, f"../outputs/lgb_model/model{self.split}.pkl")
            fold += 1
        gc.collect()
        return

    def xgb(self, params):
        if not os.path.exists('../outputs/xgb_model'):
            os.mkdir('../outputs/xgb_model')
        fold = 1
        for tr_idx, val_idx in self.kfold.split(self.X, self.y, groups=self.groups):
            print(f'====== Fold {fold:0.0f} of {self.split} ======')
            X_tr, X_val = self.X.iloc[tr_idx], self.X.iloc[val_idx]
            y_tr, y_val = self.y[tr_idx], self.y[val_idx]
            train_set = xgb.DMatrix(X_tr, y_tr)
            val_set = xgb.DMatrix(X_val, y_val)
            del X_tr, X_val, y_tr, y_val
            model = xgb.train(params,
                              train_set,
                              num_boost_round=self.num_boost_round,
                              early_stopping_rounds=self.early_stopping_rounds,
                              evals=[(train_set, 'train'), (val_set, 'val')],
                              verbose_eval=self.verbose_eval)
            print(f"Dumping model{self.split} for xgb")
            joblib.dump(model, f"../outputs/xgb_model/model{self.split}.pkl")
            fold += 1
        gc.collect()
        return

    def cat(self, params):
        if not os.path.exists('../outputs/cat_model'):
            os.mkdir('../outputs/cat_model')
        fold = 1
        for tr_idx, val_idx in self.kfold.split(self.X, self.y, groups=self.groups):
            print(f'====== Fold {fold:0.0f} of {self.split} ======')
            X_tr, X_val = self.X.iloc[tr_idx], self.X.iloc[val_idx]
            y_tr, y_val = self.y[tr_idx], self.y[val_idx]
            train_set = Pool(X_tr, y_tr)
            val_set = Pool(X_val, y_val)
            del X_tr, X_val, y_tr, y_val
            model = CatBoostRegressor(task_type=params['task_type'],
                                      iterations=params['iterations'],
                                      learning_rate=params['learning_rate'],
                                      random_seed=params['random_seed'],
                                      depth=params['depth'],
                                      eval_metric=params['eval_metric'],
                                      subsample=params['subsample'],
                                      reg_lambda=params['reg_lambda'],
                                      early_stopping_rounds=self.early_stopping_rounds)
            model.fit(train_set, eval_set=val_set, verbose=self.verbose_eval)
            print(f"dumping model{self.split} for cat")
            joblib.dump(model, f"../outputs/cat_model/model{self.split}.pkl")
            fold += 1
        gc.collect()
        return
