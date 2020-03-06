lgb_parameters = {'num_leaves': 321,
                  'min_child_weight': 0.034,
                  'feature_fraction': 0.379,
                  'bagging_fraction': 0.418,
                  'min_data_in_leaf': 106,
                  'objective': 'regression',
                  'max_depth': -1,
                  'learning_rate': 0.08,
                  "boosting_type": "gbdt",
                  "bagging_seed": 11,
                  "metric": 'rmse',
                  "verbosity": -1,
                  'reg_alpha': 1,  # 0.3899,
                  'reg_lambda': 2,  # 0.648,
                  'random_state': 47,
                  'n_jobs': -1
                  }

xgb_parameters = {'colsample_bytree': 0.825,
                  'max_depth': 4,
                  'learning_rate': 0.07,  # 0.05
                  'subsample': 0.81,
                  'objective': 'reg:squarederror',
                  'eval_metric': 'rmse',
                  'n_estimators': 22222,
                  'silent': False,
                  'verbosity': 2,
                  'n_jobs': -1,
                  'tree_method': 'gpu_hist',
                  'min_child_samples': 6,  # 8
                  'reg_alpha': 1,
                  'reg_lambda': 2,
                  'min_child_weight': 4,
                  }

cat_parameters = {'task_type': 'CPU',
                  'iterations': 30000,
                  'learning_rate': 0.08,
                  'random_seed': 7,
                  'depth': 4,
                  'eval_metric': 'RMSE',
                  'subsample': 0.81,
                  'reg_lambda': 2,
                  }

other_parameters = {'lgb_flag': True,
                    'xgb_flag': True,
                    'cat_flag': True,
                    'shuffle': True,
                    'target_columns': ['open_channels'],
                    'splits': 5,
                    'random_state': 42,
                    'num_boost_round': 30000,
                    'early_stopping_rounds': 250,
                    'verbose_eval': 100,
                    'group_by': 'batch'
                    }

importance_factor = {'lgb_factor': 0.4,
                     'xgb_factor': 0.4,
                     'cat_factor': 0.2
                     }

non_feature_columns = ['time', 'open_channels', 'batch', 'batch_index', 'batch_slices', 'batch_slices2']
