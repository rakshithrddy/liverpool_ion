import gc
import joblib
from . import dispatcher
from . import model


class Trainer:
    def __init__(self):
        self.train = joblib.load('../inputs/feature_engineered/train.pkl')
        self.feature_columns = [c for c in self.train if c not in dispatcher.non_feature_columns]
        self.lgb_parameters = dispatcher.lgb_parameters
        self.xgb_parameters = dispatcher.xgb_parameters
        self.cat_parameters = dispatcher.cat_parameters
        self.params = dispatcher.other_parameters
        self.lgb_flag = self.params['lgb_flag']
        self.xgb_flag = self.params['xgb_flag']
        self.cat_flag = self.params['cat_flag']
        self.split = self.params['splits']
        self.target_columns = self.params['target_columns']
        self.random_state = self.params['random_state']
        self.num_boost_round = self.params['num_boost_round']
        self.early_stopping_rounds = self.params['early_stopping_rounds']
        self.verbose_eval = self.params['verbose_eval']
        self.group_by = self.params['group_by']
        self.shuffle = self.params['shuffle']
        self.model_object = model.Models(train=self.train,
                                         feature_columns=self.feature_columns,
                                         target_columns=self.target_columns,
                                         splits=self.split,
                                         random_state=self.random_state,
                                         num_boost_round=self.num_boost_round,
                                         early_stopping_rounds=self.early_stopping_rounds,
                                         verbose_eval=self.verbose_eval,
                                         group_by=self.group_by,
                                         shuffle=self.shuffle)

    def main(self):
        if self.lgb_flag:
            print("Training lgb model")
            self.model_object.lgb(params=self.lgb_parameters)
        if self.xgb_flag:
            print('Training xgb model')
            self.model_object.xgb(params=self.xgb_parameters)
        if self.cat_flag:
            print("Training cat model")
            self.model_object.cat(params=self.cat_parameters)
        gc.collect()


if __name__ == '__main__':
    train_object = Trainer()
    train_object.main()
