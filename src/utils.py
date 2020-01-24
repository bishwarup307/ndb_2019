'''
 Created on Fri Dec 27 2019
 __author__: bishwarup
'''

from __future__ import division, print_function
import os
import re
import time
from functools import partial, wraps
import json
import itertools
import operator
import numpy as np
import pandas as pd
import joblib
import copy
import warnings
from numba import jit, autojit
import pickle
from collections import Counter, defaultdict, OrderedDict
from tqdm import tqdm_notebook, tnrange
from scipy.optimize import fmin_powell, minimize
from ast import literal_eval
from sklearn.utils.multiclass import unique_labels
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import cohen_kappa_score, mean_squared_error, confusion_matrix, classification_report, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool, CatBoostRegressor

from .config import *
from .game_stats import *

global KAGGLE
KAGGLE = False
warnings.filterwarnings('ignore')

def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print(f"total elapsed: {int((end-start) // 60)}m {(end-start) % 60:.0f}s...")
        return result
    return wrapper

def print_shape(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        print(f"output shape: {result.shape}")
        return result
    return wrapper

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
#@print_shape
@timing
def read_datasets():
    #print("loading datasets...")
    if KAGGLE:
        input_dir = '../input/data-science-bowl-2019'
        train = pd.read_csv(os.path.join(input_dir, 'train.csv'), parse_dates = ['timestamp'])
        test = pd.read_csv(os.path.join(input_dir, 'test.csv'), parse_dates = ['timestamp'])
        labels = pd.read_csv(os.path.join(input_dir, 'train_labels.csv'))
            
    else:
        pkl_exists = np.logical_and(os.path.isfile(os.path.join(ROOT_DIR, 'input', 'train.pkl')), \
                                    os.path.isfile(os.path.join(ROOT_DIR, 'input', 'test.pkl')))
        print(f'pkl_exists: {pkl_exists}')

        TRAIN_FILE = os.path.join(ROOT_DIR, 'input', 'train.pkl') if pkl_exists else  os.path.join(ROOT_DIR, 'input', 'train.csv')
        TEST_FILE = os.path.join(ROOT_DIR, 'input', 'test.pkl') if pkl_exists else  os.path.join(ROOT_DIR, 'input', 'test.csv')        
        
        train = pd.read_pickle(TRAIN_FILE) if pkl_exists else pd.read_csv(TRAIN_FILE, parse_dates = ['timestamp'])
        test = pd.read_pickle(TEST_FILE) if pkl_exists else pd.read_csv(TEST_FILE, parse_dates = ['timestamp'])
        labels = pd.read_csv(os.path.join(ROOT_DIR, 'input', 'train_labels.csv'))

        if not pkl_exists:
            train.to_pickle(TRAIN_FILE.replace('.csv', '.pkl'))
            test.to_pickle(TEST_FILE.replace('.csv', '.pkl'))

    for df in [train, test]:
        df['game_time'] /= 1000
        grp = df.groupby('game_session')
        df["gs_starttime"] = grp["timestamp"].transform("min")
        df["gs_endtime"] = grp["timestamp"].transform("max")
        df["gs_duration"] = grp["game_time"].transform("max")    
    return train, test, labels
        
def check_correct(x):
    try:
        out = str(literal_eval(x.replace("true", "True").replace("false", "False")).get('correct', np.nan))[:1]
    except ValueError:
        out = "No_Results"
    return out

def keep_fn(row):
    if row["event_code"] == 2000:
        return False
    if np.logical_and(row['title'] == 'Mushroom Sorter (Assessment)', row['event_code'] == 4100):
        return False
    if np.logical_and(row['title'] == 'Bird Measurer (Assessment)', row['event_code'] == 4110):
        return False
    if np.logical_and(row['title'] == 'Cart Balancer (Assessment)', row['event_code'] == 4100):
        return False
    if np.logical_and(row['title'] == 'Cauldron Filler (Assessment)', row['event_code'] == 4100):
        return False
    if np.logical_and(row['title'] == 'Chest Sorter (Assessment)', row['event_code'] == 4100):
        return False
    return True

def get_accuracy_group(row):
    if np.logical_and(row["T"] == 0, row["F"] == 0):
        return -1
    if row["T"] == 0:
        return 0
    if row["T"] > 0:
        if row["F"] == 0:
            return 3
        if row["F"] == 1:
            return 2
        return 1

def get_accuracy(row):
    if np.logical_and(row["T"] == 0, row["F"] == 0):
        return -1
    elif row["T"] == 0:
        return 0
    else:
        return row["T"] / (row["T"] + row["F"])

def map_alphabet(x):
    s = alphabet_map_treetopcity.get(x, 'Z')
    if s != 'Z':
        return s
    s = alphabet_map_magmapeak.get(x, 'Z')
    if s != 'Z':
        return s
    s = alphabet_map_crystalcaves.get(x, 'Z')
    return s

def clean_seq(x):
    x = x.lstrip("Z")
    x = ''.join([x for x, _ in itertools.groupby(x)])
    return x

@jit
def qwk(a1, a2):
    """
    Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/114133#latest-660168

    :param a1:
    :param a2:
    :param max_rat:
    :return:
    """
    max_rat = 3
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o +=  (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e

def QWK_XGB(y_pred: np.ndarray, dtrain: xgb.DMatrix):
    y = dtrain.get_label()
    y_pred[y_pred <= 1.12232214] = 0
    y_pred[np.where(np.logical_and(y_pred > 1.12232214, y_pred <= 1.73925866))] = 1
    y_pred[np.where(np.logical_and(y_pred > 1.73925866, y_pred <= 2.22506454))] = 2
    y_pred[y_pred > 2.22506454] = 3
    return 'Kappa', qwk(y, y_pred)

def eval_qwk_lgb_regr(y_pred, train_data):
    """
    Fast cappa eval function for lgb.
    """
    y_true = train_data.get_label()
    y_pred[y_pred <= 1.12232214] = 0
    y_pred[np.where(np.logical_and(y_pred > 1.12232214, y_pred <= 1.73925866))] = 1
    y_pred[np.where(np.logical_and(y_pred > 1.73925866, y_pred <= 2.22506454))] = 2
    y_pred[y_pred > 2.22506454] = 3
    return 'kappa', qwk(y_true, y_pred), True
    
def QWK(x, yhat, y):
    y = np.array(y)
    y = y.astype(int)
    x = sorted(x)
    yhat = np.digitize(yhat, bins = x)
    return -cohen_kappa_score(yhat, y, weights="quadratic")

class QWKMetric(object):
    def get_final_error(self, error, weight):
        return error 

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])
        approx = approxes[0]

        approx = np.digitize(approx, bins = [1.12232214, 1.73925866, 2.22506454])
        #qwk = cohen_kappa_score(np.uint8(approx), np.uint8(target), weights='quadratic')
        WK = qwk(np.uint8(target), np.uint8(approx))
        return WK, 1.

def get_grouped_splits(df, n_splits = 5, out_path = 
               os.path.abspath(''), random_seed = 2019, group_col = "installation_id"):
    out_path = os.path.join(out_path, f'grouped_splits_{n_splits}.json')
    if os.path.isfile(out_path):
        print("loading saved splits...")
        with open(out_path, 'r') as f:
            folds = json.load(f)
        folds = {int(k): v for k, v in folds.items()}
        return folds
    df = df[["id", "accuracy_group", group_col]].copy()
    folds = {}
    gkf = GroupKFold(n_splits=n_splits)
    for i, (_, in_val) in enumerate(gkf.split(df.id, df.accuracy_group, df.installation_id)):
        val_ids = df[group_col][in_val].unique().tolist()
        folds[i] = val_ids
    print("saving folds to disk...")
    with open(out_path, 'w') as f:
        json.dump(folds, f)
    return folds

def RandomGroupKFold_split(groups, n_splits = 5, seed=None):
    groups = pd.Series(groups)
    unique = np.unique(groups)
    np.random.RandomState(seed).shuffle(unique)
    return dict(zip(np.arange(n_splits), [x.tolist() for x in np.array_split(unique, n_splits)]))
########


########
class GBDTTrainer(object):
    def __init__(self, model_prefix, features, params, io_dir, submission_dir, eval_fn = None,
                id_col = "id", model_type = "LGB", save_predictions = True, target_name = "accuracy_group", verbose_freq = 100,
                save_optimized_cutoffs = True, initial_guess = [1.14, 1.652, 2.17], apply_cutoffs_foldwise = False,
                max_rounds = 30_000, early_stop_rounds = 100, categoricals = None, feature_importance = False,
                weights_col = None, target_encode = None, global_response_mean = None,
                categorical_indices = None, plot = False, feval = None, pred_transform = None, splits  = None,
                folds = None, fobj = None, apply_cutoffs_to_eval = False, return_importance = False, adverserial = False, 
                adv_params = None, truncate_eval = False, write_submission = False, optimize_thresholds = False):
        if np.logical_and(folds is None, splits is None):
            raise ValueError("you must specify either one of `folds` or `splits`")
            
        if save_optimized_cutoffs:
            if eval_fn is None:
                raise ValueError("you must specify `eval_fn` when `save_optimized_cutoffs = True`")
        
        self.adverserial = adverserial
        self.feats_df = pd.DataFrame({"Features": features})
        self.adv_params = adv_params
        self.save_optimized_cutoffs = save_optimized_cutoffs
        self.eval_fn = eval_fn
        self.initial_guess = initial_guess
        self.optimized_cutoffs = []
        self.apply_cutoffs_foldwise = apply_cutoffs_foldwise
        self.apply_cutoffs_to_eval = apply_cutoffs_to_eval
        self.optimize_thresholds = optimize_thresholds
        self.optimized_thresholds = None
        self.overall_optimized_score = None
        self.model_prefix = model_prefix
        self.return_importance = return_importance
        self.truncate_eval = truncate_eval
        self.n_fold = len(splits) if splits is not None else len(folds)
        self.folds = folds
        self.fobj = fobj
        self.splits = splits
        self.params = params
        self.plot = plot
        self.io_dir = io_dir
        self.submission_dir = submission_dir
        self.save_predictions = save_predictions
        self.id_col = id_col
        self.target_name = target_name
        self.features = features
        self.categoricals = categoricals
        self.y_test = None
        self.eval_cuts = []
        self.max_rounds = max_rounds
        self.feval = feval
        self.weights_col = weights_col
        self.early_stop_rounds = early_stop_rounds
        self.verbose_freq = verbose_freq
        self.feature_importance = feature_importance
        self.target_encode = target_encode
        self.global_response_mean = global_response_mean
        self.gts = []
        self.oof_preds = []
        self.test_preds = None
        self.fold = 0
        self.pred_transform = pred_transform
        self.write_submission = write_submission
        self.cv =[]
        
        if model_type not in ['XGB', 'LGB', 'CAT']:
            raise ValueError('`model_type` must be either `LGB` or `XGB` or `CAT`')
        self.model_type = model_type
        self.categorical_indices = categorical_indices
#         if np.logical_and(self.feval is not None, self.pred_transform is None):
#             warnings.warn("`feval` is specified but no tranform to predictions will be applied")
        self.base_dir = os.path.join(self.io_dir, self.model_prefix)
        if os.path.isdir(self.base_dir):
            shutil.rmtree(self.base_dir)
        os.makedirs(self.base_dir)
        
    def _write_summary_to_file(self):
        pass
    
    def _get_model(self):
        pass

    def _generate_submission(self, df):
        df = df.rename(columns = {self.model_prefix: self.target_name, 'id' : 'installation_id'})
        out_path = os.path.join(self.submission_dir, self.model_prefix + ".csv")
        df.to_csv(out_path, index=False)
        
    def _train_LGB(self, X_train, y_train, X_val, y_val, w_train = None):
        if w_train is not None:
            dtrain = lgb.Dataset(X_train, y_train, feature_name=self.features, categorical_feature=self.categoricals,
                                weight = w_train)
        else:
            dtrain = lgb.Dataset(X_train, y_train, feature_name=self.features, categorical_feature=self.categoricals)
        
        dval = lgb.Dataset(X_val, y_val, reference=dtrain, feature_name=self.features, categorical_feature=self.categoricals,
                          weight = np.ones(len(y_val)))
        
        if self.feval is not None:
            bst = lgb.train(
                params=self.params,
                feval = self.feval,
                train_set=dtrain, 
                valid_sets=[dtrain, dval],
                valid_names=['train', 'eval'],
                num_boost_round=self.max_rounds,
                verbose_eval=self.verbose_freq,
                early_stopping_rounds=self.early_stop_rounds
            )
        else:
            bst = lgb.train(
                params=self.params,
                fobj = self.fobj,
                train_set=dtrain, 
                valid_sets=[dtrain, dval],
                valid_names=['train', 'eval'],
                num_boost_round=self.max_rounds,
                verbose_eval=self.verbose_freq,
                early_stopping_rounds=self.early_stop_rounds
            )
        #print(bst.best_score['eval'])
        if self.feval is not None:
            score_, iter_ = bst.best_score['eval']['kappa'], bst.best_iteration
        else:
            score_, iter_ = bst.best_score['eval']['rmse'], bst.best_iteration
        return bst, score_, iter_
    
    def _train_XGB(self, X_train, y_train, X_val, y_val):
        dtrain = xgb.DMatrix(data=X_train, label=y_train, missing=np.nan)
        dval = xgb.DMatrix(data = X_val, label=y_val, missing=np.nan)
        if self.feval is not None:
            bst = xgb.train(
                params=self.params,
                dtrain=dtrain,
                obj = self.fobj,
                feval = self.feval,
                maximize = True,
                num_boost_round=self.max_rounds,
                early_stopping_rounds=self.early_stop_rounds,
                evals=[(dtrain, 'train'), (dval, 'eval')],
                verbose_eval=200
            )
        else:
            bst = xgb.train(
                params=self.params,
                dtrain=dtrain,
                obj = self.fobj,
                num_boost_round=self.max_rounds,
                early_stopping_rounds=self.early_stop_rounds,
                evals=[(dtrain, 'train'), (dval, 'eval')],
                verbose_eval=200)
        score_, iter_ = bst.best_score, bst.best_iteration
        return bst, score_, iter_
    
    def _train_CAT(self, X_train, y_train, X_val, y_val):
        X_train.fillna(-999, inplace = True)
        X_val.fillna(-999, inplace = True)
        bst = CatBoostRegressor(**self.params)
        categorical_features_indices = np.where(X_train.dtypes != np.float)[0] if \
                                        self.categorical_indices is None else self.categorical_indices
        train_pool, eval_pool = Pool(X_train, y_train, cat_features=categorical_features_indices), \
                                Pool(X_val, y_val, cat_features = categorical_features_indices)
        bst.fit(train_pool, eval_set=eval_pool, plot = self.plot)
        score_ = mean_squared_error(y_val, bst.predict(X_val))
        iter_ = bst.tree_count_
        return bst, score_, iter_
    
    def _predict(self, bst, X, iter_):
        if self.model_type == 'LGB':
            pred = bst.predict(X, num_iteration = iter_)
        elif self.model_type == 'XGB':
            if not isinstance(X, xgb.DMatrix):
                X = xgb.DMatrix(X, missing = np.nan)
            pred = bst.predict(X, ntree_limit = iter_)
        elif self.model_type == 'CAT':
            X.fillna(-999, inplace = True)
            pred = bst.predict(X)
        else:
            raise NotImplementedError
        if self.pred_transform:
            pred = self.pred_transform(pred)
        return pred
    
    def _truncate_eval(self, df):
        full_len = len(df)
        df.sort_values(["installation_id", "timestamp"], inplace = True)
        df["ass_sno"] = df.groupby("installation_id").cumcount()
        assessment_lens = df.installation_id.value_counts().to_dict()
        random_cuts = {k: np.random.choice(np.arange(v)) for k, v in assessment_lens.items()}
        self.eval_cuts.append(random_cuts)
        df['cuts'] = df['installation_id'].map(random_cuts)
        df = df.query('ass_sno == cuts')
        df.drop(['ass_sno', 'cuts'], axis = 1, inplace = True)
        print(f"truncated {full_len - len(df)} records...")
        return df

    def train_one_fold(self, train_df, eval_df, test_df):
        if np.logical_and(self.weights_col is not None, self.weights_col in train_df.columns):
            weights = train_df[self.weights_col].values
        else:
            weights = None

        if self.truncate_eval:
            eval_df = self._truncate_eval(eval_df)

        X_train, y_train = train_df[self.features], train_df[self.target_name].values
        X_val, y_val = eval_df[self.features], eval_df[self.target_name].values
        self.gts.extend(y_val.tolist())
        ### adverserial validation ###
        if self.adverserial:
            print("starting adverserial validation...")
            y_adv_train = np.zeros_like(y_train)
            y_adv_val = np.ones_like(y_val)
            
            X_adv = pd.concat([X_train, X_val], ignore_index=True)
            y_adv = np.concatenate([y_adv_train, y_adv_val])
            
            skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
            adv_auc = []
            if self.adv_params is None:
                self.adv_params  = {
                            "objective": "binary",
                            "metric" : "auc",
                            "learning_rate": 0.03,
                            "num_leaves": 56,
                            "max_depth" : -1,
                            "feature_fraction": 0.7,
                            "verbosity": 0,
                            "subsample": 1.,
                            "num_threads" : -1,
                                }
            for _, (itr, icv) in enumerate(skf.split(X_adv, y_adv)):
                X_tr, y_tr = X_adv.iloc[itr], y_adv[itr]
                X_ev, y_ev = X_adv.iloc[icv], y_adv[icv]
                train_dataset = lgb.Dataset(X_tr, y_tr)
                ev_dataset = lgb.Dataset(X_ev, y_ev, reference=train_dataset)
                
                clf = lgb.train(params=self.adv_params, 
                                train_set=train_dataset, 
                                num_boost_round=2000, 
                                valid_sets = [train_dataset, ev_dataset], 
                                valid_names=['train', 'eval'], 
                                verbose_eval=0, 
                                early_stopping_rounds=200)
                adv_auc.append(clf.best_score['eval']['auc'])
                gain = clf.feature_importance(importance_type='gain')
                
                imp_df = pd.DataFrame({"Features": self.features, f"gain_{str(self.fold)}": gain})
                self.feats_df = self.feats_df.merge(imp_df, on = "Features", how = "left")

            print(f"adverserial validation auc: {np.mean(adv_auc):.4f}")

        if self.model_type == 'LGB':
            bst, score_, iter_ = self._train_LGB(X_train, y_train, X_val, y_val, weights)
        elif self.model_type == 'XGB':
            bst, score_, iter_ = self._train_XGB(X_train, y_train, X_val, y_val)
        elif self.model_type == 'CAT':
            bst, score_, iter_ = self._train_CAT(X_train, y_train, X_val, y_val)
        else:
            raise NotImplementedError
            
        print(f"best_score : {score_:.6f}")
        
        if self.save_predictions:
            valid_preds = self._predict(bst, X_val, iter_)
            prediction_df = eval_df[["installation_id", "game_session"]].reset_index(drop = True)
            prediction_df[self.model_prefix] = valid_preds
            self.oof_preds.append(prediction_df)
            
        if self.save_optimized_cutoffs:
            y, yhat = y_val, valid_preds
            sol = minimize(self.eval_fn, self.initial_guess, args= (yhat, y, ), method = 'Nelder-Mead', options = {"disp" : False, "xtol" : 1e-8})
            for i in range(5):
                sol = minimize(self.eval_fn, sol.x, args= (yhat, y, ), method = 'Nelder-Mead', options = {"disp" : False, "xtol" : 1e-8})
            self.optimized_cutoffs.append(sol.x.tolist())
            self.cv.append(-sol.fun)
            print(f"max QWK: {-sol.fun}")
        
        if test_df is not None:
            test_preds = self._predict(bst, test_df[self.features], iter_)
            if self.apply_cutoffs_foldwise:
                test_preds = np.digitize(test_preds, bins = self.optimized_cutoffs[-1])
            self.test_preds += test_preds

        if np.logical_and(self.feature_importance, self.fold == 4):
            if self.model_type == 'LGB':
                print("calculating feature importance...")
                fimp_gain = bst.feature_importance(importance_type='gain')
                fimp_split = bst.feature_importance(importance_type='split')
                df = pd.DataFrame({"features": self.features, "gain": fimp_gain, "split" : fimp_split})
                df.sort_values("gain", ascending = False, inplace= True)
                print(df.head())
                print("saving feature importance...")
                df.to_csv(os.path.join(self.base_dir, "feature_importance.csv"), index = False)
                self.importance_df = df
            if self.model_type == 'CAT':
                feature_score = pd.DataFrame(list(zip(X_train.dtypes.index, bst.get_feature_importance(Pool(
                    X_train, label=y_train, cat_features=self.categorical_indices)))),columns=['Feature','Score'])
                feature_score = feature_score.sort_values(by='Score', ascending=False, inplace=False, kind='quicksort', na_position='last')
                print(feature_score.head(10))
                print("saving feature importance...")
                feature_score.to_csv(os.path.join(self.base_dir, "feature_importance.csv"), index = False)
                self.importance_df = feature_score
        # if self.apply_cutoffs_to_eval:
        #     prediction_dict = dict(zip(prediction_dict.keys(), np.digitize(np.array(list(prediction_dict.values())), bins = self.optimized_cutoffs[-1])))
        
        return score_, iter_
    
    def fit(self, train, test = None):
        cv_summary = {}
        best_iterations = []
        
        if test is not None:
            test_ids = test[self.id_col].values
            self.test_preds = np.zeros(test.shape[0])
        
        start_time = time.time()
        if self.splits is not None:
            for fold in range(self.n_fold):
                self.fold = fold
                print("*" * 18)
                print(f"training - fold: {fold + 1}")
                print("*" * 18)
                val_ids = self.splits.get(fold)
                train_df = train[~train["installation_id"].isin(val_ids)].copy()
                eval_df = train[train["installation_id"].isin(val_ids)].copy()
                print(f"train_df = {train_df.shape}, eval_df = {eval_df.shape}")
                print(f"# installation_id in train: {train_df.installation_id.nunique()}")
                print(f"# installation_id in eval: {eval_df.installation_id.nunique()}")
                
                metric, best_iter = self.train_one_fold(train_df, eval_df, test)
                cv_summary[fold] = {'score' : metric, 'iter' : best_iter}
                best_iterations.append(best_iter)

        else:
            for fold, val_ids in self.folds.items():
                print("*" * 18)
                print(f"training - fold: {fold + 1}")
                print("*" * 18)
                self.fold = fold
                train_df = train[~train[self.id_col].isin(val_ids)].copy()
                eval_df = train[train[self.id_col].isin(val_ids)].copy()

                print(f"train_df = {train_df.shape}, eval_df = {eval_df.shape}")

                print(f"# installation_id in train: {train_df.installation_id.nunique()}")
                print(f"# installation_id in eval: {eval_df.installation_id.nunique()}")
            
                if self.target_encode is not None:
                    print(f"target encoding following cols: {self.target_encode}")
                    for col in self.target_encode:
                        if col in self.features:
                            if test is not None:
                                test_encoded = test.copy()
                            smry = train_df.groupby(col)[self.target_name].mean().reset_index().rename(columns = {
                                self.target_name : col + "_tec"
                            })
                            train_df = train_df.merge(smry, on = col, how = "left")
                            eval_df = eval_df.merge(smry, on = col, how = "left")
                            fill_value = self.global_response_mean if self.global_response_mean else train_df[self.target_name].mean()
                            eval_df[col].fillna(fill_value)
                            train_df.drop(col, axis = 1, inplace = True)
                            eval_df.drop(col, axis = 1, inplace = True)
                            colnames = [x.replace("_tec", "") for x in train_df.columns]
                            train_df.columns, eval_df.columns = colnames, colnames

                            test_encoded = test.merge(smry, on = col, how = "left")
                            test_encoded.drop(col, axis = 1, inplace = True)
                            colnames = [x.replace("_tec", "") for x in test_encoded.columns]
                            test_encoded.columns = colnames
                if self.target_encode is not None:
                    metric, best_iter = self.train_one_fold(train_df, eval_df, test_encoded)
                else:
                    metric, best_iter = self.train_one_fold(train_df, eval_df, test)

                cv_summary[fold] = {'score' : metric, 'iter' : best_iter}
                best_iterations.append(best_iter)

        if self.test_preds is not None:
            self.test_preds /= self.n_fold
        
        if self.optimize_thresholds:
            print("optimizing thresholds...")
            y_hat = pd.concat(self.oof_preds)[self.model_prefix].values
            sol = minimize(self.eval_fn, self.initial_guess, args= (y_hat, self.gts, ), method = 'Nelder-Mead', options = {"disp" : True, "xtol" : 1e-8})
            for i in range(5):
                sol = minimize(self.eval_fn, sol.x, args= (y_hat, self.gts, ), method = 'Nelder-Mead', options = {"disp" : True, "xtol" : 1e-8})
            self.optimized_thresholds = sol.x
            self.overall_optimized_score = -sol.fun
            print(f"optimized score: {self.overall_optimized_score:.4f}")
            if self.test_preds is not None:
                print("applying optimized thresholds to test predictions...")
                self.y_test = np.digitize(self.test_preds, bins = self.optimized_thresholds)
            
        if self.save_predictions:
            print("saving predictions to disk...")
            self.oof_preds = pd.concat(self.oof_preds)
            out_name = os.path.join(self.base_dir, "eval.csv")
            self.oof_preds.to_csv(out_name, index = False)
            
            if self.test_preds is not None:
                preds = self.y_test if self.optimize_thresholds else self.test_preds
                test_prediction_df = pd.DataFrame({self.id_col: test['installation_id'].values, self.model_prefix: preds})
                assert test.shape[0] == test_prediction_df.shape[0]
                out_name = os.path.join(self.base_dir, "test.csv")
                test_prediction_df.to_csv(out_name, index=False)
                if self.write_submission:
                    print("generating submission...")
                    self._generate_submission(test_prediction_df)
        print(color.BLUE + f"mean-cv QWK: {np.mean(self.cv) : .4f}" + color.END)
        print("saving summary...")
        cvs = []
        for _, v in cv_summary.items():
            cvs.append(v['score'])
        mean_cv                                 = np.mean(cvs)
        summary_dict                            = {}
        summary_dict['model_prefix']            = self.model_prefix
        summary_dict['model_type']              = self.model_type
        summary_dict['cv_summary']              = cv_summary
        summary_dict['cv']                      = cvs
        summary_dict['mean_cv']                 = mean_cv
        summary_dict['hyperparams']             = self.params
        summary_dict['features']                = self.features
        summary_dict['truncated_validations']   = self.eval_cuts
        #print(summary_dict)
        out_name = os.path.join(self.base_dir, 'summary.txt')
       
        with open(out_name, 'w') as f:
            f.write(str(summary_dict))

        print(f"best iteration : {np.mean(best_iterations):.2f}")
        print(f"training complete in {time.time() - start_time}")
        return self.oof_preds, test_prediction_df

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap= plt.cm.Blues):
    """
    Refer to: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, fontsize=25)
    plt.yticks(tick_marks, fontsize=25)
    plt.xlabel('Predicted label',fontsize=25)
    plt.ylabel('True label', fontsize=25)
    plt.title(title, fontsize=30)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size="5%", pad=0.15)
    cbar = ax.figure.colorbar(im, ax=ax, cax=cax)
    cbar.ax.tick_params(labelsize=20)
    
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
#            title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    fontsize=20,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def get_agg_map(df, include_present = False):
    assessment_game_sessions = df.query('type == "Assessment" & (event_code == 4100 | event_code == 4110)').\
                                game_session.unique()
    #print(f"# assessment sessions: {len(assessment_game_sessions)}")
    assessment_game_sessions = dict(zip(assessment_game_sessions, np.repeat(True, len(assessment_game_sessions))))

    gs_map = df.drop_duplicates(["game_session"])[["installation_id", "game_session", "timestamp"]].copy()
    gs_map["is_game_session"] = gs_map["game_session"].map(assessment_game_sessions)
    gs_map.is_game_session.fillna(False, inplace = True)
    gs_map.sort_values(["installation_id", "timestamp"], inplace = True)
    gs_map["session_aggregate"] = np.nan
    gs_map["session_aggregate"][gs_map["is_game_session"]] = gs_map["game_session"]
    gs_map.sort_values(["installation_id", "timestamp"], inplace=True)
    gs_map["session_aggregate"] = gs_map.groupby("installation_id")["session_aggregate"].bfill()
    if not include_present:
        gs_map["session_aggregate"] = gs_map.groupby("installation_id")["session_aggregate"].shift(-1)
    gs_map.sort_values(["installation_id", "timestamp"], inplace=True)
    gs_map = gs_map[["game_session", "session_aggregate"]]
    return gs_map

@print_shape
@timing
def get_session_features(df):
    df = copy.deepcopy(df)
    df.sort_values(["installation_id", "timestamp"], inplace=True)
    sessions = df.groupby(["installation_id", "game_session"])["timestamp"].agg(["min", "max"]).reset_index()
    sessions.sort_values(["installation_id", "min"], inplace=True)
    sessions["last_session_endtime"] = sessions.groupby("installation_id")["max"].\
                                                    transform(lambda x: x.shift(1))
    sessions["diff"] = (sessions["min"].dt.tz_localize(None) - \
                                    sessions["last_session_endtime"].dt.tz_localize(None)).dt.total_seconds()
    sessions["counter"] = sessions["diff"].map(lambda x: 1 if x > 300 else 0)
    sessions["session"] = sessions.groupby("installation_id")["counter"].cumsum()
    sessions.drop(["last_session_endtime", "counter", "diff"], axis =1, inplace=True)


    sessions["session_start_timestamp"] = sessions.groupby(["installation_id", "session"])["min"]\
                                        .transform("min")
    sessions["session_end_timestamp"] = sessions.groupby(["installation_id", "session"])["max"]\
                                        .transform("max")
    sessions["time_since_session_start"] = (sessions["min"] - sessions["session_start_timestamp"]).dt.total_seconds()
    sessions["session_month"] = sessions["min"].dt.month
    sessions["session_day"] = sessions["min"].dt.day
    sessions["session_weekday"] =  sessions["min"].dt.weekday
    sessions["session_start_hour"] =  sessions["session_start_timestamp"].dt.hour
    sessions["session_duration"] = (sessions["session_end_timestamp"].dt.tz_localize(None) - \
                                         sessions["session_start_timestamp"].dt.tz_localize(None)).dt.total_seconds()
    sessions.drop(["min", "max"], axis = 1, inplace = True)

    tmp_df = sessions.drop_duplicates(["installation_id", "session"], keep = "last").\
                    sort_values(["installation_id", "session"])\
                    [["installation_id", "session", "session_start_timestamp", "session_end_timestamp"]]
    tmp_df["set_prev"] = tmp_df.groupby("installation_id")["session_end_timestamp"].shift(1)
    tmp_df["time_since_last_session"] = (tmp_df["session_start_timestamp"].dt.tz_localize(None) - \
                                        tmp_df["set_prev"].dt.tz_localize(None)).dt.total_seconds()
    tmp_df.drop(["session_start_timestamp", "session_end_timestamp", "set_prev"], axis = 1, inplace=True)
    sessions = sessions.merge(tmp_df, on = ["installation_id", "session"], how = "left")
    del tmp_df
    sessions.drop(["session_start_timestamp", "session_end_timestamp", "installation_id"], axis = 1, inplace=True)
    return sessions

@print_shape
@timing
def get_accuracy_features(df, is_test = False, all_assessments = False):
    sp = df[["installation_id", "game_session", "gs_starttime", "gs_endtime", "gs_duration", "title", "event_count"]]\
                .drop_duplicates("game_session", keep = "last")
    df.sort_values(["installation_id", "timestamp"], inplace=True)
    pass_events = df.query('type == "Assessment" & (event_code == 4100 | event_code == 4110 | event_code == 2000)')\
                                [["event_data", "game_session", "title", "event_code"]].copy()
    pass_events.sort_values(["game_session", "event_code"], inplace = True)
    pass_events["len_sess"] = pass_events.groupby("game_session")["event_code"].transform("size")
    pass_events["drop_row"] = pass_events.apply(lambda row: np.logical_and(row["len_sess"] > 1, row["event_code"] == 2000), 
                                               axis = 1)
    pass_events = pass_events[~pass_events['drop_row']].drop(["len_sess", "drop_row"], axis = 1)
    pass_events["correct"] = pass_events["event_data"].map(check_correct)

    # drop rows corresponding to `Bird Measurer` and `event_code == 4100`
    pass_events['drop_row'] = pass_events.apply(keep_fn, axis = 1)
    pass_events = pass_events[~pass_events['drop_row']]

    pass_events.drop(["event_data", "drop_row"], axis =1, inplace = True)

    pvt = pass_events.pivot_table(index = ["game_session"], columns = ["correct"], values = "event_code", 
                                  aggfunc=lambda x: len(x), fill_value=0).reset_index()
    pvt["calc_accuracy_score"] = pvt.apply(get_accuracy_group, axis = 1)
    pvt["calc_accuracy"] = pvt.apply(get_accuracy, axis = 1)
    pvt = pvt.merge(sp, on = "game_session", how = "left")

    if is_test:
        temp_df = df.drop_duplicates(["installation_id"], keep = "last")[["game_session"]]
        temp_df["test_session"] = 1
        pvt = pvt.merge(temp_df, on = "game_session", how = "left")
        pvt["test_session"].fillna(0, inplace = True)
        pvt["calc_accuracy_score"][pvt["test_session"] == 1] = 99

    pvt.sort_values(["installation_id", "gs_starttime"], inplace=True)
    if all_assessments:
        pvt['cum_accuracy'] = pvt.query('calc_accuracy != -1').groupby(['installation_id', 'title'])['calc_accuracy'].transform(
                        lambda x: x.expanding(1).mean())
        pvt_title = pd.pivot_table(pvt, index = ["game_session"], columns = ["title"], values = "cum_accuracy", 
                             fill_value=np.nan, dropna = False).reset_index()
        pvt_title = pvt_title.merge(sp[["installation_id", "game_session", "gs_starttime"]], 
            on = "game_session", how = "left")
        pvt_title.sort_values(["installation_id", "gs_starttime"], inplace=True)
        for col in pvt.title.unique():
            pvt_title[col] = pvt_title.groupby('installation_id')[col].transform(lambda x: x.shift(1).ffill())
        pvt_title.fillna(-1, inplace = True)
        pvt_title.drop(["installation_id", "gs_starttime"], axis = 1, inplace = True)
        rename_dict = dict(zip(pvt.title.unique(), [x + "_acc" for x in pvt.title.unique().tolist()]))
        pvt_title.rename(columns = rename_dict, inplace = True)

    grp = pvt.query('calc_accuracy_score != -1').groupby(["installation_id"])
    pvt["accum_accuracy_group"] = grp["calc_accuracy_score"].apply(lambda x: x.expanding(min_periods = 1).median().shift(1))
    pvt["accum_accuracy"] = grp["calc_accuracy"].apply(lambda x: x.expanding(min_periods = 1).median().shift(1))
    pvt["time_since_last_assessment"] = grp["gs_endtime"].transform(lambda x:x.diff().dt.total_seconds())
    #pvt["time_since_last_assessment"] = pvt["time_since_last_assessment"] - pvt["gs_duration"]
    pvt["accuracy_last_assessment"] = grp["calc_accuracy_score"].shift(1)
    pvt["num_assess_td"] = grp["game_session"].cumcount()
    pvt["assessment_duration_mean_td"] = grp["gs_duration"].apply(lambda x: x.expanding(min_periods = 1).mean().shift(1))
    pvt["assessment_event_count_mean_td"] = grp["event_count"].apply(lambda x: x.expanding(min_periods = 1).mean().shift(1))
    pvt["assessment_n_correct_td"] = grp["T"].apply(lambda x: x.expanding(min_periods = 1).sum().shift(1))
    pvt["assessment_n_incorrect_td"] = grp["F"].apply(lambda x: x.expanding(min_periods = 1).sum().shift(1))

    grp = pvt.query('calc_accuracy_score != -1').groupby(["installation_id", "title"])
    pvt["accum_accuracy_group_title"] = grp["calc_accuracy_score"].apply(lambda x: x.expanding(min_periods = 1).median().shift(1))
    pvt["accum_accuracy_title"] = grp["calc_accuracy"].apply(lambda x: x.expanding(min_periods = 1).median().shift(1))
    pvt["time_since_last_assessment_title"] = grp["gs_endtime"].apply(lambda x:x.diff().dt.total_seconds())
    #pvt["time_since_last_assessment_title"] = pvt["time_since_last_assessment_title"] - pvt["gs_duration"]
    pvt["accuracy_last_assessment_title"] = grp["calc_accuracy_score"].shift(1)
    pvt["num_assess_td_title"] = grp["game_session"].cumcount()
    pvt["assess_duration_mean_td_title"] = grp["gs_duration"].apply(lambda x: x.expanding(min_periods = 1).mean().shift(1))
    pvt["assess_event_count_mean_td_title"] = grp["event_count"].apply(lambda x: x.expanding(min_periods = 1).mean().shift(1))
    pvt["assess_n_correct_td_title"] = grp["T"].apply(lambda x: x.expanding(min_periods = 1).sum().shift(1))
    pvt["assess_n_incorrect_td_title"] = grp["F"].apply(lambda x: x.expanding(min_periods = 1).sum().shift(1))
    del grp

    pvt["n_incomplete_attemps_td"] = pvt.groupby(["installation_id", "title"])["calc_accuracy_score"].apply(
        lambda x: (x == -1).cumsum()
    )
    pvt["time_since_last_attempt"] = np.nan
    pvt["time_since_last_attempt"][pvt["calc_accuracy_score"] == -1] = pvt["gs_endtime"]
    pvt["time_since_last_attempt"] = pvt.groupby(["installation_id", "title"])["time_since_last_attempt"].ffill()
    pvt["time_since_last_attempt"] = (pvt["gs_starttime"].dt.tz_localize(None) - pvt["time_since_last_attempt"].dt.tz_localize(None)).dt.total_seconds()
    #pvt["time_since_last_attempt"] = pvt["time_since_last_attempt"] - pvt["gs_duration"]

    if all_assessments:
        pvt.drop(["cum_accuracy"], axis = 1, inplace = True)
        pvt = pvt.merge(pvt_title, on = "game_session", how = "left")

    if is_test:
        pvt = pvt.query('calc_accuracy_score == 99')
        pvt.drop("test_session", axis = 1, inplace = True)
       
    if not is_test:
        drop_cols = ["F", "T", "n", "calc_accuracy", "calc_accuracy_score", "title",
                     "gs_starttime", "gs_endtime", "gs_duration", "event_count", "installation_id"]
        pvt.drop(drop_cols, axis = 1, inplace=True)
    else:
        drop_cols = ["F", "T", "n", "calc_accuracy", "calc_accuracy_score", "title",
                     "gs_starttime", "gs_endtime", "gs_duration", "event_count", "game_session"]
        pvt.drop(drop_cols, axis = 1, inplace=True)
    pvt.fillna(-1, inplace = True)
    return pvt


@print_shape
@timing
def get_count_time_features(df, is_test = False):   
    df = copy.deepcopy(df)
    aggregate_cols = [
        "total__events", "n__distinct_eventid", "n__distinct_eventcode", "n__distinct_titles",
        "n__distinct_Clips", "n__distinct_Games", "n__distinct_Activity", "n_game_sessions", "sum_gametime"
    ]
    if not is_test:
        gs_map = get_agg_map(df)
        df.sort_values(["installation_id", "timestamp"], inplace= True)
        df["total__events"] = df.groupby("installation_id")["event_id"].cumcount()
        df["n__distinct_eventid"] = df.groupby('installation_id')['event_id'].transform(
            lambda x: pd.Series(pd.factorize(x)[0] + 1).cummax())
        df["n__distinct_eventcode"] = df.groupby('installation_id')['event_code'].transform(
            lambda x: pd.Series(pd.factorize(x)[0] + 1).cummax())
        df["n__distinct_titles"] = df.groupby('installation_id')['title'].transform(
            lambda x: pd.Series(pd.factorize(x)[0] + 1).cummax())
        df["n__distinct_Clips"] = df.query('type == "Clip"').groupby('installation_id')['title'].transform(
            lambda x: pd.Series(pd.factorize(x)[0] + 1).cummax())
        df["n__distinct_Games"] = df.query('type == "Game"').groupby('installation_id')['title'].transform(
            lambda x: pd.Series(pd.factorize(x)[0] + 1).cummax())
        df["n__distinct_Activity"] = df.query('type == "Activity"').groupby('installation_id')['title'].transform(
            lambda x: pd.Series(pd.factorize(x)[0] + 1).cummax())
        for col in ['n__distinct_Clips', 'n__distinct_Games', 'n__distinct_Activity']:
            df[col] = df.groupby('installation_id')[col].ffill()

        df.drop_duplicates("game_session", keep = "last", inplace = True)
        df["n_game_sessions"] = df.groupby('installation_id')["game_session"].cumcount()
        df["sum_gametime"] = df.groupby('installation_id')["game_time"].transform("cumsum")
        df = df.merge(gs_map, on = 'game_session', how = 'left')
        df.drop_duplicates("session_aggregate", keep = "last", inplace = True)
        df = df[["session_aggregate"] + aggregate_cols]
        df.rename(columns = {"session_aggregate" : "game_session"}, inplace= True)
        df.fillna(0, inplace= True)
        return df
    
    else:
        agg_dict = {
            "event_id": ["size", pd.Series.nunique],
            "event_code": pd.Series.nunique,
            "title": pd.Series.nunique,
        }
        agg = df.groupby('installation_id').agg(agg_dict).reset_index()
        agg.columns  = ["_".join((str(j), i)).strip("_") for i,j in agg.columns]
        agg2 = df.query('type == "Game"').groupby("installation_id")["title"].agg(pd.Series.nunique).reset_index().rename(
            columns = {"title" : "n__distinct_Games"})
        agg = agg.merge(agg2, on = 'installation_id', how= 'left')
        agg2 = df.query('type == "Activity"').groupby("installation_id")["title"].agg(pd.Series.nunique).reset_index().rename(
            columns = {"title" : "n__distinct_Activity"})
        agg = agg.merge(agg2, on = 'installation_id', how= 'left')
        agg2 = df.query('type == "Clip"').groupby("installation_id")["title"].agg(pd.Series.nunique).reset_index().rename(
            columns = {"title" : "n__distinct_Clips"})
        agg = agg.merge(agg2, on = 'installation_id', how= 'left')
        
        df.sort_values(["installation_id", "timestamp"], inplace= True)
        df.drop_duplicates("game_session", keep = "last", inplace= True)
        
        agg2 = df.groupby("installation_id").agg({
            "game_session": "size",
            "game_time": "sum"
        }).reset_index()
        agg = agg.merge(agg2, on = 'installation_id', how= 'left')
        
        agg.rename(columns = {
            "size_event_id": "total__events",
            "nunique_event_id": "n__distinct_eventid",
            "nunique_event_code": "n__distinct_eventcode",
            "nunique_title": "n__distinct_titles",
            "game_time": "sum_gametime",
            "game_session": "n_game_sessions"
        }, inplace = True)
        agg["n_game_sessions"] = agg["n_game_sessions"] - 1
        agg.fillna(0, inplace = True)
        return agg       


@print_shape
@timing
def get_activity_type_features(df, sess_df, is_test = False):
    df = copy.deepcopy(df)
    if not is_test:
        temp_df = df[["installation_id", "game_session", "timestamp"]].drop_duplicates(["installation_id", "game_session"], keep = 'first')
        gs_map = get_agg_map(df)
    
    df = df.drop_duplicates("game_session", keep = 'last')
    df = df[["installation_id", "game_session", "type", "game_time", "gs_starttime", "gs_endtime"]].copy()
    
    if not is_test:
        df = df.merge(gs_map, on = "game_session", how = "left")
    
    df = df.merge(sess_df[["game_session", "session"]], on = "game_session", how = "left")
    df.sort_values(["installation_id", "gs_starttime"], inplace = True)
    df.drop("game_session", axis =1, inplace=True)
    df["next_gs_starttime"] = df.groupby(["installation_id", "session"])["gs_starttime"].transform(lambda x: x.shift(-1))

    df["duration"] = (df["next_gs_starttime"].dt.tz_localize(None) -  df["gs_starttime"].dt.tz_localize(None)).dt.total_seconds()
    df["game_time"][df.type == 'Clip'] = df['duration']
    df.drop(['gs_starttime', 'gs_endtime', 'next_gs_starttime', 'duration', 'session'], axis = 1, inplace=True)

    df['sum_gametime__type'] = df.groupby(['installation_id', 'type'])['game_time'].transform('cumsum')
    
    if is_test:
        df.drop(['game_time'], axis = 1, inplace = True)
        df.drop_duplicates(["installation_id", "type"], keep = "last", inplace=True)
        df = pd.pivot_table(df, index='installation_id', columns=['type'], values='sum_gametime__type').reset_index()
        del df.columns.name
    else:
        df.drop(['installation_id', 'game_time'], axis = 1, inplace = True)
        df.drop_duplicates(["session_aggregate", "type"], keep = "last", inplace=True)
        df = pd.pivot_table(df, index='session_aggregate', columns=['type'], values='sum_gametime__type').reset_index()
        del df.columns.name
        df = df.merge(temp_df, left_on = "session_aggregate", right_on = 'game_session', how= 'left')
        df.drop("game_session", axis = 1, inplace = True)
        df.sort_values(["installation_id", "timestamp"], inplace = True)
        for col in ["Activity", "Assessment", "Clip", "Game"]:
            df[col] = df.groupby('installation_id')[col].ffill()
        df.drop(["installation_id", "timestamp"], axis= 1, inplace = True)    
    
    df.fillna(0., inplace = True)
    
    col_rename_dict = {
        'Activity': 'total_Activity_time',
        'Clip' : 'total_Clip_time',
        'Game' : 'total_Game_time',
        'Assessment' : 'total_Assessment_time'
    }
    if not is_test:
        col_rename_dict.update({"session_aggregate": "game_session"})    
    df.rename(columns = col_rename_dict, inplace = True)

    df["sm"] = df.iloc[:, 1:].sum(axis = 1)
    for col in ["total_Activity_time", "total_Clip_time", "total_Game_time", "total_Assessment_time"]:
        df[col] = df[col] / df["sm"]
    df.drop("sm", axis =1, inplace = True)
    return df

@print_shape
@timing
def get_world_features(df, is_test = False):
    df = copy.deepcopy(df)
    df.sort_values(["installation_id", "timestamp"], inplace= True)
    if not is_test:
        gs_map = get_agg_map(df)
        
    df = df.drop_duplicates("game_session", keep = 'last')\
                [["installation_id", "game_session", "game_time", "world"]]
    
    df["game_sessions__world"] = df.groupby(["installation_id", "world"]).cumcount()
    df["total_gametime__world"] = df.groupby(["installation_id", "world"])["game_time"].transform("cumsum")
    
    if not is_test:
        df = df.merge(gs_map, on = 'game_session', how= 'left')
        df = df.drop_duplicates(["session_aggregate", "world"], keep = "last")[["session_aggregate", "world", "game_sessions__world", 
                                                                                    "total_gametime__world"]]
        df = df.query('world != "NONE"')
        df.rename(columns = {"session_aggregate" : "game_session"}, inplace= True)
        return df
    else:
        df = df.drop_duplicates(["installation_id", "world"], keep = "last")[["installation_id", "world", "game_sessions__world", 
                                                                                    "total_gametime__world"]]
        df = df.query('world != "NONE"')
        return df
    
@print_shape
@timing
def get_last_activity_features(df, sess_df):
    df = copy.deepcopy(df)
    df = df.drop_duplicates("game_session", keep = 'last')[["installation_id", "game_session", "title", "timestamp"]]
    df = df.merge(sess_df, on = 'game_session', how = 'left')
    df.sort_values(["installation_id", "timestamp"], inplace= True)
    grp = df.groupby(['installation_id', 'session'])
    df['last_activity_1'] = grp['title'].shift(1)
    df['last_activity_2'] = grp['title'].shift(2)
    df['last_activity_3'] = grp['title'].shift(3)
    return df[["installation_id", "game_session", "last_activity_1", "last_activity_2", "last_activity_3"]]


@print_shape
@timing
def events_per_sec(df, is_test = False):
    df = copy.deepcopy(df)
    if not is_test:
        gs_map = get_agg_map(df)
    df = df.drop_duplicates(["game_session"], keep = "last")[["installation_id", "game_session", "timestamp", "event_count", "gs_duration"]]
    df = df.query('gs_duration != 0')
    df["events_per_sec"] = df["event_count"] / df["gs_duration"]
    df["evps"] = df.groupby('installation_id')['events_per_sec'].transform(lambda x: x.expanding(min_periods = 1).mean())
    df["mean_event_count"] = df.groupby('installation_id')['event_count'].transform(lambda x: x.expanding(min_periods = 1).mean())
    if not is_test:
        df = df.merge(gs_map, on = "game_session", how = "left")
        df.sort_values(['installation_id', 'timestamp'], inplace = True)
        df.drop_duplicates('session_aggregate', keep = 'last', inplace = True)
        df = df[["session_aggregate", "evps", "mean_event_count"]].rename(columns = {'session_aggregate': 'game_session'})
        df.fillna(-1, inplace= True)
        return df
    df.sort_values(['installation_id', 'timestamp'], inplace = True)
    df.drop_duplicates('installation_id', keep = 'last', inplace = True)
    df = df[['installation_id', 'evps', 'mean_event_count']]
    df.fillna(-1, inplace= True)
    return df

@print_shape
@timing
def events_per_sec_session(df, sess_df, is_test= False):
    df = copy.deepcopy(df)
    if not is_test:
        gs_map = get_agg_map(df)
    
    df = df.drop_duplicates(["game_session"], keep = "last")[["installation_id", "game_session", "timestamp", "event_count", "gs_duration"]]
    df = df.query('gs_duration != 0')
    df["events_per_sec"] = df["event_count"] / df["gs_duration"]
    df = df.merge(sess_df[['game_session', 'session']], on = 'game_session', how = 'left')
    df.sort_values(['installation_id', 'timestamp'], inplace = True)
    df["evps_session"] = df.groupby(['installation_id', 'session'])['events_per_sec'].\
                                        transform(lambda x: x.expanding(min_periods = 1).mean())
    df["mean_event_count_session"] = df.groupby(['installation_id', 'session'])['event_count'].\
                                        transform(lambda x: x.expanding(min_periods = 1).mean())
    
    if not is_test:
        df = df.merge(gs_map, on = 'game_session', how = 'left')
        df.sort_values(['installation_id', 'timestamp'], inplace = True)
        df.drop_duplicates('session_aggregate', keep = 'last', inplace = True)
        df = df[["session_aggregate", "evps_session", "mean_event_count_session"]].rename(columns = {'session_aggregate': 'game_session'})
        df.fillna(-1, inplace= True)
        return df
    
    df.sort_values(['installation_id', 'timestamp'], inplace = True)
    df.drop_duplicates(['installation_id', 'session'], keep = 'last', inplace = True)
    df = df[['installation_id', 'evps_session', 'mean_event_count_session']]
    df.fillna(-1, inplace= True)
    return df

@print_shape
@timing
def get_game_stats(df, is_test = False, labels_df = None):
    df = copy.deepcopy(df)
    if np.logical_and(not is_test, labels_df is None):
        raise ValueError('Must specify `labels_df` for train transformation!')
        
    df.sort_values(['installation_id', 'timestamp'], inplace= True)
    merge_df = df.drop_duplicates(["game_session"])[["installation_id", "game_session", "timestamp"]]
    if is_test:
        test_df = df.drop_duplicates(["installation_id"], keep = 'last')[["installation_id", "title"]]
    
    game_cols = ['happy_camel', 'leaf_leader', 'pan_balance', 'dino_dive', 'chow_time', 'scrub_a_dub', 'all_star_sorting', 'air_show', 
            'dino_drink', 'bubble_bath', 'crystals_rule'] + \
            ["cb_balance_ratio", "ed_activity_level", "sc_activity_level", "bf_n_jars", "bf_max_round", "wh_activity_level", 
             "fw_activity_level", "flw_activity_level", "bm_activity_level"]
    
    key = 'game_session'
    chow_time = Chow_Time(df)
    dino_drink = Dino_Drink(df)
    dino_dive = Dino_Dive(df)
    crystals_rule = Crystals_Rule(df)
    happy_camel = Happy_Camel(df)
    bubble_bath = Bubble_Bath(df)
    scrub_a_dub = Scrub_A_Dub(df)
    pan_balance = Pan_Balance(df)
    all_star_sorting = All_Star_Sorting(df)
    air_show = Air_Show(df)
    leaf_leader = Leaf_Leader(df)

    chicken_balancer = Chicken_Balancer(df)
    egg_dropper = Egg_Dropper(df)
    fireworks = Fireworks(df)
    flower_waterer = Flower_Waterer(df)
    watering_hole = Watering_Hole(df)
    sandcastle = Sandcastle_Builder(df)
    bottle_filler = Bottle_Filler(df)
    bug_measurer = Bug_Measurer(df)
    
    mushroom_sorter = Mushroom_Sorter(df)
    bird_measurer = Bird_Measurer(df)
    cauldron_filler = Cauldron_Filler(df)
    cart_balancer = Cart_Balancer(df)
    chest_sorter = Chest_Sorter(df)
    
    merge_df = merge_df.merge(happy_camel.rename(columns = {"acc" : "happy_camel"}), on = key, how = "left" )
    merge_df = merge_df.merge(leaf_leader.rename(columns = {"acc" : "leaf_leader"}), on = key, how = "left" )
    merge_df = merge_df.merge(pan_balance.rename(columns = {"acc" : "pan_balance"}), on = key, how = "left" )
    merge_df = merge_df.merge(dino_dive.rename(columns = {"acc" : "dino_dive"}), on = key, how = "left" )
    merge_df = merge_df.merge(chow_time.rename(columns = {"acc" : "chow_time"}), on = key, how = "left" )
    merge_df = merge_df.merge(scrub_a_dub.rename(columns = {"acc" : "scrub_a_dub"}), on = key, how = "left" )
    merge_df = merge_df.merge(all_star_sorting.rename(columns = {"acc" : "all_star_sorting"}), on = key, 
                  how = "left" )
    merge_df = merge_df.merge(air_show.rename(columns = {"acc" : "air_show"}), on = key, how = "left" )
    merge_df = merge_df.merge(dino_drink.rename(columns = {"acc" : "dino_drink"}), on = key, how = "left" )
    merge_df = merge_df.merge(bubble_bath.rename(columns = {"acc" : "bubble_bath"}), on = key, how = "left" )
    merge_df = merge_df.merge(crystals_rule.rename(columns = {"acc" : "crystals_rule"}), on = key, how = "left" )

    merge_df = merge_df.merge(chicken_balancer, on = key, how = "left")
    merge_df = merge_df.merge(egg_dropper, on = key, how = "left")
    merge_df = merge_df.merge(sandcastle, on = key, how = "left")
    merge_df = merge_df.merge(bottle_filler, on = key, how = "left")
    merge_df = merge_df.merge(watering_hole, on = key, how = "left")
    merge_df = merge_df.merge(fireworks, on = key, how = "left")
    merge_df = merge_df.merge(flower_waterer, on = key, how = "left")
    merge_df = merge_df.merge(bug_measurer, on = key, how = "left")
    
    A = pd.concat([mushroom_sorter, bird_measurer, cauldron_filler, cart_balancer, chest_sorter], ignore_index=True)
    
    if not is_test:
        labels_df = copy.deepcopy(labels_df)
        labels_df["assessment"] = True
        merge_df = merge_df.merge(labels_df[["game_session", "title", "assessment", "accuracy_group"]], on = key, how = "left")
        merge_df["na_sum"] = merge_df.iloc[:, 3:].isnull().sum(axis = 1)
        merge_df = merge_df.query('na_sum < 23')
        merge_df.sort_values(["installation_id", "timestamp"], inplace=True)
        for col in game_cols:
            merge_df[col] = merge_df.groupby('installation_id')[col].ffill()
        merge_df.dropna(subset=["title"], inplace=True)
        merge_df.drop(["assessment", "na_sum"], axis = 1, inplace= True)
        
        T_assess = df.query('type == "Assessment"').drop_duplicates("game_session", keep = "first")[["installation_id", "game_session", "title", "timestamp"]]
        gs_map = get_agg_map(df)
        T_assess = T_assess.merge(gs_map, on = "game_session", how = "left")
        T_assess = T_assess.merge(A, on = "game_session", how = "left")
        T_assess.sort_values(["installation_id", "timestamp"], inplace=True)
        T_assess = T_assess[T_assess.game_session.isin(labels_df.game_session)]
        T_assess["cumulative_conf"] = T_assess.groupby(["installation_id", "title"])["conf"].transform(lambda x: x.expanding(1).mean().shift(1))
        T_assess["cumulative_conf"].fillna(-1, inplace = True)
        
        merge_df = merge_df.merge(T_assess[["game_session", "cumulative_conf"]], on = "game_session", how = "left")
        merge_df.fillna(-1, inplace = True)
        
    else:
        
        T_assess_test = df.query('type == "Assessment"').drop_duplicates("game_session", keep = "first")[["installation_id", "game_session", "title", "timestamp"]]
        T_assess_test = T_assess_test.merge(A, on = "game_session", how = "left")
        T_assess_test.sort_values(["installation_id", "timestamp"], inplace=True)
        T_assess_test.dropna(subset = ['conf'], inplace = True)
        T_assess_test["cumulative_conf"] = T_assess_test.groupby(["installation_id", "title"])["conf"].transform(lambda x: x.expanding(1).mean())
        T_assess_test.drop_duplicates(["installation_id", "title"], keep = "last", inplace=True)
        T_assess_test.drop(["game_session", "timestamp", "conf"], axis= 1, inplace = True)
        test_df = test_df.merge(T_assess_test, on = ['installation_id', 'title'], how = 'left')
        
        merge_df.sort_values(["installation_id", "timestamp"], inplace=True)
        for col in game_cols:
            merge_df[col] = merge_df.groupby('installation_id')[col].ffill()
        merge_df.drop_duplicates('installation_id', keep = 'last', inplace = True)
        merge_df = test_df.merge(merge_df, on = 'installation_id', how = 'left')
        merge_df.drop(['timestamp'], axis = 1, inplace= True)
        merge_df.fillna(-1, inplace = True)
    return merge_df

# @print_shape
# @timing
# def get_path_eff(df, sess_df, labels_df, is_test = False, ignore = ignore_path_nodes, gamma = 0.6):
#     if not is_test:
#         gs_map = get_agg_map(df, include_present = True)
    
#     seq_df = df.drop_duplicates(["game_session"], keep = "first")[["installation_id", "game_session", "timestamp","title", "world"]].copy()
#     seq_df = seq_df.merge(sess_df[["game_session", "session"]], on = "game_session", how = "left")
#     if not is_test:
#         seq_df = seq_df.merge(gs_map[["game_session", "session_aggregate"]], on = "game_session", how = "left")
#         seq_df.dropna(subset = ["session_aggregate"], inplace=True)
#     seq_df.sort_values(["installation_id", "timestamp"], inplace=True)
#     if ignore is not None:
#         seq_df = seq_df[~seq_df.title.isin(ignore)]
    
#     seq_df["title_TreeTopCity"] = seq_df["title"].map(alphabet_map_treetopcity)
#     seq_df["title_MagmaPeak"] = seq_df["title"].map(alphabet_map_magmapeak)
#     seq_df["title_CrystalCaves"] = seq_df["title"].map(alphabet_map_crystalcaves)
    
#     seq_df["seq_TreeTopCity"] = [''.join(y.title_TreeTopCity.tolist()[:z+1]) for x, y in 
#                                          seq_df.groupby(['installation_id', 'session']) for z in range(len(y))]
#     seq_df["seq_MagmaPeak"] = [''.join(y.title_MagmaPeak.tolist()[:z+1]) for x, y in 
#                                          seq_df.groupby(['installation_id', 'session']) for z in range(len(y))]
#     seq_df["seq_CrystalCaves"] = [''.join(y.title_CrystalCaves.tolist()[:z+1]) for x, y in 
#                                          seq_df.groupby(['installation_id', 'session']) for z in range(len(y))]
    
#     if not is_test:
#         seq_df.drop_duplicates("session_aggregate", keep = "last", inplace = True)
#     else:
#         seq_df.drop_duplicates("installation_id", keep = "last", inplace = True)
        
#     seq_df["seq"] = "P"
#     seq_df["seq"][seq_df.world == 'TREETOPCITY'] = seq_df['seq_TreeTopCity']
#     seq_df["seq"][seq_df.world == 'CRYSTALCAVES'] = seq_df['seq_CrystalCaves']
#     seq_df["seq"][seq_df.world == 'MAGMAPEAK'] = seq_df['seq_MagmaPeak']
#     seq_df["seq"] = seq_df.seq.map(clean_seq)

#     seq_df = seq_df.merge(labels_df[["game_session", "accuracy_group"]], on = "game_session", how = "left")
    
#     path_eff = {}
#     for title in assess_titles:
#         s1 = seq_df.query('title == @title')['seq']
#         s2 = seq_df.query('title == @title')['accuracy_group'].replace(0, -1)
#         all_ = zip(s1, s2)
#         mapped = [dict(zip([(x[i-1], x[i]) for i in range(len(x)-1, 0 ,-1)], [y*gamma**(i+1) for i in range(len(x) - 1)])) for x, y in all_]
#         dd = defaultdict(list)
#         for d in mapped:
#             for k, v in d.items():
#                 dd[k].append(v)
#         dd = {k: np.nanmean(v) * (1 - np.exp(-len(v)/30.)) for k, v in dd.items()}
#         path_eff[title] = dd
        
#     seq_df['path_eff'] = seq_df.apply(lambda row:
#                                       np.sum([path_eff[row['title']][x] for x in [(row['seq'][i-1], row['seq'][i]) for 
#                                                                         i in range(len(row['seq'][:-1]), 0, -1)]]) *  np.exp(-len(row['seq'])/10.), 
#                                   axis = 1)
#     seq_df = seq_df[["game_session", "path_eff"]]
#     return seq_df

# def jumps(path):
#     indices = np.array([ord(x) for x in list(path)])
#     return np.mean(indices[1:] - indices[:-1])

class PathTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, ignore = ignore_path_nodes, gamma = 0.6, assess_titles = assess_titles):
        #self.sess_df = sess_df
        self.ignore = ignore_path_nodes
        self.gamma = gamma
        self.assess_titles = assess_titles
        self.gs_map = None  
        self.path_eff = {}
        self.seq_df = None
        self.compiled = False
        
    def get_sequences(self, df, sess_df):
        df = copy.deepcopy(df)
        df = df.drop_duplicates(["game_session"], keep = "first")[["installation_id", "game_session", "timestamp","title", "world"]].copy()
        df = df.merge(sess_df[["game_session", "session"]], on = "game_session", how = "left")
        
        if self.ignore is not None:
            df = df[~df.title.isin(self.ignore)]
        
        df.sort_values(["installation_id", "timestamp"], inplace=True)
        
        df["title_TreeTopCity"] = df["title"].map(alphabet_map_treetopcity)
        df["title_MagmaPeak"] = df["title"].map(alphabet_map_magmapeak)
        df["title_CrystalCaves"] = df["title"].map(alphabet_map_crystalcaves)

        df["seq_TreeTopCity"] = [''.join(y.title_TreeTopCity.tolist()[:z+1]) for x, y in 
                                             df.groupby(['installation_id', 'session']) for z in range(len(y))]
        df["seq_MagmaPeak"] = [''.join(y.title_MagmaPeak.tolist()[:z+1]) for x, y in 
                                             df.groupby(['installation_id', 'session']) for z in range(len(y))]
        df["seq_CrystalCaves"] = [''.join(y.title_CrystalCaves.tolist()[:z+1]) for x, y in 
                                             df.groupby(['installation_id', 'session']) for z in range(len(y))]
        
        df["seq"] = "P"
        df["seq"][df.world == 'TREETOPCITY'] = df['seq_TreeTopCity']
        df["seq"][df.world == 'CRYSTALCAVES'] = df['seq_CrystalCaves']
        df["seq"][df.world == 'MAGMAPEAK'] = df['seq_MagmaPeak']
        df["seq"] = df.seq.map(clean_seq)
        return df
        
    @timing
    def fit(self, X, Y, sess_df):        
        self.gs_map = get_agg_map(X, include_present = True)
        seq_df = self.get_sequences(X, sess_df)
        seq_df = seq_df.merge(self.gs_map[["game_session", "session_aggregate"]], on = "game_session", how = "left")
        seq_df.drop_duplicates("session_aggregate", keep = "last", inplace = True)

        seq_df = seq_df.merge(Y[["game_session", "accuracy_group"]], on = "game_session", how = "left")

        for title in self.assess_titles:
            s1 = seq_df.query('title == @title')['seq']
            s2 = seq_df.query('title == @title')['accuracy_group'].replace(0, -1)
            all_ = zip(s1, s2)
            mapped = [dict(zip([(x[i-1], x[i]) for i in range(len(x)-1, 0 ,-1)], [y*self.gamma**(i+1) for i in range(len(x) - 1)])) for x, y in all_]
            dd = defaultdict(list)
            for d in mapped:
                for k, v in d.items():
                    dd[k].append(v)
            dd = {k: np.nanmean(v) * (1 - np.exp(-len(v)/30.)) for k, v in dd.items()}
            self.path_eff[title] = dd
        self.seq_df = seq_df
        self.compiled = True

#     @timing
#     def transform_jumps(self, X):
#         if not self.compiled:
#             raise ValueError("Please call `fit` before `transform`!")
#         if not is_test:
#             self.seq_df["jump"] = self.seq_df["seq"].map(jumps)
            
    
    @timing
    def transform(self, X, is_test = False, sess_df = None):
        if np.logical_and(is_test, sess_df is None):
            raise ValueError("Please specify session mapping for the test records!")
        
        if not is_test:
            self.seq_df['path_eff'] = self.seq_df.apply(lambda row:
                                      np.sum([self.path_eff[row['title']][x] for x in [(row['seq'][i-1], row['seq'][i]) for 
                                                                        i in range(len(row['seq'][:-1]), 0, -1)]]) *  np.exp(-len(row['seq'])/10.), 
                                  axis = 1)
            #self.seq_df["jumps"] = self.seq_df["seq"].map(jumps)
            #self.seq_df["jumps"].fillna(-1, inplace = True)
            #self.seq_df = self.seq_df[["game_session", "path_eff", "jumps"]]
            return self.seq_df[["game_session", "path_eff"]]
        else:
            seq_df = self.get_sequences(X, sess_df)
            seq_df.sort_values(["installation_id", "timestamp"], inplace = True)
            seq_df.drop_duplicates("installation_id", keep = "last", inplace = True)
            seq_df['path_eff'] = seq_df.apply(lambda row:
                                      np.sum([self.path_eff[row['title']].get(x, 0) for x in [(row['seq'][i-1], row['seq'][i]) for 
                                                                        i in range(len(row['seq'][:-1]), 0, -1)]]) *  np.exp(-len(row['seq'])/10.), 
                                  axis = 1)
            #seq_df["jumps"] = seq_df["seq"].map(jumps)
            #seq_df.jumps.fillna(-1, inplace = True)
            return seq_df[["game_session", "path_eff"]]
        
class PivotTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, is_test = False):
        self.is_test = is_test
        self.gs_map = None
        self.columns = columns
    
    def fit(self, data):
        self.data = copy.deepcopy(data)
        if not self.is_test:
            if 'session_aggregate' not in self.data.columns.values:
                gs_map = get_agg_map(self.data)
                self.data = self.data.merge(gs_map, on = 'game_session', how = 'left')
        self.data.sort_values(["installation_id", "timestamp"], inplace= True)
        return self
    
    @print_shape
    @timing
    def transform(self, data):
        if not self.is_test:
            return_df = self.data[["session_aggregate"]].dropna().drop_duplicates().rename(columns = {"session_aggregate" : "game_session"})
        else:
            return_df = self.data[["installation_id"]].dropna().drop_duplicates()
        
        for col in self.columns:
            if col == 'type':
                self.data[col + "_cnt"] = self.data.groupby(["installation_id", col])['game_session'].transform(lambda x: 
                                                                                                pd.Series(pd.factorize(x)[0] + 1).cummax())
            else:
                self.data[col + "_cnt"] = self.data.groupby(["installation_id", col]).cumcount() + 1
            if not self.is_test:
                temp_df = self.data[["installation_id", "game_session", "session_aggregate", col] + [col+"_cnt"]].copy()
                temp_df.drop_duplicates(["installation_id", "session_aggregate", col], keep = 'last', inplace = True)
                temp_df = temp_df.pivot_table(columns = [col], index = ['session_aggregate'], 
                                 values = [col + "_cnt"]).reset_index()
                temp_df.columns = ['game_session'] + [str(x) for x in temp_df.columns.droplevel(0)[1:]]
                temp_df = temp_df.merge(self.data[["installation_id", "game_session", "timestamp"]]\
                                        .drop_duplicates(["installation_id", "game_session"], keep = 'first'), on = 'game_session', how = 'left')
                temp_df.sort_values(["installation_id", "timestamp"], inplace= True)
                del temp_df.columns.name
                for col in temp_df.columns.values:
                    if col not in ["installation_id", "game_session", "timestamp"]:
                        temp_df[col] = temp_df.groupby("installation_id")[col].ffill()
                        temp_df[col].fillna(0, inplace = True)
                        temp_df[col] = temp_df[col].astype(np.int32)
                temp_df.drop(["installation_id", "timestamp"], axis = 1, inplace= True)
                return_df = return_df.merge(temp_df, on = 'game_session', how = 'left')
            else:
                temp_df = self.data[["installation_id", "game_session", col] + [col+"_cnt"]].copy()
                temp_df.drop_duplicates(["installation_id", col], keep = 'last', inplace = True)
                temp_df = temp_df.pivot_table(columns = [col], index = ['installation_id'], 
                                 values = [col + "_cnt"]).reset_index()
                temp_df.columns = ['installation_id'] + [str(x) for x in temp_df.columns.droplevel(0)[1:]]
                return_df = return_df.merge(temp_df, on = 'installation_id', how = 'left')
                return_df.fillna(0, inplace = True)
                for col in return_df.columns:
                    if col != 'installation_id':
                        return_df[col] = return_df[col].astype(np.int32)
        return return_df