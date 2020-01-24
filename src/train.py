'''
 Created on Sat Dec 28 2019
 __author__: bishwarup
'''

from __future__ import division, print_function
import os
import sys
import re
import json
import joblib
import copy
import random
import warnings
import itertools
import operator
import time
import shutil
import argparse
from scipy.optimize import fmin_powell, minimize
import numpy as np
import pandas as pd
from sklearn.utils.multiclass import unique_labels
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import cohen_kappa_score, mean_squared_error, confusion_matrix, classification_report, r2_score

from .utils import *
from .config import lgb_params, title_world_map, ROOT_DIR, CACHE_DIR

if __name__ == '__main__':
    print("loading datasets...")
    train, test, labels = read_datasets()
    print(f'train_shape: {train.shape}')
    print(f'test_shape: {test.shape}')
    print(f'labels_shape: {labels.shape}')

    print("getting count/time features...")
    train_file_pkl, test_file_pkl   = os.path.join(ROOT_DIR, 'input', 'count_features_train.pkl'), \
                                      os.path.join(ROOT_DIR, 'input', 'count_features_test.pkl')
    COUNT_FEATURE_EXISTS            = np.logical_and(os.path.isfile(train_file_pkl), os.path.isfile(test_file_pkl))
    count_features_train            = pd.read_pickle(train_file_pkl) if COUNT_FEATURE_EXISTS else get_count_time_features(train)
    count_features_test             = pd.read_pickle(test_file_pkl) if COUNT_FEATURE_EXISTS else get_count_time_features(test, is_test = True)
    if not COUNT_FEATURE_EXISTS:
        count_features_train.to_pickle(train_file_pkl)
        count_features_test.to_pickle(test_file_pkl)
    del COUNT_FEATURE_EXISTS

    print("getting sessions features...")
    train_file_pkl, test_file_pkl   = os.path.join(ROOT_DIR, 'input', 'sessions_train.pkl'), \
                                      os.path.join(ROOT_DIR, 'input', 'sessions_test.pkl')
    SESSION_FEATURE_EXIST           = np.logical_and(os.path.isfile(train_file_pkl), os.path.isfile(test_file_pkl))
    sessions_train                  = pd.read_pickle(train_file_pkl) if SESSION_FEATURE_EXIST else get_session_features(train)
    sessions_test                   = pd.read_pickle(test_file_pkl) if SESSION_FEATURE_EXIST else get_session_features(test)
    if not SESSION_FEATURE_EXIST:
        sessions_train.to_pickle(train_file_pkl)
        sessions_test.to_pickle(test_file_pkl)
    del SESSION_FEATURE_EXIST

    print("getting activity features...")
    train_file_pkl, test_file_pkl    =  os.path.join(ROOT_DIR, 'input', 'activity_features_train.pkl'), \
                                        os.path.join(ROOT_DIR, 'input', 'activity_features_test.pkl')
    ACTTIVITY_FEATURE_EXISTS         = np.logical_and(os.path.isfile(train_file_pkl), os.path.isfile(test_file_pkl))
    activity_features_train          = pd.read_pickle(train_file_pkl) if ACTTIVITY_FEATURE_EXISTS else get_activity_type_features(train, sessions_train)
    activity_features_test           = pd.read_pickle(test_file_pkl) if ACTTIVITY_FEATURE_EXISTS else get_activity_type_features(test, sessions_test, is_test=True)
    if not ACTTIVITY_FEATURE_EXISTS:
        activity_features_train.to_pickle(train_file_pkl)
        activity_features_test.to_pickle(test_file_pkl)
    del ACTTIVITY_FEATURE_EXISTS

    print("getting world features...")
    train_file_pkl, test_file_pkl    =   os.path.join(ROOT_DIR, 'input', 'world_features_train.pkl'), \
                                         os.path.join(ROOT_DIR, 'input', 'world_features_test.pkl')
    WORLD_FEATURE_EXISTS             = np.logical_and(os.path.isfile(train_file_pkl), os.path.isfile(test_file_pkl))
    world_features_train             = pd.read_pickle(train_file_pkl) if WORLD_FEATURE_EXISTS else get_world_features(train)
    world_features_test              = pd.read_pickle(test_file_pkl) if WORLD_FEATURE_EXISTS else get_world_features(test, is_test=True)
    if not WORLD_FEATURE_EXISTS:
        world_features_train.to_pickle(train_file_pkl)
        world_features_test.to_pickle(test_file_pkl)
    del WORLD_FEATURE_EXISTS

    print("getting past accuracy features...")
    train_file_pkl, test_file_pkl    = os.path.join(ROOT_DIR, 'input', 'accuracy_features_train.pkl'), \
                                       os.path.join(ROOT_DIR, 'input', 'accuracy_features_test.pkl')
    ACCURACY_FEATURE_EXISTS          = np.logical_and(os.path.isfile(train_file_pkl), os.path.isfile(test_file_pkl))
    accuracy_features_train          = pd.read_pickle(train_file_pkl) if ACCURACY_FEATURE_EXISTS else get_accuracy_features(train, all_assessments = True)
    accuracy_features_test           = pd.read_pickle(test_file_pkl) if ACCURACY_FEATURE_EXISTS else get_accuracy_features(test, all_assessments = True, is_test = True)
    if not ACCURACY_FEATURE_EXISTS:
        accuracy_features_train.to_pickle(train_file_pkl)
        accuracy_features_test.to_pickle(test_file_pkl)
    del ACCURACY_FEATURE_EXISTS

    print("calculating path efficiency...")
    train_file_pkl, test_file_pkl    =  os.path.join(ROOT_DIR, 'input', 'path_efficiency_train.pkl'), \
                                        os.path.join(ROOT_DIR, 'input', 'path_efficiency_test.pkl')
    PATH_FEATURE_EXISTS              =  np.logical_and(os.path.isfile(train_file_pkl), os.path.isfile(test_file_pkl))
    if PATH_FEATURE_EXISTS:
        train_path_eff      = pd.read_pickle(train_file_pkl)
        test_path_eff       = pd.read_pickle(test_file_pkl)
    else:
        pt              = PathTransformer()
        pt.fit(train, labels, sessions_train)
        train_path_eff  = pt.transform(train)
        test_path_eff   = pt.transform(test, is_test=True, sess_df=sessions_test)
        train_path_eff.to_pickle(train_file_pkl)
        test_path_eff.to_pickle(test_file_pkl)
    del PATH_FEATURE_EXISTS

    print("getting one-hot features...")
    train_file_pkl, test_file_pkl    =  os.path.join(ROOT_DIR, 'input', 'onehot_features_train.pkl'), \
                                        os.path.join(ROOT_DIR, 'input', 'onehot_features_test.pkl')
    ONEHOT_FEATURE_EXISTS       =    np.logical_and(os.path.isfile(train_file_pkl), os.path.isfile(test_file_pkl))
    if ONEHOT_FEATURE_EXISTS:
        onehot_features_train   =   pd.read_pickle(train_file_pkl)
        onehot_features_test    =   pd.read_pickle(test_file_pkl)
    else:
        train["title_event"]    = train["title"] + "_" + train["event_code"].astype(str)
        test["title_event"]     = test["title"] + "_" + test["event_code"].astype(str)
        PvtTr                   = PivotTransformer(columns = ['title', 'type', 'event_code', 'event_id', 'title_event'])
        onehot_features_train   = PvtTr.fit_transform(train)
        PvtTr                   = PivotTransformer(columns = ['title', 'type', 'event_code', 'event_id', 'title_event'], is_test=True)
        onehot_features_test    = PvtTr.fit_transform(test)
        onehot_features_train.to_pickle(train_file_pkl)
        onehot_features_test.to_pickle(test_file_pkl)


    print("getting last activity features...")
    train_file_pkl, test_file_pkl    =  os.path.join(ROOT_DIR, 'input', 'last_activity_train.pkl'), \
                                        os.path.join(ROOT_DIR, 'input', 'last_activity_test.pkl')
    LAST_ACT_FEATURE_EXISTS          = np.logical_and(os.path.isfile(train_file_pkl), os.path.isfile(test_file_pkl))
    last_activity_train              = pd.read_pickle(train_file_pkl) if LAST_ACT_FEATURE_EXISTS else get_last_activity_features(train, sessions_train)
    last_activity_test               = pd.read_pickle(test_file_pkl) if LAST_ACT_FEATURE_EXISTS else get_last_activity_features(test, sessions_test)
    if not LAST_ACT_FEATURE_EXISTS:
        last_activity_train.to_pickle(train_file_pkl)
        last_activity_test.to_pickle(test_file_pkl)
    del LAST_ACT_FEATURE_EXISTS

    print("getting game/assessment features...")
    train_file_pkl, test_file_pkl    =  os.path.join(ROOT_DIR, 'input', 'game_stats_train.pkl'), \
                                        os.path.join(ROOT_DIR, 'input', 'game_stats_test.pkl')
    GAME_STATS_EXISTS                = np.logical_and(os.path.isfile(train_file_pkl), os.path.isfile(test_file_pkl))
    train_df                         = pd.read_pickle(train_file_pkl) if GAME_STATS_EXISTS else get_game_stats(train, labels_df = labels)
    test_df                          = pd.read_pickle(test_file_pkl) if GAME_STATS_EXISTS else get_game_stats(test, is_test = True)
    if not GAME_STATS_EXISTS:
        train_df.to_pickle(train_file_pkl)
        test_df.to_pickle(test_file_pkl)
    del GAME_STATS_EXISTS

    train_df["world"] = train_df["title"].map(title_world_map)
    test_df["world"] = test_df["title"].map(title_world_map)

    print("merging train features...")
    train_df = train_df.merge(count_features_train, on = 'game_session', how = 'left')
    train_df = train_df.merge(activity_features_train, on = 'game_session', how = 'left')
    train_df = train_df.merge(sessions_train.drop("session_duration", axis = 1), on = 'game_session', how = 'left')
    train_df = train_df.merge(world_features_train, on = ['game_session', 'world'], how = 'left')
    train_df = train_df.merge(accuracy_features_train, on = 'game_session', how = 'left')
    train_df = train_df.merge(train_path_eff, on = 'game_session', how = 'left')
    train_df = train_df.merge(last_activity_train.drop('installation_id', axis = 1), on = 'game_session', how = 'left')

    print("merging test features...")
    test_df = test_df.merge(count_features_test, on = 'installation_id', how = 'left')
    test_df = test_df.merge(activity_features_test, on = 'installation_id', how = 'left')
    test_df = test_df.merge(sessions_test.drop("session_duration", axis = 1), on = 'game_session', how = 'left')
    test_df = test_df.merge(world_features_test, on = ['installation_id', 'world'], how = 'left')
    test_df = test_df.merge(accuracy_features_test, on = 'installation_id', how = 'left')
    test_df = test_df.merge(test_path_eff, on = 'game_session', how = 'left')
    test_df = test_df.merge(last_activity_test.drop('installation_id', axis = 1), on = 'game_session', how = 'left')

    onehot_cols = [
        "Crystal Caves - Level 3",
        "Clip",
        "Tree Top City - Level 3",
        "4070", "3120", "2000", "3020", "4020", "2030", "4035", "4030",
        "27253bdc", "7372e1a5", "562cec5f",
    ]

    onehot_features     = ["game_session"] + onehot_cols
    train_df            = train_df.merge(onehot_features_train[onehot_features], on = 'game_session', how = 'left')
    onehot_features     = ["installation_id"] + onehot_cols
    test_df             = test_df.merge(onehot_features_test[onehot_features], on = 'installation_id', how = 'left')

    train_df['id']      = np.arange(train_df.shape[0])
    test_df['id']       = test_df['installation_id']

    train_df["title"]   = train_df["title"].map(title_coding)
    test_df["title"]    = test_df["title"].map(title_coding)

    del count_features_train, count_features_test
    del activity_features_train, activity_features_test
    del train_path_eff, test_path_eff
    del accuracy_features_train, accuracy_features_test
    del world_features_train, world_features_test
    del sessions_train, sessions_test
    del onehot_features_train, onehot_features_test
    del last_activity_train, last_activity_test
    #del pt, PvtTr

    features = [col for col in train_df.columns if col not in [
        "id", "sno", "max_sno", "cut", "installation_id", "game_session", "gs_starttime", "gs_endtime", "gs_duration",
        "timestamp", "accuracy_group", "jumps", "time_since_session_start", "session", "last_activity_2", "last_activity_3"]]
    print(f"total features: {len(features)}")

    for col in features:
    #print(col)
        if train_df[col].dtype == "O":
            train_df[col] = train_df[col].fillna("NONE")
            test_df[col] = test_df[col].fillna("NONE")
            le = LabelEncoder()
            le.fit(list(set(np.concatenate([train_df[col].unique(), test_df[col].unique()]))))
            train_df[col] = le.transform(train_df[col])
            test_df[col] = le.transform(test_df[col])

    
    print("finished processing features...")
    del train, test

    bagged_cv = []
    oof_df, te_df = pd.DataFrame({"id" : train_df.id.values}), pd.DataFrame({"id" : test_df.id.values})
    for i in range(N_BAG):
        seed = np.random.randint(0, 2000, size = 1)[0]
        print(f"==> training bag: {i+1} with seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        lgb_params.update({"feature_fraction_seed" : seed})
        folds = RandomGroupKFold_split(train_df.installation_id, seed=seed)
        
        trainer = GBDTTrainer(
            model_prefix = MODEL_PREFIX, 
            model_type= "LGB",
            params=lgb_params,
            save_optimized_cutoffs = False,
            eval_fn = QWK,
            apply_cutoffs_foldwise = False,
            apply_cutoffs_to_eval=False,
            feval = eval_qwk_lgb_regr,
            plot = False,
            splits=folds,
            features=features,  
            io_dir="./", submission_dir="./", 
            early_stop_rounds=400,
            verbose_freq=0, 
            feature_importance=False,
        )

        eval_preds, test_preds, cv = trainer.fit(train_df, test_df)
        bagged_cv.append(cv)
        oof_df = oof_df.merge(eval_preds.rename(columns = {MODEL_PREFIX : "bag_" + str(i+1)}), on = "id", how = "left")
        te_df = te_df.merge(test_preds.rename(columns = {MODEL_PREFIX : "bag_" + str(i+1)}), on = "id", how = "left")
        
    print(f"bagged-cv: {np.mean(bagged_cv)}")

    bag_cols = [col for col in oof_df.columns if col.startswith('bag')]
    oof_df["mean_oof"] = oof_df[bag_cols].mean(axis = 1)
    oof_df = oof_df.merge(train_df[["id","installation_id", "accuracy_group"]], on = "id", how = "left")
    yhat = oof_df.mean_oof.values
    y = oof_df.accuracy_group.values
    
    sol = minimize(QWK, [1, 1.5, 2], args= (yhat, y, ), method = 'Nelder-Mead', options = {"disp" : True, "xtol" : 1e-8})
    for i in range(3):
        sol = minimize(QWK, sol.x, args= (yhat, y, ), method = 'Nelder-Mead', tol = 1e-8, options = {"disp" : True, "xtol" : 1e-9})
        
    #yhat = np.digitize(oof_df.mean_oof.values, bins=sol.x)
    #class_names = np.array(['acc_0', 'acc_1', 'acc_2', 'acc_3'])
    # plot_confusion_matrix(y.astype(np.uint8), yhat.astype(np.uint8), 
    #                       classes=class_names,
    #                       normalize=True,
    #                       title='Confusion matrix with all(62) features')

    te_df["accuracy_group"] = te_df[bag_cols].mean(axis = 1)
    te_df["accuracy_group"] = np.digitize(te_df.accuracy_group.values, bins=sol.x)
    te_df = te_df[["id", "accuracy_group"]].rename(columns = {"id" : "installation_id"})
    te_df.to_csv(os.path.join(ROOT_DIR, "output", "submission.csv"), index = False)

        