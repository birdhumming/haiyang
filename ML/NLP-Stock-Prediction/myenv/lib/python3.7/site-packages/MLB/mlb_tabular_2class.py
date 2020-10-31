# coding: utf-8

"""
Machine Lerning Bruto-Force Attack
"""

# Author: Taketo Kimura <taketo_kimura@micin.jp>
# License: BSD 3 clause


import matplotlib.pyplot   as plt 
import matplotlib.cm       as cm
import numpy               as np
import math
import pandas              as pd
import sys
import os
import os.path
import sklearn
import sklearn.datasets
import matplotlib.gridspec as gridspec
from datetime                  import datetime
from datetime                  import timedelta
from dateutil.relativedelta    import relativedelta
from IPython.core.display      import display
from sklearn                   import datasets

from sklearn.svm               import SVC
from sklearn.linear_model      import Ridge, LogisticRegression
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from xgboost import XGBClassifier
from sklearn.preprocessing     import MinMaxScaler, StandardScaler
from sklearn.pipeline          import Pipeline
from sklearn.model_selection   import train_test_split
from sklearn.metrics           import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.decomposition     import PCA, KernelPCA as KPCA
from sklearn.manifold          import TSNE
from umap                      import UMAP
from sklearn.cluster           import KMeans
from scipy.sparse.csgraph      import connected_components
from sklearn.feature_selection import SelectKBest, f_classif, chi2, f_regression, SelectFromModel, RFE
#Rのglmを使用可能にするための関数
import statsmodels.formula.api as smf 
import lime
import lime.lime_tabular

# pandas display option (ajustable)
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 30)


class MLB:

    # 
    def __init__(self,               \
                 solver      = None, \
                 projection  = None):

        self.solver      = solver
        self.projection  = projection

    #
    def data_load_X(self,                  \
                    X_file_name    = None, \
                    X_header_row   = 0,    \
                    X_index_column = None, \
                    X_value_column = None):
        
        X_delimiter = ''
        if (X_file_name[-4:].lower() == '.csv'):
            X_delimiter = ','
        if (X_file_name[-4:].lower() == '.tsv'):
            X_delimiter = '\t'

        df_X = pd.read_csv(filepath_or_buffer = X_file_name,    \
                           header             = X_header_row,   \
                           index_col          = X_index_column, \
                           usecols            = X_value_column, \
                           delimiter          = X_delimiter)
        
        print('X data size is [%d, %d]' % (np.shape(df_X.values)[0], np.shape(df_X.values)[1]))
        return df_X.values, df_X.columns

    #
    def data_load_y(self,                  \
                    y_file_name    = None, \
                    y_header_row   = 0,    \
                    y_index_column = None, \
                    y_value_column = None):
        
        y_delimiter = ''
        if (y_file_name[-4:].lower() == '.csv'):
            y_delimiter = ','
        if (y_file_name[-4:].lower() == '.tsv'):
            y_delimiter = '\t'

        df_y = pd.read_csv(filepath_or_buffer = y_file_name,    \
                           header             = y_header_row,   \
                           index_col          = y_index_column, \
                           usecols            = y_value_column, \
                           delimiter          = y_delimiter)
        
        print('y data size is [%d, %d]' % (np.shape(df_y.values)[0], np.shape(df_y.values)[1]))
        return df_y.values, df_y.columns

    #     
    def data_load_Xy(self,                   \
                     Xy_file_name    = None, \
                     Xy_header_row   = 0,    \
                     Xy_index_column = None, \
                     X_value_column  = None, \
                     y_value_column  = None):
        
        if (Xy_file_name[-4:].lower() == '.csv'):
            Xy_delimiter = ','
        if (Xy_file_name[-4:].lower() == '.tsv'):
            Xy_delimiter = '\t'

        df_X = pd.read_csv(filepath_or_buffer = Xy_file_name,    \
                           header             = Xy_header_row,   \
                           index_col          = Xy_index_column, \
                           usecols            = X_value_column,  \
                           delimiter          = Xy_delimiter)
        
        df_y = pd.read_csv(filepath_or_buffer = Xy_file_name,    \
                           header             = Xy_header_row,   \
                           index_col          = Xy_index_column, \
                           usecols            = y_value_column,  \
                           delimiter          = Xy_delimiter)
        
        print('X data size is [%d, %d]' % (np.shape(df_X.values)[0], np.shape(df_X.values)[1]))
        print('y data size is [%d, %d]' % (np.shape(df_y.values)[0], np.shape(df_y.values)[1]))
        return df_X.values, df_X.columns, df_y.values, df_y.columns

    # 
    def check_X_and_y_status(self,                     \
                             X,                        \
                             y,                        \
                             X_item_name,              \
                             watch_corr_rank   = 20,   \
                             cross_plot        = True, \
                             cross_plot_assign = None):
        
        # calc pos and neg ratio
        X_pos = X[(y == 1).ravel(), :]
        y_pos = y[(y == 1).ravel(), :]
        
        X_neg = X[(y == 0).ravel(), :]
        y_neg = y[(y == 0).ravel(), :]

        print('Total data num    :%7d件' % len(X))
        print('Positive data num :%7d件 (%5.2f%%)' % (len(X_pos), (len(X_pos) / len(X) * 100)))
        print('Negative data num :%7d件 (%5.2f%%)' % (len(X_neg), (len(X_neg) / len(X) * 100)))
        print('')
        print('')

        # leakage and predictable check
        X_          = X - np.mean(X, axis=0)
        y_          = y - np.mean(y, axis=0)
        corr_with_y = np.dot(X_.T, y_).T / (np.sqrt(np.sum((X_ ** 2), axis=0)) * np.sqrt(np.sum((y_ ** 2), axis=0)))
        corr_with_y = np.abs(corr_with_y).ravel()

        sort_idx    = np.argsort(-corr_with_y)[:watch_corr_rank]

        fig = plt.figure(figsize=(12,8),dpi=100)

        ax = plt.subplot(2, 1, 1)
        plt.grid(which='major',color=[0.7, 0.7, 0.7],linestyle='-')
        if (len(corr_with_y) > 1000):
            width = 1.6
        else:
            width = 0.8
        plt.bar(np.arange(len(corr_with_y)), corr_with_y, width=width)
        plt.ylim(-0.05, 1.05)
        plt.xlim(-0.9, (len(corr_with_y) - 0.1))
        plt.title('correlation with X and y of all column')
        plt.ylabel('absolute correlation value')
        # plt.xlabel('column index')

        ax = plt.subplot(2, 1, 2)
        plt.grid(which='major',color=[0.7, 0.7, 0.7],linestyle='-')
        plt.bar(np.arange(watch_corr_rank), corr_with_y[sort_idx])
        plt.ylim(-0.05, 1.05)
        plt.xlim(-0.9, (watch_corr_rank - 0.1))
        plt.xticks(np.arange(watch_corr_rank))
        ax.set_xticklabels(X_item_name[sort_idx], rotation=90)
        plt.title('correlation with X and y (TOP%d)' % watch_corr_rank)
        plt.ylabel('absolute correlation value')

        print('leakage check')
        plt.show()
        print('')
        print('')

        if (cross_plot):
            print('cross plot')
            # cross plot, boxplot, histogram and ratio at range
            if (cross_plot_assign == None):
                cross_plot_assign = np.arange(X.shape[1])
            if (len(cross_plot_assign) > 50):
                cross_plot_assign = cross_plot_assign[:50]
                print('(save number of plot item [%d -> %d])' % (len(cross_plot_assing), 50))

            for item_i in cross_plot_assign:
                
                # get values of 1 column
                X_       = X[:, item_i]
                
                # sort for visibility
                sort_idx = np.argsort(X_)
                X_sort   = X_[sort_idx]
                y_sort   = y[sort_idx]
                X_sort_  = np.concatenate([X_sort[(y_sort == 0).ravel()], X_sort[(y_sort == 1).ravel()]])
                y_sort_  = np.concatenate([y_sort[(y_sort == 0).ravel()], y_sort[(y_sort == 1).ravel()]])
                
                # make histogram
                X_hist_info        = np.histogram(X_[(y == 0).ravel()], bins=20)
                X_hist_info[1][-1] = X_hist_info[1][-1] + 1e-10
                X_hist_0           = np.array(X_hist_info[0]) * 1.0
                
                bins_mean          = np.zeros(len(X_hist_info[1]) - 1)
                X_hist_1           = np.zeros(len(X_hist_info[1]) - 1)
                for bins_i in range(len(bins_mean)):
                    bins_mean[bins_i] = (X_hist_info[1][bins_i] + X_hist_info[1][bins_i + 1]) / 2
                    X_hist_1[bins_i]  = np.sum((X_hist_info[1][bins_i] <= X_[(y == 1).ravel()]) & (X_[(y == 1).ravel()] < X_hist_info[1][bins_i + 1]))
                
                # 各変数ごとのクロスプロットと、箱ヒゲ図を出力する
                fig  = plt.figure(figsize=(12,6),dpi=100)
                
                ax11 = plt.subplot(2, 2, 1)
                plt.grid(which='major',color=[0.7, 0.7, 0.7],linestyle='-')
                plt.scatter(np.arange(len(X_sort_)), X_sort_, alpha=0.1, label='x')
                plt.ylabel('x value')
                plt.ylabel('data index')
                plt.title('%s' % X_item_name[item_i])
                ax12 = ax11.twinx()  # 2つのプロットを関連付ける
                plt.scatter([0], [0], alpha=0.05, label='x')
                plt.scatter(np.arange(len(y_sort_)), y_sort_, alpha=0.1, label='y')
                plt.yticks([])
                plt.legend(loc='upper left', fontsize=8)
                
                ax21 = plt.subplot(2, 2, 2)
                plt.grid(which='major',color=[0.7, 0.7, 0.7],linestyle='-')
                plt.boxplot((X_[(y == 0).ravel()], X_[(y == 1).ravel()]))
                ax21.set_xticklabels(['y == 0', 'y == 1'], fontsize='small')
                plt.ylabel('x value')
                
                plt.subplot(2, 2, 3)
                plt.grid(which='major',color=[0.7, 0.7, 0.7],linestyle='-')
                plt.bar(bins_mean, X_hist_0, alpha=0.5, label='y == 0', width=((bins_mean[1] - bins_mean[0]) * 0.9))
                plt.bar(bins_mean, X_hist_1, alpha=0.5, label='y == 1', width=((bins_mean[1] - bins_mean[0]) * 0.9))
                plt.legend(loc='upper right', fontsize=8)
                plt.ylabel('frequency')
                plt.xlabel('x value')
                
                plt.subplot(2, 2, 4)
                plt.grid(which='major',color=[0.7, 0.7, 0.7],linestyle='-')
                plt.bar(bins_mean, (X_hist_0 / (X_hist_0 + X_hist_1 + 1e-10)), label='y == 0', width=((bins_mean[1] - bins_mean[0]) * 0.9))
                plt.bar(bins_mean, (X_hist_1 / (X_hist_0 + X_hist_1 + 1e-10)), bottom=(X_hist_0 / (X_hist_0 + X_hist_1 + 1e-10)), label='y == 1', width=((bins_mean[1] - bins_mean[0]) * 0.9))
                plt.legend(loc='upper right', fontsize=8)
                plt.ylabel('exist ratio')
                plt.xlabel('x value')
                
                print('---------------------------------------------')
                print('[%d]' % item_i)
                plt.show()
                print('')
                print('')
            
        return corr_with_y 

    #
    def X_regularization(self, X, min=None, max=None, eps=1e-20):

        if (min is None):
            min = np.min(X, axis=0)

        if (max is None):
            max = np.max(X, axis=0)

        return ((X - min) / (max - min + eps)), min, max

    # 
    def X_normalization(self, X, mean=None, std=None, eps=1e-20):

        if (mean is None):
            mean = np.mean(X, axis=0)

        if (std is None):
            std = np.std(X, axis=0)
        
        return ((X - mean) / (std + eps)), mean, std

    # 
    def machine_learning_bruteforce(self,                   \
                                    X,                      \
                                    y,                      \
                                    cv_num         = 5,     \
                                    cv_k_fold      = False, \
                                    cv_train_ratio = 0.7,   \
                                    log_param      = None,  \
                                    svc_param      = None,  \
                                    rf_param       = None,  \
                                    gb_param       = None,  \
                                    xg_param       = None,  \
                                    rfe_param      = None):
        
        # set pipeline
        pipe_line      = []
        pipe_name      = []

        if (log_param != None):
            for log_i in range(len(log_param)):
                pipe_line.append(LogisticRegression(random_state=0, penalty='l2', C=log_param[log_i]))
                pipe_name.append('Logistic(C=%f)' % log_param[log_i])
        if (svc_param != None):
            for svc_i in range(len(svc_param)):
                pipe_line.append(SVC(random_state=0, C=svc_param[svc_i], kernel='linear', probability=True))
                pipe_name.append('SVC(C=%f)' % svc_param[svc_i])
        if (rf_param != None):
            for rf_i in range(len(rf_param)):
                pipe_line.append(RandomForestClassifier(random_state=0, max_depth=rf_param[rf_i][0], n_estimators=rf_param[rf_i][1]))
                pipe_name.append('RandomForest(d=%d,e=%d)' % (rf_param[rf_i][0], rf_param[rf_i][1]))            
        if (gb_param != None):
            for gb_i in range(len(gb_param)):
                pipe_line.append(GradientBoostingClassifier(random_state=0, max_depth=gb_param[gb_i][0], n_estimators=gb_param[gb_i][1]))
                pipe_name.append('GradientBoosting(d=%d,e=%d)' % (gb_param[gb_i][0], gb_param[gb_i][1]))            
        if (xg_param != None):
            for xg_i in range(len(xg_param)):
                pipe_line.append(XGBClassifier(random_state=0, max_depth=xg_param[xg_i][0], n_estimators=xg_param[xg_i][1]))
                pipe_name.append('XGBClassifier(d=%d,e=%d)' % (xg_param[xg_i][0], xg_param[xg_i][1]))            

        pipe_num = len(pipe_line)
        print('pipeline num : %d' % pipe_num)
        print('')
        
        #
        measure_result   = []
        importance_stock = []
        remain_idx_stock = []

        # roop of algorithm and parameter
        for (pipe_i, model) in enumerate(pipe_line):

            measure_result_tmp,   \
            importance_stock_tmp, \
            remain_idx_stock_tmp, \
            rfe_param_,           \
            threshold = self.measure_model(X              = X,                 \
                                           y              = y,                 \
                                           model          = model,             \
                                           model_name     = pipe_name[pipe_i], \
                                           rfe_param      = rfe_param,         \
                                           random_seed    = 0,                 \
                                           cv_num         = cv_num,            \
                                           cv_k_fold      = cv_k_fold,         \
                                           cv_train_ratio = cv_train_ratio,    \
                                           threshold_num  = 21)
            
            measure_result.append(measure_result_tmp)
            importance_stock.append(importance_stock_tmp)
            remain_idx_stock.append(remain_idx_stock_tmp)
        
        return measure_result, importance_stock, remain_idx_stock, pipe_line, pipe_name, rfe_param_, threshold

    #
    def measure_model(self,                       \
                      X,                          \
                      y,                          \
                      model,                      \
                      model_name,                 \
                      rfe_param           = None, \
                      random_seed         = 0,    \
                      cv_num              = 5,    \
                      cv_k_fold           = True, \
                      cv_train_ratio      = 0.7,  \
                      threshold_num       = 21,   \
                      watch_predict_ratio = 0.2):

        # prepare variable for measuring on result
        threshold     = np.arange(threshold_num) / (threshold_num - 1)

        # 
        if (rfe_param == None):
            rfe_param_ = np.array([np.shape(X)[1]])
        else:
            rfe_param_ = np.unique(np.concatenate([rfe_param, [np.shape(X)[1]]]))
            rfe_param_ = rfe_param_[rfe_param_ <= np.shape(X)[1]]
            rfe_param_ = np.abs(np.sort(-rfe_param_))

        # var for result of measuring
        rfe_num       = len(rfe_param_)
        TP_num        = np.empty([rfe_num, 2, cv_num, threshold_num]) # number of True Positive
        FP_num        = np.empty([rfe_num, 2, cv_num, threshold_num]) # number of False Positive
        TN_num        = np.empty([rfe_num, 2, cv_num, threshold_num]) # number of True Negative
        FN_num        = np.empty([rfe_num, 2, cv_num, threshold_num]) # number of False Negative
        TPR           = np.empty([rfe_num, 2, cv_num, threshold_num]) # True Positive Rate（TP / (TP + FN)）：Recall
        TNR           = np.empty([rfe_num, 2, cv_num, threshold_num]) # True Negative Rate（TN / (TN + FP)）
        ACC           = np.empty([rfe_num, 2, cv_num, threshold_num]) # Accuracy（(TN + TN) / (TP + FP + TN + FN)）
        ACC_pos       = np.empty([rfe_num, 2, cv_num, threshold_num]) # Accuracy of Positive（TP / (TP + FP + TN + FN)）
        ACC_neg       = np.empty([rfe_num, 2, cv_num, threshold_num]) # Accuracy of Negative（TN / (TP + FP + TN + FN)）

        # var for result of importance
        importance_stock = np.zeros([rfe_num, cv_num, np.shape(X)[1]], dtype='float32')
        remain_idx_stock = np.ones([rfe_num, np.shape(X)[1]], dtype='bool')

        # set random seed
        np.random.seed(random_seed)

        ###################################################################
        # display processing progress 
        process_num      = cv_num # set number of process
        process_break    = np.round(np.linspace(1, process_num, 50)) 
        watch_idx = np.random.permutation(process_num)[:int(np.round(watch_predict_ratio * process_num))] 
        ###################################################################
        
        for rfe_i in range(rfe_num):

            if (rfe_i == 0):
                X_ = X
            else:
                importance_mean  = np.mean(importance_stock[(rfe_i - 1), :, :], axis=0)
                threshold_remain = importance_mean[np.argsort(-importance_mean)[rfe_param_[rfe_i]]]
                remain_idx       = importance_mean > threshold_remain
                X_               = X[:, remain_idx]
                #
                remain_idx_stock[rfe_i, :] = remain_idx

            ###############################################################
            process_i        = 0  
            str_now          = datetime.now()  
            print('train on cross valid (model:%s, RFE:%d) [start time is %s]' % (model_name, rfe_param_[rfe_i], str_now)) 
            print('--------------------------------------------------')
            print('START                                          END') 
            print('----+----1----+----2----+----3----+----4----+----5') 
            ###############################################################
            
            if (cv_k_fold):
                cv_k_fold_range     = np.round(np.linspace(0, len(X), (cv_num + 1))).astype('int32')
                cv_k_fold_range[0]  = 0
                cv_k_fold_range[-1] = len(X)

            # loop of cross valid
            for cv_i in range(cv_num):
            
                ###########################################################
                # update processing progress
                process_i = process_i + 1   
                if (sum(process_break == process_i) > 0):
                    for print_i in range(sum(process_break == process_i)): 
                        print('*', end='')                              
                ###########################################################
                
                if (cv_k_fold):
                    X_train   = np.zeros([0, np.shape(X_)[1]])
                    y_train   = np.zeros([0, 1])
                    
                    # prepare train data and test data
                    for k_fold_i in range(cv_num):
                        
                        if (k_fold_i == cv_i):
                            X_test  = X_[cv_k_fold_range[k_fold_i]:cv_k_fold_range[k_fold_i + 1], :]
                            y_test  = y[cv_k_fold_range[k_fold_i]:cv_k_fold_range[k_fold_i + 1]]
                        else:
                            X_train = np.concatenate([X_train, X_[cv_k_fold_range[k_fold_i]:cv_k_fold_range[k_fold_i + 1], :]], axis=0)
                            y_train = np.concatenate([y_train, y[cv_k_fold_range[k_fold_i]:cv_k_fold_range[k_fold_i + 1]]], axis=0)
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X_, y, train_size=cv_train_ratio, shuffle=True, random_state=cv_i)
                
                # learning
                model.fit(X_train, y_train)
                
                # 
                if hasattr(model, 'coef_'):
                    importance_tmp = np.abs(model.coef_)
                else:
                    importance_tmp = model.feature_importances_

                if (rfe_i == 0):
                    remain_idx = np.ones(np.shape(X)[1], 'bool')
                else:
                    remain_idx = remain_idx_stock[rfe_i, :]
                importance_stock[rfe_i, cv_i, remain_idx] = importance_tmp
                
                # predict probability (positive)
                y_hat_train = model.predict_proba(X_train).T[1]
                y_hat_test  = model.predict_proba(X_test).T[1]
                
                # adjust dim
                y_train_    = y_train.ravel()
                y_test_     = y_test.ravel()
                
                ############################################################
                # plot predict result
                if (np.sum(watch_idx == cv_i) > 0):
                    fig = plt.figure(figsize=(12, 2), dpi=100)

                    # sort for visualization
                    sort_idx         = np.argsort(y_hat_train)
                    y_hat_train_sort = y_hat_train[sort_idx]
                    y_train_sort     = y_train_[sort_idx]
                    y_hat_train_sort = np.concatenate([y_hat_train_sort[(y_train_sort == 0).ravel()], y_hat_train_sort[(y_train_sort == 1).ravel()]])
                    y_train_sort     = np.concatenate([y_train_sort[(y_train_sort == 0).ravel()], y_train_sort[(y_train_sort == 1).ravel()]])

                    plt.subplot(1, 2, 1)
                    plt.grid(which='major', color=[0.7, 0.7, 0.7], linestyle='-')
                    plt.scatter(np.arange(len(y_train_sort)), y_train_sort, alpha=np.min([0.1, (100 / len(y_train_sort))]), label='actual')
                    plt.scatter(np.arange(len(y_hat_train_sort)), y_hat_train_sort, alpha=np.min([0.1, (100 / len(y_hat_train_sort))]),  label='predict')
                    plt.scatter([(sum(y_train_sort == 0) / 2), (sum(y_train_sort == 0) + (sum(y_train_sort == 1) / 2))], 
                                [np.mean(y_hat_train_sort[(y_train_sort == 0).ravel()]), np.mean(y_hat_train_sort[(y_train_sort == 1).ravel()])], alpha=0.5, s=100, label='mean of predict')
                    plt.legend(loc='upper left', fontsize=8)
                    plt.title('data select case [%d / %d]\nlabel value and predict prob \n at train data' % ((cv_i + 1), process_num))
                    plt.ylabel('label value and predict prob \n at train data')
                    plt.xlabel('data index (sorted by correct label)\n[diff of mean = %.3f]' % np.abs(np.mean(y_hat_train_sort[(y_train_sort == 0).ravel()]) - np.mean(y_hat_train_sort[(y_train_sort == 1).ravel()])))

                    # sort for visualization
                    sort_idx        = np.argsort(y_hat_test)
                    y_hat_test_sort = y_hat_test[sort_idx]
                    y_test_sort     = y_test_[sort_idx]
                    y_hat_test_sort = np.concatenate([y_hat_test_sort[(y_test_sort == 0).ravel()], y_hat_test_sort[(y_test_sort == 1).ravel()]])
                    y_test_sort     = np.concatenate([y_test_sort[(y_test_sort == 0).ravel()], y_test_sort[(y_test_sort == 1).ravel()]])

                    plt.subplot(1, 2, 2)
                    plt.grid(which='major', color=[0.7, 0.7, 0.7], linestyle='-')
                    plt.scatter(np.arange(len(y_test_sort)), y_test_sort, alpha=np.min([0.1, (100 / len(y_test_sort))]), label='actual')
                    plt.scatter(np.arange(len(y_hat_test_sort)), y_hat_test_sort, alpha=np.min([0.1, (100 / len(y_hat_test_sort))]), label='predict')
                    plt.scatter([(sum(y_test_sort == 0) / 2), (sum(y_test_sort == 0) + (sum(y_test_sort == 1) / 2))], 
                                [np.mean(y_hat_test_sort[(y_test_sort == 0).ravel()]), np.mean(y_hat_test_sort[(y_test_sort == 1).ravel()])], alpha=0.5, s=100, label='mean of predict')
                    plt.legend(loc='upper left', fontsize=8)
                    plt.title('label value and predict prob\nat test data')
                    plt.xlabel('data index (sorted by correct label)\n[diff of mean = %.3f]' % np.abs(np.mean(y_hat_test_sort[(y_test_sort == 0).ravel()]) - np.mean(y_hat_test_sort[(y_test_sort == 1).ravel()])))
                ############################################################
                
                # loop of threshold
                for threshold_i in range(threshold_num):
                    
                    # classification by threshold
                    y_hat_train_ = np.round(y_hat_train + 0.5 - threshold[threshold_i])
                    y_hat_test_  = np.round(y_hat_test  + 0.5 - threshold[threshold_i])
                    
                    # measuring practical estimate
                    # train
                    TP_num[rfe_i, 0, cv_i, threshold_i]  = sum(y_hat_train_[y_hat_train_ == 1] == y_train_[y_hat_train_ == 1])
                    FP_num[rfe_i, 0, cv_i, threshold_i]  = sum(y_hat_train_[y_hat_train_ == 1] != y_train_[y_hat_train_ == 1])
                    TN_num[rfe_i, 0, cv_i, threshold_i]  = sum(y_hat_train_[y_hat_train_ == 0] == y_train_[y_hat_train_ == 0])
                    FN_num[rfe_i, 0, cv_i, threshold_i]  = sum(y_hat_train_[y_hat_train_ == 0] != y_train_[y_hat_train_ == 0])
                    TPR[rfe_i, 0, cv_i, threshold_i]     = TP_num[rfe_i, 0, cv_i, threshold_i] / (TP_num[rfe_i, 0, cv_i, threshold_i] + FN_num[rfe_i, 0, cv_i, threshold_i] + 1e-20)
                    TNR[rfe_i, 0, cv_i, threshold_i]     = TN_num[rfe_i, 0, cv_i, threshold_i] / (TN_num[rfe_i, 0, cv_i, threshold_i] + FP_num[rfe_i, 0, cv_i, threshold_i] + 1e-20)
                    ACC[rfe_i, 0, cv_i, threshold_i]     = (TP_num[rfe_i, 0, cv_i, threshold_i] + TN_num[rfe_i, 0, cv_i, threshold_i]) / (TP_num[rfe_i, 0, cv_i, threshold_i] + FP_num[rfe_i, 0, cv_i, threshold_i] + TN_num[rfe_i, 0, cv_i, threshold_i] + FN_num[rfe_i, 0, cv_i, threshold_i] + 1e-20)
                    ACC_pos[rfe_i, 0, cv_i, threshold_i] = TP_num[rfe_i, 0, cv_i, threshold_i] / (TP_num[rfe_i, 0, cv_i, threshold_i] + FP_num[rfe_i, 0, cv_i, threshold_i] + TN_num[rfe_i, 0, cv_i, threshold_i] + FN_num[rfe_i, 0, cv_i, threshold_i] + 1e-20)
                    ACC_neg[rfe_i, 0, cv_i, threshold_i] = TN_num[rfe_i, 0, cv_i, threshold_i] / (TP_num[rfe_i, 0, cv_i, threshold_i] + FP_num[rfe_i, 0, cv_i, threshold_i] + TN_num[rfe_i, 0, cv_i, threshold_i] + FN_num[rfe_i, 0, cv_i, threshold_i] + 1e-20)
                    # test
                    TP_num[rfe_i, 1, cv_i, threshold_i]  = sum(y_hat_test_[y_hat_test_ == 1] == y_test_[y_hat_test_ == 1])
                    FP_num[rfe_i, 1, cv_i, threshold_i]  = sum(y_hat_test_[y_hat_test_ == 1] != y_test_[y_hat_test_ == 1])
                    TN_num[rfe_i, 1, cv_i, threshold_i]  = sum(y_hat_test_[y_hat_test_ == 0] == y_test_[y_hat_test_ == 0])
                    FN_num[rfe_i, 1, cv_i, threshold_i]  = sum(y_hat_test_[y_hat_test_ == 0] != y_test_[y_hat_test_ == 0])
                    TPR[rfe_i, 1, cv_i, threshold_i]     = TP_num[rfe_i, 1, cv_i, threshold_i] / (TP_num[rfe_i, 1, cv_i, threshold_i] + FN_num[rfe_i, 1, cv_i, threshold_i] + 1e-20)
                    TNR[rfe_i, 1, cv_i, threshold_i]     = TN_num[rfe_i, 1, cv_i, threshold_i] / (TN_num[rfe_i, 1, cv_i, threshold_i] + FP_num[rfe_i, 1, cv_i, threshold_i] + 1e-20)
                    ACC[rfe_i, 1, cv_i, threshold_i]     = (TP_num[rfe_i, 1, cv_i, threshold_i] + TN_num[rfe_i, 1, cv_i, threshold_i]) / (TP_num[rfe_i, 1, cv_i, threshold_i] + FP_num[rfe_i, 1, cv_i, threshold_i] + TN_num[rfe_i, 1, cv_i, threshold_i] + FN_num[rfe_i, 1, cv_i, threshold_i] + 1e-20)
                    ACC_pos[rfe_i, 1, cv_i, threshold_i] = TP_num[rfe_i, 1, cv_i, threshold_i] / (TP_num[rfe_i, 1, cv_i, threshold_i] + FP_num[rfe_i, 1, cv_i, threshold_i] + TN_num[rfe_i, 1, cv_i, threshold_i] + FN_num[rfe_i, 1, cv_i, threshold_i] + 1e-20)
                    ACC_neg[rfe_i, 1, cv_i, threshold_i] = TN_num[rfe_i, 1, cv_i, threshold_i] / (TP_num[rfe_i, 1, cv_i, threshold_i] + FP_num[rfe_i, 1, cv_i, threshold_i] + TN_num[rfe_i, 1, cv_i, threshold_i] + FN_num[rfe_i, 1, cv_i, threshold_i] + 1e-20)

            ACC_train_tmp = ACC[rfe_i, 0, :, :]
            ACC_test_tmp  = ACC[rfe_i, 1, :, :]

            ###############################################################
            print('\nmean [max accuracy of sevral threshold] of cross valid = [train:%.3f, test:%.3f]' % (np.mean(np.max(ACC_train_tmp, axis=1)), np.mean(np.max(ACC_test_tmp, axis=1))))
            plt.show()
            ###############################################################
        
        measure_result = []
        measure_result.append(TP_num)
        measure_result.append(FP_num)
        measure_result.append(TN_num)
        measure_result.append(FN_num)
        measure_result.append(TPR)
        measure_result.append(TNR)
        measure_result.append(ACC)
        measure_result.append(ACC_pos)
        measure_result.append(ACC_neg)
        return measure_result, importance_stock, remain_idx_stock, rfe_param_, threshold

    # 
    def plot_feature_selected_result(self,               \
                                     remain_idx_stock,   \
                                     importance_stock,   \
                                     X_item_name = None, \
                                     watch_rank  = 20):

        if (X_item_name is None):
            X_item_name_tmp = np.arange(np.sum(exp_var_column_idx))
            X_item_name     = [('ExpVar' + str(num)) for num in X_item_name_tmp]
    
        importance_stock_mean = np.mean(importance_stock, axis=1)

        plt.figure(figsize=(12,12),dpi=200)

        ax = plt.subplot(3, 2, 1)
        plt.grid(which='major', color=[0.7, 0.7, 0.7], linestyle='-')
        plt.imshow(remain_idx_stock.astype('uint8'), interpolation='nearest', aspect='auto', vmin=0, vmax=1)
        plt.colorbar()
        plt.title('feature selected result [0, 1]')
        plt.ylabel('measure exec index\n(feature selection num)')
        plt.xlabel('column index')

        ax = plt.subplot(3, 2, 2)
        plt.grid(which='major', color=[0.7, 0.7, 0.7], linestyle='-')
        plt.imshow(importance_stock_mean, interpolation='nearest', aspect='auto', vmin=np.min(importance_stock), vmax=np.max(importance_stock))
        plt.colorbar()
        plt.title('importance of several learning')
        plt.ylabel('measure exec index\n(feature selection num)')
        plt.xlabel('column index')
        
        ax = plt.subplot(3, 2, 3)
        if (np.shape(remain_idx_stock)[1] > 1000):
            width = 1.6
        else:
            width = 1
        plt.grid(which='major', color=[0.7, 0.7, 0.7], linestyle='-')
        plt.bar(np.arange(np.shape(remain_idx_stock)[1]), np.mean(remain_idx_stock.astype('float32'), axis=0), width=width)
        plt.ylabel('mean of feature selected or not')
        plt.xlabel('column index')

        ax = plt.subplot(3, 2, 4)
        plt.grid(which='major', color=[0.7, 0.7, 0.7], linestyle='-')
        plt.bar(np.arange(np.shape(importance_stock_mean)[1]), np.mean(importance_stock_mean, axis=0), width=width)
        plt.ylabel('mean of importance')
        plt.xlabel('column index')

        sort_idx = np.argsort(np.mean(-remain_idx_stock.astype('float32'), axis=0))[:watch_rank]

        ax = plt.subplot(3, 2, 5)
        plt.grid(which='major', color=[0.7, 0.7, 0.7], linestyle='-')
        plt.bar(np.arange(watch_rank), np.mean(remain_idx_stock.astype('float32'), axis=0)[sort_idx])
        plt.xticks(np.arange(watch_rank))
        ax.set_xticklabels(X_item_name[sort_idx], rotation=90, fontsize='small')
        plt.ylabel('mean of feature selected or not')
        plt.xlabel('column index')

        sort_idx = np.argsort(np.mean(-np.abs(importance_stock_mean), axis=0))[:watch_rank]

        ax = plt.subplot(3, 2, 6)
        plt.grid(which='major', color=[0.7, 0.7, 0.7], linestyle='-')
        plt.bar(np.arange(watch_rank), np.mean(importance_stock_mean, axis=0)[sort_idx])
        plt.xticks(np.arange(watch_rank))
        ax.set_xticklabels(X_item_name[sort_idx], rotation=90, fontsize='small')
        plt.ylabel('mean of importance')
        plt.xlabel('column index')

        plt.show()
    
    #
    def plot_ACC(self,           \
                 measure_result, \
                 pipe_name,      \
                 rfe_param,      \
                 threshold):
        #
        pipe_num = len(measure_result)
        rfe_num  = len(rfe_param)

        # plot ACC (mean of cross validation)
        print('plot ACC (mean of cross validation)')

        fig = plt.figure(figsize=(12, 12), dpi=100)

        plt.subplot(2, 1, 1)
        plt.grid(which='major', color=[0.7, 0.7, 0.7], linestyle='-')

        for pipe_i in range(pipe_num):

            ACC_tmp = measure_result[pipe_i][6]
            
            for rfe_i in range(rfe_num):
                plt.plot(threshold, np.mean(ACC_tmp[rfe_i, 0, :, :], axis=0), alpha=0.7, label=('%s RFE:%d' % (pipe_name[pipe_i], rfe_param[rfe_i])), linestyle='-', color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
                plt.scatter(threshold, np.mean(ACC_tmp[rfe_i, 0, :, :], axis=0), alpha=0.7, color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
                plt.plot(threshold, np.std(ACC_tmp[rfe_i, 0, :, :], axis=0), alpha=0.05, linewidth=5, color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
        plt.ylim(-0.05, 1.05)
        plt.xlabel('threshold (Predict value is under thredhold -> Judge to [0:Negative], Predict value is over threshold -> Judge to [1:Positive])')
        plt.ylabel('ACC of Train Data \n(mean of cross_valid[%d])' % np.shape(ACC_tmp)[0])

        plt.subplot(2, 1, 2)
        plt.grid(which='major', color=[0.7, 0.7, 0.7], linestyle='-')

        for pipe_i in range(pipe_num):

            ACC_tmp = measure_result[pipe_i][6]
            
            for rfe_i in range(rfe_num):
                plt.plot(threshold, np.mean(ACC_tmp[rfe_i, 1, :, :], axis=0), alpha=0.7, label=('%s RFE:%d' % (pipe_name[pipe_i], rfe_param[rfe_i])), linestyle='-', color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
                plt.scatter(threshold, np.mean(ACC_tmp[rfe_i, 1, :, :], axis=0), alpha=0.7, color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
                plt.plot(threshold, np.std(ACC_tmp[rfe_i, 1, :, :], axis=0), alpha=0.05, linewidth=5, color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
        plt.ylim(-0.05, 1.05)
        plt.xlabel('threshold (Predict value is under thredhold -> Judge to [0:Negative], Predict value is over threshold -> Judge to [1:Positive])')
        plt.ylabel('ACC of Test Data \n(mean of cross_valid[%d])' % np.shape(ACC_tmp)[0])

        plt.show()

    #
    def plot_ACC_pos(self,           \
                     measure_result, \
                     pipe_name,      \
                     rfe_param,      \
                     threshold):
        #
        pipe_num = len(measure_result)
        rfe_num  = len(rfe_param)

        # 
        print('plot ACC of Positive (mean of cross validation)')

        fig = plt.figure(figsize=(12, 12), dpi=100)

        plt.subplot(2, 1, 1)
        plt.grid(which='major', color=[0.7, 0.7, 0.7], linestyle='-')

        for pipe_i in range(pipe_num):

            ACC_pos_tmp = measure_result[pipe_i][7]
            
            for rfe_i in range(rfe_num):
                plt.plot(threshold, np.mean(ACC_pos_tmp[rfe_i, 0, :, :], axis=0), alpha=0.7, label=('%s RFE:%d' % (pipe_name[pipe_i], rfe_param[rfe_i])), linestyle='-', color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
                plt.scatter(threshold, np.mean(ACC_pos_tmp[rfe_i, 0, :, :], axis=0), alpha=0.7, color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
                plt.plot(threshold, np.std(ACC_pos_tmp[rfe_i, 0, :, :], axis=0), alpha=0.05, linewidth=5, color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
        plt.ylim(-0.05, 1.05)
        plt.xlabel('threshold (Predict value is under thredhold -> Judge to [0:Negative], Predict value is over threshold -> Judge to [1:Positive])')
        plt.ylabel('ACC of Train Data \n(mean of cross_valid[%d])' % np.shape(ACC_pos_tmp)[0])

        plt.subplot(2, 1, 2)
        plt.grid(which='major', color=[0.7, 0.7, 0.7], linestyle='-')

        for pipe_i in range(pipe_num):

            ACC_pos_tmp = measure_result[pipe_i][7]
            
            for rfe_i in range(rfe_num):
                plt.plot(threshold, np.mean(ACC_pos_tmp[rfe_i, 1, :, :], axis=0), alpha=0.7, label=('%s RFE:%d' % (pipe_name[pipe_i], rfe_param[rfe_i])), linestyle='-', color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
                plt.scatter(threshold, np.mean(ACC_pos_tmp[rfe_i, 1, :, :], axis=0), alpha=0.7, color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
                plt.plot(threshold, np.std(ACC_pos_tmp[rfe_i, 1, :, :], axis=0), alpha=0.05, linewidth=5, color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
        plt.ylim(-0.05, 1.05)
        plt.xlabel('threshold (Predict value is under thredhold -> Judge to [0:Negative], Predict value is over threshold -> Judge to [1:Positive])')
        plt.ylabel('ACC of Test Data \n(mean of cross_valid[%d])' % np.shape(ACC_pos_tmp)[0])

        plt.show()

    #
    def plot_ACC_neg(self,           \
                     measure_result, \
                     pipe_name,      \
                     rfe_param,      \
                     threshold):
        #
        pipe_num = len(measure_result)
        rfe_num  = len(rfe_param)

        # 
        print('plot ACC of Negtive (mean of cross validation)')

        fig = plt.figure(figsize=(12, 12), dpi=100)

        plt.subplot(2, 1, 1)
        plt.grid(which='major', color=[0.7, 0.7, 0.7], linestyle='-')

        for pipe_i in range(pipe_num):

            ACC_neg_tmp = measure_result[pipe_i][8]
            
            for rfe_i in range(rfe_num):
                plt.plot(threshold, np.mean(ACC_neg_tmp[rfe_i, 0, :, :], axis=0), alpha=0.7, label=('%s RFE:%d' % (pipe_name[pipe_i], rfe_param[rfe_i])), linestyle='-', color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
                plt.scatter(threshold, np.mean(ACC_neg_tmp[rfe_i, 0, :, :], axis=0), alpha=0.7, color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
                plt.plot(threshold, np.std(ACC_neg_tmp[rfe_i, 0, :, :], axis=0), alpha=0.05, linewidth=5, color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
        plt.ylim(-0.05, 1.05)
        plt.xlabel('threshold (Predict value is under thredhold -> Judge to [0:Negative], Predict value is over threshold -> Judge to [1:Positive])')
        plt.ylabel('ACC of Train Data \n(mean of cross_valid[%d])' % np.shape(ACC_neg_tmp)[0])

        plt.subplot(2, 1, 2)
        plt.grid(which='major', color=[0.7, 0.7, 0.7], linestyle='-')

        for pipe_i in range(pipe_num):

            ACC_neg_tmp = measure_result[pipe_i][8]
            
            for rfe_i in range(rfe_num):
                plt.plot(threshold, np.mean(ACC_neg_tmp[rfe_i, 1, :, :], axis=0), alpha=0.7, label=('%s RFE:%d' % (pipe_name[pipe_i], rfe_param[rfe_i])), linestyle='-', color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
                plt.scatter(threshold, np.mean(ACC_neg_tmp[rfe_i, 1, :, :], axis=0), alpha=0.7, color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
                plt.plot(threshold, np.std(ACC_neg_tmp[rfe_i, 1, :, :], axis=0), alpha=0.05, linewidth=5, color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
        plt.ylim(-0.05, 1.05)
        plt.xlabel('threshold (Predict value is under thredhold -> Judge to [0:Negative], Predict value is over threshold -> Judge to [1:Positive])')
        plt.ylabel('ACC of Test Data \n(mean of cross_valid[%d])' % np.shape(ACC_neg_tmp)[0])

        plt.show()

    #
    def plot_TPR_neg(self,           \
                     measure_result, \
                     pipe_name,      \
                     rfe_param,      \
                     threshold):
        #
        pipe_num = len(measure_result)
        rfe_num  = len(rfe_param)

        # 
        print('plot True Positive Rate (mean of cross validation)')

        fig = plt.figure(figsize=(12, 12), dpi=100)

        plt.subplot(2, 1, 1)
        plt.grid(which='major', color=[0.7, 0.7, 0.7], linestyle='-')

        for pipe_i in range(pipe_num):

            TPR_tmp = measure_result[pipe_i][4]
            
            for rfe_i in range(rfe_num):
                plt.plot(threshold, np.mean(TPR_tmp[rfe_i, 0, :, :], axis=0), alpha=0.7, label=('%s RFE:%d' % (pipe_name[pipe_i], rfe_param[rfe_i])), linestyle='-', color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
                plt.scatter(threshold, np.mean(TPR_tmp[rfe_i, 0, :, :], axis=0), alpha=0.7, color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
                plt.plot(threshold, np.std(TPR_tmp[rfe_i, 0, :, :], axis=0), alpha=0.05, linewidth=5, color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
        plt.ylim(-0.05, 1.05)
        plt.xlabel('threshold (Predict value is under thredhold -> Judge to [0:Negative], Predict value is over threshold -> Judge to [1:Positive])')
        plt.ylabel('TPR of Train Data \n(mean of cross_valid[%d])' % np.shape(TPR_tmp)[0])

        plt.subplot(2, 1, 2)
        plt.grid(which='major', color=[0.7, 0.7, 0.7], linestyle='-')

        for pipe_i in range(pipe_num):

            TPR_tmp = measure_result[pipe_i][4]
            
            for rfe_i in range(rfe_num):
                plt.plot(threshold, np.mean(TPR_tmp[rfe_i, 1, :, :], axis=0), alpha=0.7, label=('%s RFE:%d' % (pipe_name[pipe_i], rfe_param[rfe_i])), linestyle='-', color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
                plt.scatter(threshold, np.mean(TPR_tmp[rfe_i, 1, :, :], axis=0), alpha=0.7, color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
                plt.plot(threshold, np.std(TPR_tmp[rfe_i, 1, :, :], axis=0), alpha=0.05, linewidth=5, color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
        plt.ylim(-0.05, 1.05)
        plt.xlabel('threshold (Predict value is under thredhold -> Judge to [0:Negative], Predict value is over threshold -> Judge to [1:Positive])')
        plt.ylabel('TPR of Test Data \n(mean of cross_valid[%d])' % np.shape(TPR_tmp)[0])

        plt.show()

    #
    def plot_TNR_neg(self,           \
                     measure_result, \
                     pipe_name,      \
                     rfe_param,      \
                     threshold):
        #
        pipe_num = len(measure_result)
        rfe_num  = len(rfe_param)

        # 
        print('plot True Negative Rate (mean of cross validation)')

        fig = plt.figure(figsize=(12, 12), dpi=100)

        plt.subplot(2, 1, 1)
        plt.grid(which='major', color=[0.7, 0.7, 0.7], linestyle='-')

        for pipe_i in range(pipe_num):

            TNR_tmp = measure_result[pipe_i][5]
            
            for rfe_i in range(rfe_num):
                plt.plot(threshold, np.mean(TNR_tmp[rfe_i, 0, :, :], axis=0), alpha=0.7, label=('%s RFE:%d' % (pipe_name[pipe_i], rfe_param[rfe_i])), linestyle='-', color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
                plt.scatter(threshold, np.mean(TNR_tmp[rfe_i, 0, :, :], axis=0), alpha=0.7, color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
                plt.plot(threshold, np.std(TNR_tmp[rfe_i, 0, :, :], axis=0), alpha=0.05, linewidth=5, color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
        plt.ylim(-0.05, 1.05)
        plt.xlabel('threshold (Predict value is under thredhold -> Judge to [0:Negative], Predict value is over threshold -> Judge to [1:Positive])')
        plt.ylabel('TNR of Train Data \n(mean of cross_valid[%d])' % np.shape(TNR_tmp)[0])

        plt.subplot(2, 1, 2)
        plt.grid(which='major', color=[0.7, 0.7, 0.7], linestyle='-')

        for pipe_i in range(pipe_num):

            TNR_tmp = measure_result[pipe_i][5]
            
            for rfe_i in range(rfe_num):
                plt.plot(threshold, np.mean(TNR_tmp[rfe_i, 1, :, :], axis=0), alpha=0.7, label=('%s RFE:%d' % (pipe_name[pipe_i], rfe_param[rfe_i])), linestyle='-', color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
                plt.scatter(threshold, np.mean(TNR_tmp[rfe_i, 1, :, :], axis=0), alpha=0.7, color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
                plt.plot(threshold, np.std(TNR_tmp[rfe_i, 1, :, :], axis=0), alpha=0.05, linewidth=5, color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
        plt.ylim(-0.05, 1.05)
        plt.xlabel('threshold (Predict value is under thredhold -> Judge to [0:Negative], Predict value is over threshold -> Judge to [1:Positive])')
        plt.ylabel('TNR of Test Data \n(mean of cross_valid[%d])' % np.shape(TNR_tmp)[0])

        plt.show()

    # 
    def plot_ROC_and_AUC(self,           \
                         measure_result, \
                         pipe_name,      \
                         rfe_param,      \
                         threshold):

        #
        pipe_num = len(measure_result)
        rfe_num  = len(rfe_param)

        # 
        print('plot ROC and AUC (mean of cross validation)')

        # plot ROC curve (mean of cross validation)
        fig = plt.figure(figsize=(12, 16), dpi=100)

        # plot Area Under the Curve of ROC (mean of holdout selection and cross validation)
        AUC = np.zeros([2, pipe_num, rfe_num])

        for pipe_i in range(pipe_num):

            TPR_tmp  = measure_result[pipe_i][4]
            TNR_tmp  = measure_result[pipe_i][5]
            
            for rfe_i in range(rfe_num):
                
                # train
                plt.subplot(2, 2, 1)
                plt.grid(which='major', color=[0.7, 0.7, 0.7], linestyle='-')

                TPR_tmp_ = np.mean(TPR_tmp[rfe_i, 0, :, :], axis=0)
                TNR_tmp_ = np.mean(TNR_tmp[rfe_i, 0, :, :], axis=0)
                
                plt.plot(TPR_tmp_, TNR_tmp_, alpha=0.7, label=('%s RFE:%d' % (pipe_name[pipe_i], rfe_param[rfe_i])), linestyle='-', color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
                plt.scatter(TPR_tmp_, TNR_tmp_, alpha=0.7, color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
                plt.xlabel('True Positive Rate of Train Data \n(mean of cross_valid[%d])' % np.shape(TPR_tmp)[0])
                plt.ylabel('True Negative Rate of Train Data \n(mean of cross_valid[%d])' % np.shape(TPR_tmp)[0])
                plt.legend(loc='lower left', fontsize=5)

                AUC_tmp = np.zeros(len(threshold) - 1)
                
                for threshold_i in range(len(threshold) - 1):
                    # height of the trapezoid of a threshold pitch
                    width        = np.abs(TPR_tmp_[threshold_i] - TPR_tmp_[threshold_i + 1]) # 一応ABS
                    # upper base of the trapezoid of a threshold pitch
                    height_right = TNR_tmp_[threshold_i]
                    # bottom base of the trapezoid of a threshold pitch
                    height_left  = TNR_tmp_[threshold_i + 1]
                    # calc area
                    AUC_tmp[threshold_i] = width * (height_right + height_left) / 2
                
                AUC[0, pipe_i, rfe_i] = np.sum(AUC_tmp)
                    
                plt.subplot(2, 2, 2)
                plt.grid(which='major',color=[0.7, 0.7, 0.7],linestyle='-')
                plt.bar((pipe_i * rfe_num + rfe_i), AUC[0, pipe_i, rfe_i], label=('%s RFE:%d' % (pipe_name[pipe_i], rfe_param[rfe_i])), color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
                plt.ylim(-0.05, 1.05)
                plt.xlabel('Learning Model index')
                plt.ylabel('AUC of Train Data \n(mean of cross_valid[%d])' % np.shape(TPR_tmp)[0])
                
                # test
                plt.subplot(2, 2, 3)
                plt.grid(which='major', color=[0.7, 0.7, 0.7], linestyle='-')

                TPR_tmp_ = np.mean(TPR_tmp[rfe_i, 1, :, :], axis=0)
                TNR_tmp_ = np.mean(TNR_tmp[rfe_i, 1, :, :], axis=0)
                
                plt.plot(TPR_tmp_, TNR_tmp_, alpha=0.7, label=('%s RFE:%d' % (pipe_name[pipe_i], rfe_param[rfe_i])), linestyle='-', color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
                plt.scatter(TPR_tmp_, TNR_tmp_, alpha=0.7, color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
                plt.xlabel('True Positive Rate of Test Data \n(mean of cross_valid[%d])' % np.shape(TPR_tmp)[0])
                plt.ylabel('True Negative Rate of Test Data \n(mean of cross_valid[%d])' % np.shape(TPR_tmp)[0])
                plt.legend(loc='lower left', fontsize=5)

                AUC_tmp = np.zeros(len(threshold) - 1)
                
                for threshold_i in range(len(threshold) - 1):
                    # height of the trapezoid of a threshold pitch
                    width        = np.abs(TPR_tmp_[threshold_i] - TPR_tmp_[threshold_i + 1]) # 一応ABS
                    # upper base of the trapezoid of a threshold pitch
                    height_right = TNR_tmp_[threshold_i]
                    # bottom base of the trapezoid of a threshold pitch
                    height_left  = TNR_tmp_[threshold_i + 1]
                    # calc area
                    AUC_tmp[threshold_i] = width * (height_right + height_left) / 2
                
                AUC[1, pipe_i, rfe_i] = np.sum(AUC_tmp)
                    
                plt.subplot(2, 2, 4)
                plt.grid(which='major',color=[0.7, 0.7, 0.7],linestyle='-')
                plt.bar((pipe_i * rfe_num + rfe_i), AUC[1, pipe_i, rfe_i], label=('%s RFE:%d' % (pipe_name[pipe_i], rfe_param[rfe_i])), color=cm.hsv((pipe_i * rfe_num + rfe_i)/(pipe_num * rfe_num * 1.1)))
                plt.ylim(-0.05, 1.05)
                plt.xlabel('Learning Model index')
                plt.ylabel('AUC of Test Data \n(mean of cross_valid[%d])' % np.shape(TPR_tmp)[0])

        plt.show()
        print('AUC is ...')
        print(AUC)
        return AUC

    #
    def lime_optimize_and_result_view(self,                                                        \
                                      training_data,                                               \
                                      feature_names,                                               \
                                      training_labels,                                             \
                                      model,                                                       \
                                      show_lime_result_num   = 20,                                 \
                                      categorical_features   = None,                               \
                                      categorical_names      = None,                               \
                                      class_names            = np.array(['Negative', 'Positive']), \
                                      labels                 = (1,),                               \
                                      mode                   = "classification",                   \
                                      feature_selection      = 'auto',                             \
                                      num_features           = 5,                                  \
                                      kernel_width_candi     = [None],                             \
                                      kernel                 = None,                               \
                                      discretize_continuous  = True,                               \
                                      discretizer            = 'quartile',                         \
                                      sample_around_instance = False,                              \
                                      top_labels             = None,                               \
                                      num_samples            = 1000,                               \
                                      model_regressor        = None,                               \
                                      distance_metric        = 'euclidean',                        \
                                      verbose                = False,                              \
                                      random_state           = None):

        model.fit(training_data, training_labels)
        importance_rank = np.argsort(-model.feature_importances_)
        y_hat           = model.predict_proba(training_data).T[1]

        grid_score      = np.zeros([len(kernel_width_candi)])

        ###################################################################
        # display processing progress 
        process_num     = show_lime_result_num # set number of process
        process_break   = np.round(np.linspace(1, process_num, 50)) 
        process_i       = 0  
        str_now         = datetime.now()  
        print('search good kernel width [start time is %s]' % str_now); 
        print('--------------------------------------------------'); 
        print('START                                          END'); 
        print('----+----1----+----2----+----3----+----4----+----5'); 
        ###################################################################

        np.random.seed(random_state)
        show_lime_result_idx = np.random.permutation(np.arange(np.shape(training_data)[0]))[:show_lime_result_num]

        for kernel_width_i in range(len(kernel_width_candi)):

            # LIMEを実施するための事前情報取得メソッド実施
            explainer = lime.lime_tabular.LimeTabularExplainer(training_data=training_data, 
                                                            feature_names=feature_names, 
                                                            class_names=class_names, 
                                                            discretize_continuous=discretize_continuous, 
                                                            discretizer=discretizer, 
                                                            training_labels=training_labels, 
                                                            random_state=random_state, 
                                                            kernel_width=kernel_width_candi[kernel_width_i])
            
            for show_lime_result_i in show_lime_result_idx:

                explain = explainer.explain_instance(data_row=training_data[show_lime_result_i], 
                                                    predict_fn=model.predict_proba, 
                                                    num_features=num_features, 
                                                    num_samples=num_samples)
                grid_score[kernel_width_i] += explain.score

                ###########################################################
                # update processing progress
                process_i = process_i + 1   
                if (sum(process_break == process_i) > 0):
                    for print_i in range(sum(process_break == process_i)): 
                        print('*', end='')                              
                ###########################################################

            grid_score[kernel_width_i] /= show_lime_result_num
            print(' -> grid_score[(%.2f)] = %.6f' % (kernel_width_candi[kernel_width_i], grid_score[kernel_width_i]))

            ###############################################################
            # reset processing progress
            process_i = 0                              
            ###############################################################
        
        kernel_width = kernel_width_candi[np.where(grid_score == np.max(grid_score))]
        print('set %.2f to kernel_width' % kernel_width) 

        # LIMEを実施するための事前情報取得メソッド実施
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=training_data, 
                                                        feature_names=feature_names, 
                                                        class_names=class_names, 
                                                        discretize_continuous=discretize_continuous, 
                                                        discretizer=discretizer, 
                                                        training_labels=training_labels, 
                                                        random_state=random_state, 
                                                        kernel_width=kernel_width)
        
        for show_lime_result_i in show_lime_result_idx:

            explain = explainer.explain_instance(data_row=training_data[show_lime_result_i], 
                                                predict_fn=model.predict_proba, 
                                                num_features=num_features, 
                                                num_samples=num_samples)
            
            print('')
            print('-------------------------------------------------------------')
            print('data row                        = %s' % show_lime_result_i)
            print('bias of local model             = %s' % explain.intercept[1])
            for feature_i in range(num_features):
                print('')
                print('feature name[%3d]               = %s [Importance Rank:%d]' % (feature_i, feature_names[explain.local_exp[1][feature_i][0]], np.where(importance_rank == explain.local_exp[1][feature_i][0])[0]+1))
                print('feature value[%4d][%3d]        = %s'    % (show_lime_result_i, explain.local_exp[1][feature_i][0], training_data[show_lime_result_i][explain.local_exp[1][feature_i][0]]))
                print('discretize means of local model = %s'   % np.round(explainer.discretizer.means[explain.local_exp[1][feature_i][0]], 3))
                print('coef of local model[%3d]        = %.5f' % (explain.local_exp[1][feature_i][0], explain.local_exp[1][feature_i][1]))
            print('')
            print('predict by local model          = %.5f' % explain.local_pred[0])
            # print('R2 score of local model         = %.5f' % explain.score)
            print('acutual label[%4d]             = %.5f' % (show_lime_result_i, training_labels[show_lime_result_i]))
            print('predict by global model[%4d]   = %.5f' % (show_lime_result_i, y_hat[show_lime_result_i]))
            print('')
            
            explain.show_in_notebook(show_table=True, show_all=False)
            explain.as_pyplot_figure()
            lime_target = model.predict_proba(explain.scaled_data) 

            plt.show()
            print('\n\n')

    # 
    def understandable_visualize(self,                  \
                                 X,                     \
                                 y,                     \
                                 model         = PCA(), \
                                 visualize_dim = [1, 2]):
        
        X_norm, X_mean, X_std = self.X_normalization(X=X)

        # model.fit(X_norm)
        # X_proj = model.transform(X_norm)
        X_proj = model.fit_transform(X_norm)

        X_proj_dim1_pos = X_proj[(y == 1).ravel(), (visualize_dim[0] - 1)]
        X_proj_dim2_pos = X_proj[(y == 1).ravel(), (visualize_dim[1] - 1)]
        X_proj_dim1_neg = X_proj[(y == 0).ravel(), (visualize_dim[0] - 1)]
        X_proj_dim2_neg = X_proj[(y == 0).ravel(), (visualize_dim[1] - 1)]

        # plot PCA
        fig = plt.figure(figsize=(12,8),dpi=100)
        gs  = gridspec.GridSpec(9,9)

        plt.subplot(gs[:6, :6])
        plt.grid(which='major',color=[0.7, 0.7, 0.7],linestyle='-')
        plt.scatter(X_proj_dim1_neg, X_proj_dim2_neg, alpha=0.5)
        plt.scatter(X_proj_dim1_pos, X_proj_dim2_pos, alpha=0.5)
        plt.title('principal component')
        plt.xlabel('pc%d' % visualize_dim[0])
        plt.ylabel('pc%d' % visualize_dim[1])

        # plot histogram
        plt.subplot(gs[-2:, :6])
        plt.grid(which='major',color=[0.7, 0.7, 0.7],linestyle='-')
        plt.hist(X_proj_dim1_neg, alpha=0.5)
        plt.hist(X_proj_dim1_pos, alpha=0.5)
        plt.xlabel('pc%d value' % visualize_dim[0])

        plt.subplot(gs[:6, -2:])
        plt.grid(which='major',color=[0.7, 0.7, 0.7],linestyle='-')
        plt.hist(X_proj_dim2_neg, alpha=0.5, orientation="horizontal")
        plt.hist(X_proj_dim2_pos, alpha=0.5, orientation="horizontal")
        plt.xlabel('pc%d value' % visualize_dim[0])
        plt.gca().invert_xaxis()

        print('2dim visualization')

    #
    def kmeans_classification(self, \
                              X,    \
                              y,    \
                              k = 10):
        
        #
        X_norm, X_mean, X_std = self.X_normalization(X=X)

        #
        kmeans        = KMeans(n_clusters=k, random_state=0, init='k-means++')
        kmeans_result = kmeans.fit(X_norm)

        # get cluster index
        kmeans_label  = kmeans_result.labels_

        data_num_of_cluster = np.zeros(k)
        pos_num_of_cluster  = np.zeros(k)
        neg_num_of_cluster  = np.zeros(k)

        # loop of kmeans cluster
        for kmeans_label_i in range(k):
            X_of_cluster                        = X_norm[(kmeans_label == kmeans_label_i), :]
            y_of_cluster                        = y[(kmeans_label == kmeans_label_i)]
            data_num_of_cluster[kmeans_label_i] = len(y_of_cluster)
            pos_num_of_cluster[kmeans_label_i]  = np.sum(y_of_cluster)
            neg_num_of_cluster[kmeans_label_i]  = len(y_of_cluster) - np.sum(y_of_cluster)

        pos_ratio_of_cluster = pos_num_of_cluster / (neg_num_of_cluster + pos_num_of_cluster)
        neg_ratio_of_cluster = neg_num_of_cluster / (neg_num_of_cluster + pos_num_of_cluster)

        sort_idx = np.argsort(neg_ratio_of_cluster)
            
        fig = plt.figure(figsize=(12,12),dpi=100)
        plt.subplot(2, 1, 1)
        plt.grid(which='major',color=[0.7, 0.7, 0.7],linestyle='-')
        plt.bar(np.arange(k), neg_num_of_cluster[sort_idx])
        plt.bar(np.arange(k), pos_num_of_cluster[sort_idx], bottom=neg_num_of_cluster[sort_idx])
        plt.xlabel('cluster index')
        plt.ylabel('number of data')
        plt.title('K[%d]-means separate' % k)

        plt.subplot(2, 1, 2)
        plt.grid(which='major',color=[0.7, 0.7, 0.7],linestyle='-')
        plt.bar(np.arange(k), neg_ratio_of_cluster[sort_idx])
        plt.bar(np.arange(k), pos_ratio_of_cluster[sort_idx], bottom=neg_ratio_of_cluster[sort_idx])
        plt.xlabel('cluster index')
        plt.ylabel('ratio of positive and negative')

        plt.show()

    #
    def knn_classification(self, \
                           X,    \
                           y,    \
                           k = 10):
        
        #
        X_norm, X_mean, X_std = self.X_normalization(X=X)

        # calc pos and neg ratio
        X_pos = X_norm[(y == 1).ravel(), :]
        y_pos = y[(y == 1).ravel(), :]
        
        X_neg = X_norm[(y == 0).ravel(), :]
        y_neg = y[(y == 0).ravel(), :]

        #
        pos_num = len(X_pos) # positive data num (If the number of data is large, specify a constant)

        # acquire peripheral data centered on positive data
        neighbor_pos_num   = np.zeros(pos_num)
        neighbor_neg_num   = np.zeros(pos_num)
        neighbor_pos_ratio = np.zeros(pos_num)
        neighbor_neg_ratio = np.zeros(pos_num)

        pos_idx            = np.array(np.where(y == 1)[0])
        pos_idx            = np.random.permutation(pos_idx)

        # loop of positive data
        for pos_i in range(pos_num):
            X_diff_L2_tmp             = np.sqrt( ((X_norm[pos_idx[pos_i], :] - X_norm)**2).sum(axis=1) ) # calc Euclid distance
            neighbor_idx              = np.argsort(X_diff_L2_tmp)
            neighbor_pos_num[pos_i]   = np.sum(y[neighbor_idx[0:(k + 1)]] == 1) - 1 # subtract 1 for the number of data of itself -> number of positive data in close distance
            neighbor_neg_num[pos_i]   = k - neighbor_pos_num[pos_i]
            neighbor_pos_ratio[pos_i] = neighbor_pos_num[pos_i] / k
            neighbor_neg_ratio[pos_i] = neighbor_neg_num[pos_i] / k

        sort_idx = np.argsort(neighbor_neg_num)    
            
        fig = plt.figure(figsize=(12,12),dpi=100)
        plt.subplot(2, 1, 1)
        plt.grid(which='major',color=[0.7, 0.7, 0.7],linestyle='-')
        plt.bar(np.arange(pos_num), neighbor_neg_num[sort_idx])
        plt.bar(np.arange(pos_num), neighbor_pos_num[sort_idx], bottom=neighbor_neg_num[sort_idx])
        plt.xlabel('positive data index')
        plt.ylabel('ratio of positive and negative')
        plt.title('K[%d]-nearest neighbor (positive data based)' % k)

        plt.subplot(2, 1, 2)
        plt.grid(which='major',color=[0.7, 0.7, 0.7],linestyle='-')
        plt.bar(np.arange(pos_num), neighbor_neg_ratio[sort_idx])
        plt.bar(np.arange(pos_num), neighbor_pos_ratio[sort_idx], bottom=neighbor_neg_ratio[sort_idx])
        plt.xlabel('positive data index')
        plt.ylabel('ratio of positive and negative')

        neg_num = len(X_neg) # negative data num (If the number of data is large, specify a constant)

        # acquire peripheral data centered on positive data
        neighbor_pos_num   = np.zeros(neg_num)
        neighbor_neg_num   = np.zeros(neg_num)
        neighbor_pos_ratio = np.zeros(neg_num)
        neighbor_neg_ratio = np.zeros(neg_num)

        neg_idx            = np.array(np.where(y == 0)[0])
        neg_idx            = np.random.permutation(neg_idx)

        # loop of positive data
        for neg_i in range(neg_num):
            X_diff_L2_tmp             = np.sqrt( ((X_norm[neg_idx[neg_i], :] - X_norm)**2).sum(axis=1) ) # calc Euclid distance
            neighbor_idx              = np.argsort(X_diff_L2_tmp)
            neighbor_neg_num[neg_i]   = np.sum(y[neighbor_idx[0:(k + 1)]] == 0) - 1 # subtract 1 for the number of data of itself -> number of positive data in close distance
            neighbor_pos_num[neg_i]   = k - neighbor_neg_num[neg_i]
            neighbor_neg_ratio[neg_i] = neighbor_neg_num[neg_i] / k
            neighbor_pos_ratio[neg_i] = neighbor_pos_num[neg_i] / k

        sort_idx = np.argsort(-neighbor_neg_num)    
            
        fig = plt.figure(figsize=(12,12),dpi=100)
        plt.subplot(2, 1, 1)
        plt.grid(which='major',color=[0.7, 0.7, 0.7],linestyle='-')
        plt.bar(np.arange(neg_num), neighbor_neg_num[sort_idx])
        plt.bar(np.arange(neg_num), neighbor_pos_num[sort_idx], bottom=neighbor_neg_num[sort_idx])
        plt.xlabel('negative data index')
        plt.ylabel('ratio of positive and negative')
        plt.title('K[%d]-nearest neighbor (negative data based)' % k)

        plt.subplot(2, 1, 2)
        plt.grid(which='major',color=[0.7, 0.7, 0.7],linestyle='-')
        plt.bar(np.arange(neg_num), neighbor_neg_ratio[sort_idx])
        plt.bar(np.arange(neg_num), neighbor_pos_ratio[sort_idx], bottom=neighbor_neg_ratio[sort_idx])
        plt.xlabel('positive data index')
        plt.ylabel('ratio of positive and negative')

        plt.show()

        

