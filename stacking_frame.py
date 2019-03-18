import numpy as np
import pandas as pd
import mlens 
from datetime import datetime
from sklearn import metrics

# Data viz
from mlens.visualization import corr_X_y, corrmat

# Model evaluation
from mlens.metrics import make_scorer
from mlens.model_selection import Evaluator
from sklearn.model_selection import cross_val_score
# Ensemble
from mlens.ensemble import SuperLearner
from scipy.stats import uniform, randint
#%matplotlib inline
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import preprocessing as pp
import feature_selection as fs
import constant

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

def single_model_test(base_learners, X, y, X_pred,y_pred):
    ''' 基学习器效果概览'''
    P = np.zeros((X_pred.shape[0], len(base_learners)))
    P = pd.DataFrame(P, columns=[e for e, _ in base_learners])
    base_learner = base_learners
    scores = []
    for est_name, est in base_learners:
        est.fit(X, y)
        p = est.predict_proba(X_pred)
        p_class = est.predict(X_pred)
        P.loc[:, est_name] = p[:,1]
        score = cross_val_score(est, X, y, cv=5, scoring = 'roc_auc')
        scores.append(score)
        print("%3s : %.4f" % (est_name, score.mean()))
        print(metrics.classification_report(y_pred, p_class))
    df_single_model = pd.DataFrame({'base_learner':base_learner, 'score':scores})
    return df_single_model, P

def plot_roc_curve (ytest, P_base_learners, labels):
    ''' 基础层roc绘图'''
    plt.figure(figsize = (10,8))
    plt.plot([0,1], [0,1], 'k--')
    cm = [plt.cm.rainbow(i)
         for i in np.linspace(0,1.0, P_base_learners.shape[1]+1)]
    
    for i in range(P_base_learners.shape[1]):
        p = P_base_learners[:, i]
        fpr, tpr, _=roc_curve(ytest, p)
        plt.plot(fpr, tpr, label = labels[i], c=cm[i + 1])
    
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(frameon=False)
    plt.savefig('model_roc.png')
    plt.show()

def base_hyperparam_tuning (X,y,base_learners, param_dicts, n_iterations = 100):
    '''基层模型超参数调节，当前评估指标为auc'''
    X = X.values
    y = y.values
    scorer = make_scorer(metrics.roc_auc_score, greater_is_better=True)
    evl = Evaluator(scorer, cv=5, verbose = 20, backend= 'multiprocessing')
    evl.fit(X, y, estimators=base_learners, param_dicts=param_dicts, n_iter = n_iterations)
    df_params = pd.DataFrame(evl.results)
    return df_params

def layer_hyperparam_tuning(X,y,pre_layer_learners, local_layer_learners, param_dicts_layer, n_iterations = 50, pre_params = 'params_base.csv'):
    '''中间层超参数调节，加入需按顺序'''
    X = X.values
    y = y.values
    scorer = make_scorer(metrics.roc_auc_score, greater_is_better=True)
    params_pre = pd.read_csv(pre_params)
    params_pre.set_index(['Unnamed: 0'], inplace = True)
    for case_name, params in params_pre["params"].items():
        case_est = case_name
        params = eval(params)
        for est_name, est in pre_layer_learners:
            if est_name == case_est:
                est.set_params(**params)
    in_layer = SuperLearner(folds = 10, backend= 'multiprocessing', model_selection=True)
    in_layer.add(pre_layer_learners,proba=True)
    preprocess = [in_layer]
    evl = Evaluator(scorer,cv=5,verbose = 20,backend= 'multiprocessing')
    evl.fit(X, y, local_layer_learners, param_dicts = param_dicts_layer, preprocessing={'meta': preprocess},n_iter=n_iterations)
    df_params_layer = pd.DataFrame(evl.results)
    return in_layer, df_params_layer

def stacking_training (X,y,X_pred,layer_list,meta_learner):
    stacking_in_layer = SuperLearner(folds = 5, backend= 'multiprocessing', model_selection=False)
    for each in layer_list:
        stacking_in_layer.add(each,proba=True)
        print ('基学习器添加成功')
    stacking_in_layer.add_meta(meta_learner,proba= True)
    print ('元学习器添加成功')
    print ('拟合中')
    stacking_in_layer.fit(X,y)
    pred_proba = stacking_in_layer.predict_proba(X_pred)
    return pred_proba,stacking_in_layer


def main():
    starter_time = timer(None)
    df_total = pd.read_csv('train.csv', index_col = None, header = 0, memory_map = True)
    df_total = df_total.drop(['ID_code'],axis = 1)
    df_total = df_total.sample(1000)
    df_total.index = range(len(df_total))
    frame_train,frame_test = pp.train_test_split(df_total, 'target', 0.3)
    frame_train = pp.normalization(frame_train,'target')
    frame_test = pp.normalization(frame_test,'target')
    X = frame_train.drop(['target'], axis = 1)
    y = frame_train['target']
    X_pred = frame_test.drop(['target'],axis = 1)
    y_truth = frame_test['target']
    print ('数据读入完成')
    base_learners = constant.base_learners
    print ('基学习器载入完成，开始训练基学习器')
    df_single_output, P = single_model_test(base_learners, X, y, X_pred,y_truth)
    plot_roc_curve (y_truth, P.values, list(P.columns))
    print ('基学习器训练完成，开始调节参数')
    base_param_dicts = constant.base_param_dicts
    df_params_base = base_hyperparam_tuning (X,y,base_learners, base_param_dicts, n_iterations = 50)
    df_params_base.to_csv('params_base.csv')
    print ('参数调节完成，开始训练中间层')
    layer1_learners = constant.layer1_learners
    
    layer1_param_dicts = constant.layer1_param_dicts
    print ('开始为中间层调参')
    
    #in_layer_1, df_params_1 = layer_hyperparam_tuning(X,y,pre_layer_learners=base_learners, local_layer_learners = layer1_learners, param_dicts_layer = layer1_param_dicts, n_iterations = 50, pre_params = 'params_base.csv')
    #df_params_1.to_csv('params1.csv')
    
    
    
    
    #设定学习器参数并确定元学习器
    print ('开始训练元学习器')
    meta_learner = constant.meta_learner
    meta_param_dicts = constant.meta_param_dicts
    meta_layer_model, df_params_meta = layer_hyperparam_tuning(X,y,pre_layer_learners = layer1_learners, local_layer_learners = meta_learner, param_dicts_layer = meta_param_dicts, n_iterations = 50, pre_params = 'params_base.csv')
    df_params_meta.to_csv('paramsMeta.csv')
    params_pre = pd.read_csv('paramsMeta.csv')
    params_pre.set_index(['Unnamed: 0'], inplace = True)
    for case_name, params in params_pre["params"].items():
        case_est = case_name
        params = eval(params)
        for est_name, est in meta_learner:
            if est_name == case_est:
                est.set_params(**params)
    layer_list = constant.layer_list
    pred_proba_1 ,stacking_model = stacking_training(X,y,X_pred,layer_list = layer_list,meta_learner = meta_learner)
    print (roc_auc_score(y_truth, pred_proba_1[:,1]))
    timer(starter_time)
    return pred_proba_1, stacking_model

if __name__ == '__main__':
    df_total_test = pd.read_csv('test.csv)
    df_test_X = df_total_test
    pred_proba_1,stacking_model = main()
    prediction = stacking_model.predict_proba(df_test_X.values)
    pd.DataFrame(prediction).to_csv('output.csv')
    


