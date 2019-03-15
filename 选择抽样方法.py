
#读取数据
df = pd.read_csv('train.csv')
#不需要id code
df = df.drop('ID_code',axis = 1)

#分测试训练集

#如需更改训练测试比例，请在此处更改

fraction = 0.7

df_train = df.sample(frac=fraction)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index,axis=0)
df_test = df.drop(df_train.index,axis=0)

#分开标签 (一般情况下不需要更改)
t_train = df_train['target']
x_train = df_train.drop('target',axis=1)

t_test = df_test['target']
x_test = df_test.drop('target',axis=1)

t_val = df_val['target']
x_val = df_val.drop('target',axis=1)

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN,SMOTETomek
from imblearn.under_sampling import ClusterCentroids


#测试抽样
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import cross_validate
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier

LogisticRegression = linear_model.LogisticRegression(solver='lbfgs',max_iter=500,tol=1e-3)
SGD = linear_model.SGDClassifier(loss="hinge", penalty="l2", max_iter=100,tol=1e-3)
RF = RandomForestClassifier(10)
GBD = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=10)
nn = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(100,50,50,20, 2),tol=1e-3)
nn2 = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(10, 2),tol=1e-3)

mds = [LogisticRegression,SGD,RF,GBD,nn,nn2]
mds_name = ['LogisticRegression','SGD','RF','GBD','nn','nn2']
sample_name = ['ClusterCentroids','SMOTEENN','SMOTETomek','SMOTE','SMOTE borderline1','SMOTE borderline2','ADASYN']
sample_methods = [ClusterCentroids(),SMOTEENN(),SMOTETomek(),SMOTE(),SMOTE(kind='borderline1'),SMOTE(kind='borderline2'),ADASYN()]


import time
sample_roc = []
i=0
for s in sample_methods:
    start_time = time.time()
    sample_x,sample_y = s.fit_sample(x_train, t_train)
    end_time = time.time()
    print('抽样方法: %s'%(sample_name[i]))
    print(('样本生成完成, 耗费时间：%s min')%(int((end_time - start_time)/60))) 
    print('\n')
    j=0
    sample_roc.append([])
    for model in mds:
        start_time = time.time()
        scores = cross_validate(model, sample_x,sample_y, cv=5, n_jobs = -1, return_estimator = True, scoring=('balanced_accuracy', 'recall', 'precision', 'roc_auc'), return_train_score=True)
        end_time = time.time()
        #对原浓度验证
        auc = []
        for model in scores['estimator']:
            try:
                auc.append(roc_auc_score(t_val,model.predict_proba(x_val)))   
            except:
                auc.append(roc_auc_score(t_val,model.predict(x_val)))   
        print(('cross vaildation完成， 耗费时间：%s min')  %(int((end_time - start_time)/60)))
        print('****************************************')
        print('抽样方法: %s'%(sample_name[i]))
        print(('模型：%s \n balanced_accuracy: %s \n recall: %s \n precision: %s \n roc_auc: %s')%(mds_name[j],np.average(scores['test_balanced_accuracy']),np.average(scores['test_recall']),np.average(scores['test_precision']),np.average(scores['test_roc_auc'])))
        print('****************************************')
        sample_roc[-1].append(np.average(auc))
        j+=1
        
    print('\n\n')

    i+=1    
    

    pd.DataFrame(sample_roc,columns=mds_name,index=sample_name)
