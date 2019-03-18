from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from scipy.stats import uniform, randint
from lightgbm import LGBMClassifier
#修改或添加层时需要对主程序中的中间层进行修改
xgb = XGBClassifier(n_estimator = 5000, scale_pos_weight = 0.11,objective = 'binary:logistic', eval_metric = 'auc')
lr = LogisticRegression()
nb = GaussianNB(priors=[0.9,0.1])
rf = RandomForestClassifier()
lgb = LGBMClassifier(application = 'binary', metric = 'auc')
base_learners = [('lr',lr),('nb',nb),('xgb',xgb),('lgb',lgb)]

base_param_dicts = {
    'lr':
    {'C': randint(1,10)},
    
    'xgb':
    {'learning_rate': uniform(0.02, 0.2),
     'max_depth': randint(3, 20),
     'sub_sample': uniform(0.75,1)
     'colsample_bytree':uniform(0.75,1)
     },
    
    'rf':
    {'max_depth': randint(2, 5),
     'min_samples_split': randint(5, 20),
     'min_samples_leaf': randint(10, 20),
     'n_estimators': randint(50, 100),
     'max_features': uniform(0.6, 0.3)
    },
    'lgb':
    {'learning_rate': uniform(0.02, 0.2)
    'feature_fraction': uniform(0.75,1)}
    } 

layer1_learners = [('lr',lr),('xgb',xgb)]
layer1_param_dicts = {
    'lr':
    {'C': randint(1,10)}
}

layer_list = [base_learners]

meta_learner = [('rf',rf)]
meta_param_dicts = {'rf':
    {'max_depth': randint(2, 5),
     'min_samples_split': randint(5, 20),
     'min_samples_leaf': randint(10, 20),
     'n_estimators': randint(50, 100),
    'max_features': uniform(0.6, 0.3)
    }}