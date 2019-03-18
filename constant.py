from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from scipy.stats import uniform, randint
#修改或添加层时需要对主程序中的中间层进行修改
xgb = XGBClassifier()
lr = LogisticRegression()
nb = GaussianNB(priors=[0.9,0.1])
rf = RandomForestClassifier()
base_learners = [('lr',lr),('nb',nb)]

base_param_dicts = {
    'lr':
    {'C': randint(1,10)},
    
    'xgb':
    {'learning_rate': uniform(0.02, 0.2),
     'max_depth': randint(3, 20),
     'n_estimators': randint(100,1000,100)
     },
    'knn':
    {'n_neighbors': randint(1,11)},
    'rf':
    {'max_depth': randint(2, 5),
     'min_samples_split': randint(5, 20),
     'min_samples_leaf': randint(10, 20),
     'n_estimators': randint(50, 100),
     'max_features': uniform(0.6, 0.3)
    },
    #'svm':
    #{'C': randint(1,10),
     #'gamma': uniform(0.1,0.3)}
    } 

layer1_learners = [('lr',lr),('xgb',xgb)]
layer1_param_dicts = {
    'lr':
    {'C': randint(1,10)}
}

layer_list = [base_learners, layer1_learners]

meta_learner = [('rf',rf)]
meta_param_dicts = {'rf':
    {'max_depth': randint(2, 5),
     'min_samples_split': randint(5, 20),
     'min_samples_leaf': randint(10, 20),
     'n_estimators': randint(50, 100),
    'max_features': uniform(0.6, 0.3)
    }}