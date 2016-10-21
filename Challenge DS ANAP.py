import numpy as np 
import pandas as pd 
import sklearn as sk 
from collections import defaultdict, Counter 
from datetime import date
data = pd.read_csv('C:/work/challenge_ds/data2.csv', sep=';')
data_test = pd.read_csv('C:/work/challenge_ds/test2.csv', sep=';')


#split the data into 3 groups 
#train data
data_with=data[data['Nombre de séjours/séances MCO des patients en ALD'] !=0]
data_eq=data_with[data_with['Nombre de séjours/séances MCO des patients en ALD']== data_with['Nombre total de séjours/séances']]
data_sup = data_with[data_with['Nombre total de séjours/séances'] > data_with['Nombre de séjours/séances MCO des patients en ALD']]
#test data
test_with=data_test[data_test['Nombre de séjours/séances MCO des patients en ALD']!=0]
test_nul=data_test[data_test['Nombre de séjours/séances MCO des patients en ALD'] == 0]
test_eq = test_with[test_with['Nombre de séjours/séances MCO des patients en ALD'] == test_with['Nombre total de séjours/séances']]
test_sup=test_with[test_with['Nombre de séjours/séances MCO des patients en ALD'] < test_with['Nombre total de séjours/séances']]

#Dict of Departements in France
dict_Region = {'idf': ['77-','75-','92-','91-','93-','95-','94-','78-'], 'Als' : ['67-','68-'],
                   'Aqu' : ['24-','33-','40-','47-','64-'],  'auv': ['03-','15-','43-','63-'], 
                   'Bnor' : ['14-','50-','61-'] , 'bourg' : ['21-','58-','71-','89-'], 
                   'Bre' : ['22-', '29-', '35-', '56-'], 'Cent' : ['18-', '28-', '36-', '37-', '41-', '45-'], 
                   'Cha' : ['08-', '10-', '51-', '52-'], 'OutrMer' : ['971','972','973','974','976'] , 
                   'ColMer' : ['984','986','987','988','975'],'Cors' : ['2A-','2B-'] , 
                   'FraC' : ['25-', '39-', '70-', '90-'] , 'Hnor' : ['27-', '76-'] , 
                   'Lrous' : ['66-', '48-', '34-', '30-', '11-'] ,  'lim' : ['19-', '23-', '87-'] , 
                   'Lor' : ['54-', '55-', '57-', '88-'], 'MPyr' : ['09-', '12-', '31-', '32-', '46-', '65-','81-','82-'], 
                   'NCal' : ['59-', '62-'] , 'Loir' : ['44-', '49-', '53-', '72-', '85-'] , 'Pic' : ['02-', '60-', '80-'], 
                   'Pit' : ['16-', '17-', '79-', '86-'], 'Alp' : ['04-', '05-', '06-', '13-', '83-', '84-'] , 
                   'Rhalp'  : ['01-', '07-', '26-', '38-', '42-', '69-', '73-', '74-']}

# feature engineering for both train and test data
# functions 
def convertingREG(dataset, Conv_Dict):
    
    for key, value in Conv_Dict.items():
        for x in value :
            dataset.loc[:,'dep'].replace(x, key, inplace=True)       
    return dataset

def calculate_age(year):
    today = date.today()
    return today.year - year

def trait_don(dataset) : 
    dataset['ENS'] = dataset.loc[:,'Raison sociale'].map(lambda x : x[0:2])
    dataset.loc[:,'ENS'].replace(['LE','UN'],['H\xc3','\xc3\x89'], inplace=True)
    dataset.loc[:,'ENS'].replace(['H\xc3','P\xc3','\xc3\x89','R\xc3'],['H3','P3','X3','R3'],inplace=True)
    dataset['dep'] = dataset.loc[:,'Provenance des patients (département)'].map(lambda x : x[0:3])
    dataset['domAct']=dataset.loc[:,'Domaines d activités'].map(lambda x : x[0:3])
    dataset.loc[:,'âge (deux classes >75 ans, <= 75 ans)'].replace(['<=75 ans','>75 ans'],['C1','C2'], inplace=True)
    dataset['annee']=dataset['annee'].map(calculate_age)
    convertingREG(dataset,dict_Region)
    del dataset['Finess']
    del dataset['id']
    del dataset['Raison sociale']
    del dataset['Domaines d activités']
    del dataset['Provenance des patients (département)']
    return dataset
def renam_col_train(dataset) : 
    mylist=list(dataset.columns)
    mylist[4]='age'
    mylist[5]='nbsej'
    mylist[6]='totSej'
    dataset.columns=mylist
    return dataset
def renam_col_test(dataset):
    mylist=list(dataset.columns)
    mylist[5]='age'
    mylist[6]='nbsej'
    mylist[7]='totSej'
    dataset.columns=mylist
    return dataset

Max_values = 100
def select_values_dum(dataset,lis_to_dum) : 
    dummy_values={}
    for feat in lis_to_dum :
        values = [value for (value,_) in Counter(dataset[feat]).most_common(Max_values)]
        dummy_values[feat]=values
    return dummy_values
def D_encode_df(dataset):
    for (feat,dummy_values) in dum_values.items() : 
        for dv in dummy_values : 
            dummy_name ='{}_{}'.format(feat,dv)
            dataset[dummy_name] = (dataset[feat] == dv).astype(float)
        del dataset[feat]
        print('Dummy done for %s' % feat)
        
def scaleD(dataset, listnum) :
    for x in listnum : 
        scale = dataset[x].mean()
        shift = dataset[x].std()
        if scale == 0. :
            del dataset[x]
        else : 
            dataset[x] = (dataset[x] - shift).astype(np.float64)/scale
            print('done for %s' %x)
        
def splitting(dataset,pr) :
    from sklearn.cross_validation import train_test_split
    train,test = train_test_split(dataset,train_size=0.7)
    train_X = train.drop('cible1',axis=1)
    test_X = test.drop('cible1',axis=1)
    train_Y = np.array(train['cible1'])
    test_Y = np.array(test['cible1'])
    return train_X,train_Y,test_X,test_Y


#Step 2  : Using the functions 

trait_don(data_eq)
trait_don(test_eq)
renam_col_train(data_eq)
renam_col_test(test_eq)

trait_don(data_sup)
trait_don(test_sup)
renam_col_train(data_sup)
renam_col_test(test_sup)

# features to dummy
dummy_features =['age','ENS','dep','domAct']

dum_values = select_values_dum(data_eq,dummy_features)
D_encode_df(data_eq)
D_encode_df(test_eq)

dum_values = select_values_dum(data_sup,dummy_features)
D_encode_df(data_sup)
D_encode_df(test_sup)

# features to scale 

num_feat=['nbsej','totSej','annee']

scaleD(data_eq,num_feat)
scaleD(test_eq,num_feat)

scaleD(data_sup,num_feat)
scaleD(test_sup,num_feat)

# splitting data 

train_X1,train_Y1,test_X1,test_Y1 = splitting(data_eq,0.7)
train_X2,train_Y2,test_X2,test_Y2 = splitting(data_sup,0.7)

# function to calculate the root mean squared error (RMSE)

from sklearn.metrics import mean_squared_error
from math import sqrt

def rmse(Yact, Ypred) :
    return sqrt(mean_squared_error(Yact, Ypred))

# training models 

#Linear regression
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(train_X1, train_Y1)
prd_regr=regr.predict(test_X1)

print(rmse(test_Y1, prd_regr))

from sklearn import datasets, linear_model
regr2 = linear_model.LinearRegression()
regr2.fit(train_X2, train_Y2)
prd_regr2=regr2.predict(test_X2)

print(rmse(test_Y2, prd_regr2))

#ElasticNet

from sklearn.linear_model import ElasticNet
enet1 = ElasticNet(alpha=0.1, l1_ratio=0.9)
y_pred_enet1 = enet1.fit(train_X1, train_Y1).predict(test_X1)

print(rmse(test_Y1, y_pred_enet1))

from sklearn.linear_model import ElasticNet
enet2 = ElasticNet(alpha=0.1, l1_ratio=0.9)
y_pred_enet2 = enet2.fit(train_X2, train_Y2).predict(test_X2)

print(rmse(test_Y2, y_pred_enet2))

# decision Tree

from sklearn.tree import DecisionTreeRegressor
dreg1 = DecisionTreeRegressor(min_samples_split=10,presort='TRUE')
dreg1.fit(train_X1, train_Y1)
dpred1 = dreg1.predict(test_X1) 
print(rmse(test_Y1, dpred1))

from sklearn.tree import DecisionTreeRegressor
dreg2 = DecisionTreeRegressor(min_samples_split=10,presort='TRUE')
dreg2.fit(train_X2, train_Y2)
dpred2 = dreg2.predict(test_X2) 
print(rmse(test_Y2, dpred2))

#Gradient boosting 

from sklearn import ensemble
params = {'n_estimators': 200, 'max_depth': 4, 'min_samples_split': 5,
          'learning_rate': 0.3, 'loss': 'huber'}
regGB1 = ensemble.GradientBoostingRegressor(**params)

regGB1.fit(train_X1, train_Y1)
gpred1 = regGB1.predict(test_X1)
print(rmse(test_Y1,gpred1))

from sklearn import ensemble
params = {'n_estimators': 200, 'max_depth': 4, 'min_samples_split': 5,
          'learning_rate': 0.3, 'loss': 'huber'}
regGB2 = ensemble.GradientBoostingRegressor(**params)

regGB2.fit(train_X2, train_Y2)
gpred2 = regGB2.predict(test_X2)
print(rmse(test_Y2,gpred2))


#RandomForest

from sklearn.ensemble import RandomForestRegressor
rf1 = RandomForestRegressor(n_estimators= 100,max_depth =6,min_samples_split=15,max_features='auto')
rf1.fit(train_X1,train_Y1)
rfpred1= rf1.predict(test_X1)
print(rmse(test_Y1,rfpred1))

from sklearn.ensemble import RandomForestRegressor
rf2 = RandomForestRegressor(n_estimators= 100,max_depth =6,min_samples_split=15,max_features='auto')
rf2.fit(train_X2,train_Y2)
rfpred2= rf2.predict(test_X2)
print(rmse(test_Y2,rfpred2))

# after comparing the rmse, i got a good rmse with Linear regression for data_eq, and for data_sup randomForest overperform the rest 

sup_pred=rf2.predict(test_sup)
SUPprobabilities = pd.DataFrame(data=sup_pred, columns ='prediction',index=test_sup.index)

eq_pred = regr.predict(test_eq)
EQprobabilities = pd.DataFrame(data=eq_pred, columns='prediction', index=test_eq.index) 

#prediction for the test data where "Nombre de séjours/séances MCO des patients en ALD" is null 

pred_nul = np.zeros((test_nul.shape[0],), dtype=np.int)
NulProbab= pd.DataFrame(data=pred_nul, index=test_nul.index)

# we join all the dataframes 

test_tab = pd.concat([EQprobabilities,SUPprobabilities],axis=0)
sub_tab = pd.concat([test_tab,NulProbab],axis=0)

# file to submit
sub_tab.to_csv('C:/work/challenge_ds/Melsub.csv')