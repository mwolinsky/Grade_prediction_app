#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd 
import numpy as np 
import streamlit as st 
from PIL import Image 
from xgboost.sklearn import XGBRegressor


#Colocamos el lugar de donde extraer el csv
data_location= 'C:/Users/LX569DW/OneDrive - EY/Documents/Data Academy/Integrador/student-por.csv'
data_location1= 'C:/Users/LX569DW/OneDrive - EY/Documents/Data Academy/Integrador/student-mat.csv'

df1= pd.read_csv(data_location, sep=';')
df2= pd.read_csv(data_location1, sep=';')

df= pd.merge(df1,df2,how= 'outer')



#reemplazamos los valores de números a clases de las variables Medu y Fedu
df.Fedu=df.Fedu.replace([0,1,2,3,4],['none','Primary_education','5th_to_9th','secondary_education','higher_education'])
df.Medu=df.Medu.replace([0,1,2,3,4],['none','Primary_education','5th_to_9th','secondary_education','higher_education'])

#Transformamos la variable tiempo de viaje
df.traveltime=df.traveltime.replace([1,2,3,4],['less_15_min','15_to_30_min','30_to_60_min', 'more_60_min'])

#Transformamos la variable studytime para pasarla a categorías string
df.studytime=df.studytime.replace([1,2,3,4], ['less_2_hs','2_to_5_hs','5_to_10_hs','more_10_hs'])


#Transformamos la variable famrel para pasarla a categorías strings
df.famrel=df.famrel.replace([1,2,3,4,5], ['very_bad','bad','good','very_good','excellent'])


#Generemos un loop para hacer el reemplazo de estas variables
list_trans=['freetime','goout','Walc','Dalc']
for i in list_trans:
    df[i]=df[i].replace([1,2,3,4,5],['very_low','low','medium','high','very_high'])
    
df.health=df.health.replace([1,2,3,4,5],['very_bad','bad','OK','good','very_good'])

categorical_columns=['school', 'sex', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health']
for column in categorical_columns:
    dummies = pd.get_dummies(df[column], prefix=column,drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=column)
    
    
X=df.loc[:,['G2', 'G1', 'age', 'higher_yes', 'freetime_low','Dalc_very_low']]
y= df.G3
#Ahora haremos un split para crear datos para entrenar y otros para predecir

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False,random_state=12)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test),columns=X_test.columns)


from sklearn.linear_model import LassoCV
model_3 = LassoCV(cv=5, random_state=0,n_alphas=5000,max_iter=20000)
model_3.fit(X_train ,y_train)

from sklearn.model_selection import cross_val_score, StratifiedKFold
model_xg = XGBRegressor(n_jobs=-1, use_label_encoder=False)

cv = StratifiedKFold(n_splits=5, random_state=41, shuffle=True)

#Hacemos un Grid Search para optimizar hiperparámetros
from sklearn.model_selection import RandomizedSearchCV
params = { 'max_depth': [3, 5, 6, 10, 15, 20],
           'learning_rate': [0.01, 0.1, 0.2, 0.3],
           'subsample': np.arange(0.5, 1.0, 0.1),
           'colsample_bytree': np.arange(0.4, 1.0, 0.1),
           'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
           'n_estimators': [100, 500, 1000]}

clf = RandomizedSearchCV(model_xg, param_distributions=params, cv=cv, verbose=1, n_jobs=-1,random_state=2123)

clf.fit(X_train,y_train)

print(clf.best_params_)
print(clf.best_score_)

model_xg = XGBRegressor(n_jobs=-1, use_label_encoder=False,random_state=10,**clf.best_params_)
model_xg.fit(X_train,y_train)



def welcome(): 
    return 'welcome all'
  
def prediction(G2, G1, age, higher_yes, freetime_low,Dalc_very_low):   
    mw=np.array([G2, G1, age, higher_yes, freetime_low,Dalc_very_low]).reshape(1,-1)
    prediction = model_xg.predict(mw)
    print(prediction) 
    return prediction 
  
def main(): 
      
    st.title("G3 Prediction") 
    html_temp = ""
    
    st.markdown(html_temp, unsafe_allow_html = True) 
    G2_input = int(st.number_input("G2") )
    G1_input = int(st.number_input("G1") )
    age_input = int(st.number_input("Age"))
    higher_yes_input = int(st.number_input('Want to have higher education? If yes, write 1, otherwise, 0'))
    freetime_low_input =  int(st.number_input('Do you have low free time? If yes, write 1, otherwise, 0'))
    Dalc_very_low_input = int(st.number_input("Is your dayly alcohol consumption very low? If yes, write '1', otherwise, '0'"))
    result ="" 

    
    
    if st.button("Predict"): 
        result = prediction(G2_input,G1_input,age_input,higher_yes_input,freetime_low_input,Dalc_very_low_input) 
    #st.success('His/her G3 result is:'.format(result)) 
    st.success(result)
if __name__=='__main__': 
    main() 
    


# In[ ]:




