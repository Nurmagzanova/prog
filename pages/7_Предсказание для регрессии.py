import pandas as pd 
import numpy as np 
import math
import pickle
from sklearn.model_selection import train_test_split 
import streamlit as st 
from sklearn.metrics import * 

df= pd.read_csv('energy_task_preprocessed.csv')
df = df.drop('date', axis = 1)
if df is not None:
    st.header("Датасет")
    st.dataframe(df)
    st.write("---")
    st.title("Appliances Prediction") 

    st.markdown('Для предсказания необходимо выделить целевой признак, а также разделить датасет на обучающую и тестовую выборку:')
    code = '''
    Y = data['Appliances']
    X = data.drop(['Appliances'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    bayesian_ridge = BayesianRidge()
    bayesian_ridge.fit(X_train, y_train)

    y_pred = bayesian_ridge.predict(X_test)
    '''

    st.code(code, language='python')
    list=[]
    df = df.drop('Appliances', axis = 1)

    for i in df.columns[:]:
        a = st.slider(i,float(df[i].min()), float(math.ceil(df[i].max())),float(df[i].max()/2))
        list.append(a)

    

    list = np.array(list).reshape(1,-1)
    list=list.tolist()
    st.title("Тип модели обучения: Nearest Neighbors Regression")
    

    button_clicked = st.button("Предсказать")
    if button_clicked:
        with open('models/r_model.pkl', 'rb') as file:
            baesian_model = pickle.load(file)
            y_pred = baesian_model.predict(list)
            st.success(y_pred)
            st.markdown('mean_absolute_error:')
            st.code(52.702360109718384, language='python')