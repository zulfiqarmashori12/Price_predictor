from pathlib import Path
import seaborn as sns
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pygam as pg
import pickle 
import time
import streamlit as st

subset = ['House_Sale_Price','Sq_Ft_Tot_Living','Sq_Ft_Lot','Bathroom_count','Bedroom_count',"Bldg_Grade"]
house = pd.read_csv("house_sales.csv")
print(house[subset].head())


predictors = ['Sq_Ft_Tot_Living',"Sq_Ft_Lot","Bathroom_count","Bedroom_count","Bldg_Grade"]
outcomes = 'House_Sale_Price'
house_lm = LinearRegression()
house_lm.fit(house[predictors],house[outcomes])

print(f'Intercept: {house_lm.intercept_:3f}')
print("Coefficients: ")
for name,coef in zip(predictors,house_lm.coef_):
     print(f"{name}:{coef}")

pickle.dump(house_lm, open("house_model.pkl","wb"))

st.header("Welcom To Price predictor: ")

col1,col2 = st.columns(2)
with col1:
     living = st.number_input("Living_Area Size (Sq ft)", min_value=300, max_value=10000)
     lot = st.number_input("Lot Size (Sq Ft)", min_value=500, max_value=50000)
     grade = st.number_input("Building Grade",1,13)
with col2:
     bedrooms = st.number_input("Bedrooms",1,10)
     bathroom = st.number_input("Bathroom_count",1,10)
     

if st.button("Predict Price 💰"):

    input_data = np.array([[living, lot, bedrooms, bathroom, grade]])
    prediction = house_lm.predict(input_data)
    with st.spinner("calculating...."):
     time.sleep(3)
    st.success(f" Estimated House Price: ${prediction[0]:,.2f}")




if st.button("Show CSV File"):
    st.write("The Model is trained on this file: ")
    if st.button("Hide"):
         st.write("Hidden")
         file = False
    file = st.write(pd.read_csv("house_sales.csv"))
    