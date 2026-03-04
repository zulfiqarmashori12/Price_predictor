import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
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
     
input_data = np.array([[living, lot, bedrooms, bathroom, grade]])
prediction = house_lm.predict(input_data)
if st.button("Predict Price 💰"):


    with st.spinner("calculating...."):
     time.sleep(3)
    st.success(f" Estimated House Price: ${prediction[0]:,.2f}")



col3,col4 = st.columns(2)
with col3:
 if st.button("Show CSV File"):
    st.write("The Model is trained on this file: ")
    if st.button("Hide"):
         st.write("Hidden")
         file = False
    file = st.write(pd.read_csv("house_sales.csv"))
with col4:
 if st.button("Summary"):
    with st.expander(f"The Predicted Price of House is  ${prediction}"):
        st.write(f"{living} Living_Area Size (Sq ft)")
        st.write(f"{lot} Lot Size (Sq Ft)")
        st.write(f"{grade} Building grade")
        st.write(f"No. of Bathrooms {bathroom}  ")
        st.write(f"Total {bedrooms} Bedroom")
        st.write("price: ",prediction)

    if st.button("Hide Summary"):
        summary = False
