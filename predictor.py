import pandas as pd
import pickle
import streamlit as st
import numpy as np

data = pd.read_csv("Final_database.csv")
pipe = pickle.load(open("LinearRegressionModel.pkl", 'rb'))

print(data['site_location'])

st.title("Pune Property Price Prediction")


location_sorted = sorted(data['site_location'].unique())



location = st.selectbox('Location', location_sorted)
bhk = st.number_input('BHK')
bath = st.number_input('Bath')
sqft = st.number_input('Total square feet')


if st.button('Predict'):
    input = pd.DataFrame([[sqft, bath, location, bhk]],columns=['total_sqft', 'bath', 'site_location','bhk'])
    prediction = pipe.predict(input)[0] * 1e5
    st.header("Prediction: ₹" + str(np.round(prediction, 2)))
    