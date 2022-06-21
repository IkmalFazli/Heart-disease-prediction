# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 12:15:55 2022

@author: Si Kemal
"""


import pickle
import os
#import trained model
import numpy as np

MODEL_PATH = os.path.join(os.getcwd(),'best_model.pkl')

with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)


# training      :  model.fit
# score         :  model.score
# deployment    :  model.predict

# EDA
# Step 1) Data Loading
# Read csv
# Manual data entry

import streamlit as st

def set_bg_hack_url():
    st.markdown(f""" <style>.stApp {{
             background:url('https://images.news18.com/ibnlive/uploads/2022/01/heart-health-16430292883x2.jpg?impolicy=website&width=510&height=356');
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True)
        
set_bg_hack_url()
        
st.markdown(""" <style> .font {font-size:42px ; 
            font-family: Garamond, serif; 
            color: #C5B358;
            background-color:black;
            opacity:0.8;
            text-align: center;
            } </style> """, unsafe_allow_html=True)

st.markdown('<p class="font">DO WE ALL REALLY SAFE FROM CARDIOVASCULAR DISEASE???', unsafe_allow_html=True)

st.write('This is an app to predict if a person has the probability of getting \
         cardiovascular disease')
         
with st.form('my_form'):
#'age','trtbps','chol','thalachh','oldpeak','cp','thall'                                     
    age = int(st.number_input("Age:"))
    trtbps =int(st.number_input("Resting blood pressure(in mm Hg):", min_value=0, max_value=320))
    chol=int(st.number_input("Your cholestrol in mg/dl: "))
    thalachh=int(st.number_input("Maximum heart rate achieved:"))
    oldpeak=int(st.number_input("Previous peak:"))
    chestpain = st.radio("Your Chest Pain Type? ", ('Typical Angina', 'Atypical angina',
                                                    'NON-angina pain','Asymptomatic'))
    if chestpain == 'Typical Angina':cp = 0
    elif chestpain == 'Atypical angina':cp = 1
    elif chestpain == 'NON-angina pain':cp = 2
    else :
        cp = 3
        
    Thalles = st.radio("Your Chest Pain Type? ", ('No', 'Fixed defect',
                                                    'Normal','Reversable defect'))
    if Thalles == 'No':thall = 0
    elif Thalles == 'Fixed defect':thall = 1
    elif Thalles == 'Normal':thall = 2
    else :
        thall = 3
    
    submit_button = st.form_submit_button(label='Predict')
    if submit_button:
        temp = np.expand_dims([age,trtbps,chol,thalachh,oldpeak,cp,thall],axis=0)
        outcome = model.predict(temp)

        outcome_dict = {0:'Less risk of getting Heart disease',
                        1:'High risk of getting Heart disease'}

        st.write(outcome_dict[outcome[0]])
        
        if outcome == 1:
            st.snow()
            unsafe_html = '''<div style="background-color:tomato";>
                             <h2 style="color:black";>You better take care of your health</h2></div>
                          '''
            st.markdown(unsafe_html,unsafe_allow_html=True)
        else:
            st.snow()
            unsafe_html = '''<div style="background-color:green";>
                             <h2 style="color:black";>Good, you're healthy</h2></div>
                          '''
            st.markdown(unsafe_html,unsafe_allow_html=True)
