import streamlit as st
import pandas as pd
import numpy as np      
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report
import pickle

def load_data():
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    df.dropna(subset=['bmi'], inplace=True)  # Fixed dropping missing values
    df.drop(['id'], axis=1, inplace=True)
    return df

df = load_data()

def load_model():
    with open('pickle_model.pkl', 'rb') as file:  # Fixed filename typo
        data = pickle.load(file)
    return data

data = load_model()
clf = data['model']
encoder = data['encoder']

st.set_page_config(page_title="Stroke Prediction ", layout="centered")
st.title("Stroke Prediction App")
