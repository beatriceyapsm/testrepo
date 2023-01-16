
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

def traintest_split(df):
    global X_train
    global X_test
    global y_train
    global y_test
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df[["Usage","Fuel consumption","Membership","Model-year"]], df["Failures"],test_size=0.2)
    return X_train, X_test, y_train, y_test

def balance_data(X_train, y_train):
    # Balance the data
    over = SMOTE(sampling_strategy=0.5)
    under = RandomUnderSampler(sampling_strategy=0.9)

    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    X_train, y_train = pipeline.fit_resample(X_train, y_train)
    return X_train, y_train


