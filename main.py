import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split


global X_train, X_test, Y_train, Y_test, df


def load_dataset():
    global df
    df = pd.read_csv('./Dataset/breast-cancer-wisconsin.data', names=[
        'id', 'clumpThickness', 'uniformityOfCellSize', 'uniformityOfCellShape', 'MarginalAdhesion', 
        'singleEpithelialCellSize', 'bareNuclei', 'blandChromatin', 'normalNucleoli', 'mitoses', 'class'
    ])

    df.drop('id', axis=1, inplace=True)

    print(df.head)


def split_data():
    raise NotImplementedError


def handle_incorrect_values_method1():
    global df

    df.replace('?', np.nan, inplace=True)

    # replacement_value = 0 
    # df.fillna(replacement_value, inplace=True)

    df.dropna(inplace=True)


def handle_incorrect_values_method2():
    global df
    
    df.replace('?', np.nan, inplace=True)
    
    # Convert columns to numeric type
    df = df.apply(pd.to_numeric, errors='coerce')
    
    column_averages = df.mean()

    df.fillna(column_averages, inplace=True)


def build_model():
    raise NotImplementedError


def main():
    load_dataset()
    handle_incorrect_values_method1()


if __name__ == "__main__":
    main()
