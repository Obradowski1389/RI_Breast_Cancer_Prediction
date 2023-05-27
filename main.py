import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


global X_train, X_test, Y_train, Y_test, df, clf


def load_dataset():
    global df
    df = pd.read_csv('./Dataset/breast-cancer-wisconsin.data', names=[
        'id', 'clumpThickness', 'uniformityOfCellSize', 'uniformityOfCellShape', 'MarginalAdhesion', 
        'singleEpithelialCellSize', 'bareNuclei', 'blandChromatin', 'normalNucleoli', 'mitoses', 'class'
    ])

    df.drop('id', axis=1, inplace=True)

    # print(df.head)


def split_data():
    raise NotImplementedError


def handle_incorrect_values_method1():
    global df

    df.replace('?', np.nan, inplace=True)

    df.dropna(inplace=True)


def handle_incorrect_values_method2():
    global df
    
    df.replace('?', np.nan, inplace=True)
    
    # Convert columns to numeric type
    df = df.apply(pd.to_numeric, errors='coerce')
    
    column_averages = df.mean()

    df.fillna(column_averages, inplace=True)


def handle_incorrect_values_method3():
    # Slicnost po ostalim kolonama KNN algorimom 
    raise NotImplementedError
    

def build_model():
    global X_train, X_test, Y_train, Y_test, df

    X_train, X_test, Y_train, Y_test = train_test_split(
        df[[col for col in df.columns if col != 'class']],
        df['class'],
        test_size=0.30,
        random_state=0)


def cnn_accuracy():
    global X_train, X_test, Y_train, Y_test, df
    
    # train_p = clf.predict(X_train)
    test_p = clf.predict(X_test)

    # train_acc = accuracy_score.score(Y_train, train_p)
    test_acc = accuracy_score(Y_test, test_p)

    return test_acc



def main():
    global X_train, X_test, Y_train, Y_test, df, clf
    load_dataset()
    handle_incorrect_values_method2()
    build_model()

    # print(df.head())

    lr = 0 # linear_regression(X_train, X_test, Y_train, Y_test, df)
    rf = 0 # random_forest(X_train, X_test, Y_train, Y_test, df)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
    clf.fit(X_train, Y_train)
    cnn = cnn_accuracy()
    print("CNN:", cnn)

    models = pd.DataFrame({'Model': ['CNN', 'Linear Regresion', 'Random Forest'], 'Score': [cnn, lr, rf]})
    models.sort_values(by='Score', ascending=False)

    bar = px.bar(data_frame=models, x='Score', y='Model', color='Score', template='plotly_dark', title='Comparison')

    bar.show()


if __name__ == "__main__":
    main()
