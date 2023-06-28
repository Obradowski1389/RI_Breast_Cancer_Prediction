import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from ann_comp_graph import *

import ml


global X_train, X_test, Y_train, Y_test, df, clf, names


def explore_data():
    global df

    df["class"] = df["class"].map({2:0, 4:1})
    print(df.describe())
    diagnosis_counts = df['class'].value_counts()
    diagnosis_labels = ['Benign', 'Malignant']

    # Creating the bar plot
    # plt.bar(diagnosis_labels, diagnosis_counts, width=0.5)

    # # Adding labels and title
    # plt.xlabel('Diagnosis')
    # plt.ylabel('Count')
    # plt.show()


def load_dataset():
    global df, names
    names = [
        'id', 'clumpThickness', 'uniformityOfCellSize', 'uniformityOfCellShape', 'MarginalAdhesion', 
        'singleEpithelialCellSize', 'bareNuclei', 'blandChromatin', 'normalNucleoli', 'mitoses', 'class'
    ]
    df = pd.read_csv('./Dataset/breast-cancer-wisconsin.data', names=names)

    df.drop('id', axis=1, inplace=True)
    df.replace('?', np.nan, inplace=True)

    # print(df.head)
    # explore_data()


def handle_incorrect_values_method1():
    global df

    df.dropna(inplace=True)


def handle_incorrect_values_method2():
    global df
    
    # Convert columns to numeric type
    df = df.apply(pd.to_numeric, errors='coerce')
    
    column_averages = df.mean()

    df.fillna(column_averages, inplace=True)
    

def build_model():
    global X_train, X_test, Y_train, Y_test, df

    X_train, X_test, Y_train, Y_test = train_test_split(
        df[[col for col in df.columns if col != 'class']],
        df['class'],
        test_size=0.30,
        random_state=0)


def cnn_accuracy():
    global X_train, X_test, Y_train, Y_test, df
    
    train_p = clf.predict(X_train)
    test_p = clf.predict(X_test)

    train_acc = accuracy_score(Y_train, train_p)
    test_acc = accuracy_score(Y_test, test_p)

    return test_acc * 100, train_acc


def tts(df, percent):
    # train=df.sample(frac=percent,random_state=200)
    train=df.sample(frac=percent)
    test=df.drop(train.index)
    return train, test


def predict_mse(nn, test_x, test_y):
    good = 0
    for x, y in zip(test_x, test_y):
        res = nn.predict(x)
        res = normalize_res(res[0])
        if (res == y):
            good += 1
    return good/len(test_x) * 100


def normalize_res(res):
    distance_to_2 = abs(res - 2)
    distance_to_4 = abs(res - 4)

    if distance_to_2 < distance_to_4:
        return 2
    else:
        return 4


def main():
    global X_train, X_test, Y_train, Y_test, df, clf
    load_dataset()
    handle_incorrect_values_method2()
    build_model()

    prediction_input = [x for x in names if x!= "class" and x != "id"]
    prediction_output = "class"

    split = [X_train, Y_train, X_test, Y_test]
    print("\n\nDecision Tree")
    dt = ml.classification_model(DecisionTreeClassifier(), df, prediction_input, prediction_output, split) * 100
    print("\n\nK Neighbors")
    kn = ml.classification_model(KNeighborsClassifier(), df, prediction_input, prediction_output, split) * 100
    print("\n\nRandom Forest")
    rf = ml.classification_model(RandomForestClassifier(n_estimators=100), df, prediction_input, prediction_output, split) * 100

    
    ann_train, ann_test = tts(df, 0.7)
    ann_train_y = np.array([ann_train['class'].to_numpy()]).transpose()
    ann_train_x = ann_train.drop('class', axis=1).to_numpy()

    ann_test_y = ann_test['class'].to_numpy()
    ann_test_x = ann_test.drop('class', axis=1).to_numpy()
    ann_scaler = StandardScaler()
    ann_train_x = ann_scaler.fit_transform(ann_train_x)
    ann_test_x = ann_scaler.transform(ann_test_x)

    nn = NeuralNetwork()
    nn.add(NeuralLayer(ann_train_x.shape[1], 16, 'sigmoid'))
    nn.add(NeuralLayer(16, 1, 'relu'))
    hist = nn.fit(ann_train_x, ann_train_y, learning_rate=0.1, momentum=0.3, nb_epochs=10, shuffle=True, verbose=1)
    ann_ftn = predict_mse(nn, ann_test_x, ann_test_y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
    clf.fit(X_train, Y_train)
    cnn, _ = cnn_accuracy()

    models = pd.DataFrame({'Model': ['CNN', 'Decision Tree', 'Random Forest', 'K Neighbors', 'ANN_FTN'], 'Score': [cnn, dt, rf, kn, ann_ftn]})
    models.sort_values(by='Score', ascending=False)

    bar = px.bar(data_frame=models, x='Score', y='Model', color='Score', title='Comparison')
    bar.update_layout(
        font=dict(
            size=18,  # Set the font size here
            color="Black"
        )
    )

    bar.show()


if __name__ == "__main__":
    main()
