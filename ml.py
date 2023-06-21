from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


global X_train, X_test, Y_train, Y_test

def build_model(df):
    global X_train, X_test, Y_train, Y_test

    X_train, X_test, Y_train, Y_test = train_test_split(
        df[[col for col in df.columns if col != 'class']],
        df['class'],
        test_size=0.30,
        random_state=0)

def classification_model(model,data, prediction_input, output):
    global X_train, X_test, Y_train, Y_test
    build_model(data)

    # TRAIN-TEST SPLIT
    model.fit(X_train, Y_train)
  
    #Make predictions on training set:
    predictions = model.predict(X_test)
  
    #Print accuracy
    accuracy = accuracy_score(predictions,Y_test)
    print("Accuracy : %s" % "{0:.3%}".format(accuracy))

    kf = KFold(n_splits=5)
    error = []
    for train, test in kf.split(data):
        train_X = (data[prediction_input].iloc[train,:])
        train_y = data[output].iloc[train]
        model.fit(train_X, train_y)
    
        
        test_X=data[prediction_input].iloc[test,:]
        test_y=data[output].iloc[test]
        error.append(model.score(test_X,test_y))
    
        print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
