from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

def classification_model(model,data, prediction_input, output, split):
    X_train = split[0]
    Y_train=split[1]
    X_test = split[2]
    Y_test = split[3]
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
    return accuracy
