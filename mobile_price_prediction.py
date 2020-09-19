
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

paths=list()
paths.append("train_data.csv")
paths.append("test_data.csv")
dataset=pd.read_csv(paths[0])
x=dataset.iloc[: , 1:-1].values
y=dataset.iloc[:,-1].values


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)


from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(x_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_





if(best_parameters['kernel']=='linear'):
    
    classifier_edited=SVC(kernel=best_parameters['kernel'],C=best_parameters['C'],random_state=0)
    
    classifier_edited.fit(x_train,y_train)
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator = classifier_edited, X = x_train, y = y_train, cv = 10)
    print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
    print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

    
else:
    classifier_edited=SVC(kernel=best_parameters['kernel'],C=best_parameters['C'],gamma=best_parameters['gamma'],random_state=0)
    classifier_edited.fit(x_train,y_train)
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator = classifier_edited, X = x_train, y = y_train, cv = 10)
    print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
    print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# ===========================================EVALUATING TEST SET RESULTS===============================================    
dataset_edited=pd.read_csv(paths[1])
x_t=dataset_edited.iloc[: ,1:]
#print(x_t)

sc_t=StandardScaler()
x_t=sc_t.fit_transform(x_t)


    

y_pred=classifier_edited.predict(x_t)

print(y_pred)        

import csv
fields=['id','price_range']

data_set=list()

for i in range(0,len(y_pred)):
    data=list()
    data.append(i+1401)
    data.append(y_pred[i])
    data_set.append(data)

filename="predicted_prices.csv"
with open(filename,"w") as csvfile:
    csvwriter=csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(data_set)
print("done")

