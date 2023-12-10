import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


sonar_data=pd.read_csv("sonardata.csv",header=None)

sonar_data.groupby(60).mean()

X=sonar_data.drop(columns=60,axis=1)
y=sonar_data[60]

# train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.1,stratify=y,random_state=1)


model=LogisticRegression()
# Fitting
model.fit(X_train,Y_train)

# accuracy on training data
X_train_pred=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_pred,Y_train)

print(f"with accuracy {training_data_accuracy}")


# accuracy on test data
X_test_pred=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_pred,Y_test)
print(f"with accuracy {test_data_accuracy}")