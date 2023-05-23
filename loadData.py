from sklearn.model_selection import KFold
import numpy as np
from keras.models import load_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import joblib



def sklearn_mod(X,y,model):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    loaded_model = joblib.load(model)

    # Use the loaded model for predictions
    y_pred = loaded_model.predict(X_test)
    print(str(model))
    print("**************")
    print("**************")
    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)
    print('Mean Squared Error (MSE):', mse)
    # Calculate R-squared
    r_squared = r2_score(y_test, y_pred)
    print('R-squared:', r_squared)
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    scores = -cross_val_score(loaded_model, X, y, cv=cv, scoring='neg_mean_squared_error')
    print("Mean Squared Error (CV):", np.mean(scores))


def keras_mod(X,y,model):
    # Load the MLP model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    mlp_model = load_model(model)  # Load the model from a file (assuming it's saved as mlp_model.h5)
    y_pred = mlp_model.predict(X_test)
    print(str(model))
    print("**************")
    print("**************")
    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)
    print('Mean Squared Error (MSE):', mse)
    # Calculate R-squared
    r_squared = r2_score(y_test, y_pred)
    print('R-squared:', r_squared)

def main_before():
    print("Before cleaning")
    data = pd.read_csv('kc_house_cleaned.csv')

    X = data.drop(['price'], axis=1)
    X = data.drop(['date'], axis=1)

    y = data['price']


    sklearn_mod(X,y,"KNNR_Before.h5")
    sklearn_mod(X,y,"RFER_Before.h5")
    sklearn_mod(X,y,"linear_Regression_Before.h5")

    keras_mod(X,y,"mlp_model_before.h5")

main_before()


def main_after():
    print("After_cleaning")
    print("***********")
    data = pd.read_csv('kc_house_High_Corr.csv')

    X = data.drop(['price'], axis=1)

    y = data['price']


    sklearn_mod(X,y,"KNNR_After.h5")
    sklearn_mod(X,y,"RFER_After.h5")
    sklearn_mod(X,y,"linear_Regression_After.h5")

    keras_mod(X,y,"MLP_After.h5")

main_after()

data = pd.read_csv('kc_house_High_Corr.csv')

X = data.drop(['price'], axis=1)

y = data['price']

sklearn_mod(X, y, "SVR_After.h5")
data = pd.read_csv('kc_house_cleaned.csv')

X = data.drop(['price'], axis=1)
X= data.drop(['date'],axis=1)
y = data['price']
sklearn_mod(X, y, "SVR_Before.h5")
