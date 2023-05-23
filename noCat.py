from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import joblib



def build_sklearn_model(X,y,modelname,model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model.fit(X, y)

    y_pred = model.predict(X_test)
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    scores = -cross_val_score(model, X, y,cv=cv, scoring='neg_mean_squared_error')
    print("Mean Squared Error (CV):",np.mean(scores))
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)

    joblib.dump(model, modelname + '.h5')

def build_MLP(X,y,modelname):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = Sequential()
    model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_test, y_test))


    y_pred = model.predict(X_test)
    model.save("mlp_model_before.h5")

    mse = mean_squared_error(y_test, y_pred)
    print('Mean Squared Error (MSE):', mse)
    r_squared = r2_score(y_test, y_pred)
    print('R-squared:', r_squared)
    model.save(modelname + '.h5')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    plt.show()

def correlation(data):
    sorted_corr_values = data.corr()['price'].sort_values()

    file_path = 'sorted_corr_values2.txt'
    file = open(file_path, 'w')

    for feature, correlation in sorted_corr_values.iteritems():
        line = f"{feature}: {correlation}\n"
        file.write(line)

    file.close()



def main():
    data = pd.read_csv('kc_house_cleaned.csv')

    X = data.drop(['price'], axis=1)
    X = data.drop(['date'], axis=1)

    y = data['price']

    correlation(data)
    build_sklearn_model(X,y,"SVR_Before",SVR(kernel='rbf'))
    build_sklearn_model(X,y,"linear_Regression_After",LinearRegression())
    build_sklearn_model(X,y,"KNNR_Before",KNeighborsRegressor(n_neighbors=5))
    build_sklearn_model(X,y,"SVR_Before",RandomForestRegressor(max_depth=2, random_state=0))

    build_MLP(X,y,"MLP_After")

main()