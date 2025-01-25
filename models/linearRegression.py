import numpy as np
import pandas as pd

class NormalLR:
    def __init__(self):

        # defining attributes for the equation --> m and b
        self.coef_ = None
        self.intercept_ = None
        print('Created..')

    def fit(self, X, y):

        # Add intercept term (add_constant)
        ones_col = np.ones( (X.shape[0], 1))
        X = np.hstack( (ones_col, X) )
        
        # compute coefficients using normal equation
        theta = np.linalg.inv(X.T @ X) @ X.T @ y

        # copy intercept_ and coef_
        self.coef_ = theta[1:]
        self.intercept_ = theta[0]

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


df = pd.read_csv('../data/restaurant_inspection_data.csv')

X = df.loc[:, ['Staff_Training_Hours', 'Customer_Complaints']]
y = df.iloc[:, 0]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 1)

nlr = NormalLR()
nlr.fit(X_train, y_train)
print(nlr.coef_, nlr.intercept_)

y_pred = nlr.predict(X_test)

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))