import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('../data/restaurant_inspection_data.csv')

# Define the independent variables (add a constant for the intercept)
X = df.iloc[:, [1, 2]]
X = sm.add_constant(X)

# Define the dependent variable
y = df.iloc[:, 0]

# Fit the model using the independent and dependent variables
model = sm.OLS(y, X).fit()
print(model.summary())

print(sm.OLS(y, X).df_model)