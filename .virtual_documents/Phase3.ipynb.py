import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


path = 'D:\\Study_code\\py\\ML\\phase1\\insurance.csv'
df = pd.read_csv(path)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


df.isnull().sum()


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1, 4, 5])],
    remainder='passthrough'
)
X = ct.fit_transform(X)


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X = ss.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0, shuffle=True)


from sklearn.linear_model import LinearRegression
model1 = LinearRegression()
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred1.reshape(len(y_pred1), 1),
                      y_test.reshape(len(y_test), 1)), 1)[:10, :])


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
model2 = LinearRegression()
model2.fit(poly_reg.fit_transform(X_train), y_train)
y_pred2 = model2.predict(poly_reg.fit_transform(X_test))
print(np.concatenate((y_pred2.reshape(len(y_pred2), 1),
                      y_test.reshape(len(y_test), 1)), 1)[:10, :])


from sklearn.tree import DecisionTreeRegressor
model3 = DecisionTreeRegressor(random_state=0)
model3.fit(X_train, y_train)
y_pred3 = model3.predict(X_test)
print(np.concatenate((y_pred3.reshape(len(y_pred3), 1),
                      y_test.reshape(len(y_test), 1)), 1)[:10, :])


from sklearn.ensemble import RandomForestRegressor
model4 = RandomForestRegressor(n_estimators=200, random_state=0)
model4.fit(X_train, y_train)
y_pred4 = model4.predict(X_test)
print(np.concatenate((y_pred4.reshape(len(y_pred4), 1),
                      y_test.reshape(len(y_test), 1)), 1)[:10, :])


y_train.shape


y_train.reshape(-1,1).shape


y_train.reshape(-1,1).reshape(-1,).shape


y_train.reshape(-1,1).ravel().shape


from sklearn.svm import SVR
ss_y = StandardScaler()
y_train_ss = ss_y.fit_transform(y_train.reshape(-1,1))
model5 = SVR()
model5.fit(X_train, y_train_ss.ravel())
y_pred5 = model5.predict(X_test)
y_pred5 = ss_y.inverse_transform(y_pred5)
print(np.concatenate((y_pred5.reshape(len(y_pred5), 1),
                      y_test.reshape(len(y_test), 1)), 1)[:10, :])


from sklearn.metrics import r2_score
print("Multiple Linear Regression R squared:\t get_ipython().run_line_magic("1.5f"%", " (r2_score(y_test, y_pred1)))")
print("Polynomial Regression R squared:\t get_ipython().run_line_magic("1.5f"%", " (r2_score(y_test, y_pred2)))")
print("Decision Tree Regression R squared:\t get_ipython().run_line_magic("1.5f"%", " (r2_score(y_test, y_pred3)))")
print("Random Forest Regression R squared:\t get_ipython().run_line_magic("1.5f"%", " (r2_score(y_test, y_pred4)))")
print("Support Vector Regression R squared:\t get_ipython().run_line_magic("1.5f"%", " (r2_score(y_test, y_pred5)))")


r2_score(y_test, y_pred1)


r2_score(y_test, y_pred2)


r2_score(y_test, y_pred3)


r2_score(y_test, y_pred4)


r2_score(y_test, y_pred5)
