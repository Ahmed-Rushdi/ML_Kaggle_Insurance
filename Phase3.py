# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # <p style="text-align: center;"> Names </p>
# ## <p style="text-align: center;"> AI S1,S2 </p>
# | <font size="4"> Name  </font> | <font size="4">  ID  </font>|
# | --- | --- |
# | <font size="4"> Ahmed Rushdi Mohammed </font> | <font size="4"> 20180008 </font>  |
# |  <font size="4"> Mohammed Waleed Mohammed </font> | <font size="4"> 20180244 </font>  |
# |  <font size="4"> Aoss Maged Sultan</font> | <font size="4"> 20180432 </font>  |
#
# # Phase 1
#
# ### 1. Load Dataset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path = 'D:\\Study_code\\py\\ML\\phase1\\insurance.csv'
df = pd.read_csv(path)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# # Phase 2
# ### Handle Empty cells

df.isnull().sum()

df

# df contains no null values

# ### Encoding

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1, 4, 5])],
    remainder='passthrough'
)
X = ct.fit_transform(X)

# One hot encode the data

# ### Feature Scaling

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X = ss.fit_transform(X)

# Data is scaled to Support SVR. And it won't affect any other model. 

# ### Train-test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0, shuffle=True)

# # Phase 3
# ## Applying Models
# ### Multiple linear regression

from sklearn.linear_model import LinearRegression
model1 = LinearRegression()
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred1.reshape(len(y_pred1), 1),
                      y_test.reshape(len(y_test), 1)), 1)[:10, :])

# ### Polynomial regression

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
model2 = LinearRegression()
model2.fit(poly_reg.fit_transform(X_train), y_train)
y_pred2 = model2.predict(poly_reg.fit_transform(X_test))
print(np.concatenate((y_pred2.reshape(len(y_pred2), 1),
                      y_test.reshape(len(y_test), 1)), 1)[:10, :])

# ### Decision Tree Regression

from sklearn.tree import DecisionTreeRegressor
model3 = DecisionTreeRegressor(random_state=0)
model3.fit(X_train, y_train)
y_pred3 = model3.predict(X_test)
print(np.concatenate((y_pred3.reshape(len(y_pred3), 1),
                      y_test.reshape(len(y_test), 1)), 1)[:10, :])

# ### Random Forest Regression

from sklearn.ensemble import RandomForestRegressor
model4 = RandomForestRegressor(n_estimators=200, random_state=0)
model4.fit(X_train, y_train)
y_pred4 = model4.predict(X_test)
print(np.concatenate((y_pred4.reshape(len(y_pred4), 1),
                      y_test.reshape(len(y_test), 1)), 1)[:10, :])

# ### Support Vector Regression

from sklearn.svm import SVR
ss_y = StandardScaler()
y_train_ss = ss_y.fit_transform(y_train.reshape(-1,1))
model5 = SVR(kernel='rbf',C=3)
model5.fit(X_train, y_train_ss.ravel())
y_pred5 = model5.predict(X_test)
y_pred5 = ss_y.inverse_transform(y_pred5)
print(np.concatenate((y_pred5.reshape(len(y_pred5), 1),
                      y_test.reshape(len(y_test), 1)), 1)[:10, :])

# ## Evaluating Performance

from sklearn.metrics import r2_score
print("Multiple Linear Regression R squared:\t %1.5f"% (r2_score(y_test, y_pred1)))
print("Polynomial Regression R squared:\t %1.5f"% (r2_score(y_test, y_pred2)))
print("Decision Tree Regression R squared:\t %1.5f"% (r2_score(y_test, y_pred3)))
print("Random Forest Regression R squared:\t %1.5f"% (r2_score(y_test, y_pred4)))
print("Support Vector Regression R squared:\t %1.5f"% (r2_score(y_test, y_pred5)))

# ## Observations
# 1. The best performing model is SVR(rbf).
# 2. Within Expectations Random Forest performed better than the Decision Tree model.
# 3. Polynomial Regression (Second degree) performed better than Multiple Linear Regression. and the model becomes overfitted at higher degrees.
