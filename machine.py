#!/usr/bin/env python
# coding: utf-8

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

# In[1]:

from sklearn.svm import SVR
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


# In[2]:


path = 'C:\\Users\\moham\\Downloads\\insurance.csv'
df = pd.read_csv(path)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


# # Phase 2
# ### Handle Empty cells

# In[3]:


df.isnull().sum()


# df contains no null values

#  ### Encoding

# In[4]:


ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1, 3, 4, 5])],
    remainder='passthrough'
)
X = ct.fit_transform(X)

# One hot encode the data

# ### Train-test Split

# In[5]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=44, shuffle=True)


# ### Plot Train Data

# In[6]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.hist(X_train[:, 14], bins=46)
ax2.hist(X_train[:, 15], bins=int(len(X_train[:, 12])/10))
ax1.set_title('Age distribution')
ax2.set_title('BMI distribution')
plt.show()


# # Phase 3
# ## Applying Models
# ### Multiple linear regression

# In[7]:


model1 = LinearRegression()
model1.fit(X_train, y_train)
z=model1.score(X_test, y_test)
print(z)
predict = model1.predict(X_test)
print(predict[0:10])
print(y_test[0:10])

# ### Polynomial regression

# In[8]:


poly_reg = PolynomialFeatures(degree=2)
model2 = LinearRegression()
model2.fit(poly_reg.fit_transform(X_train), y_train)
z=model2.score(poly_reg.fit_transform(X_test), y_test)
predict=model2.predict(poly_reg.fit_transform(X_test))
print(z)
print(predict[0:10])
print(y_test[0:10])


# ### Support Vector Regression
model3 = SVR(kernel = 'linear' , C=20 , epsilon=0.01)
model3.fit(X_train, y_train)
z=model3.score(X_test,y_test)
predict = model3.predict(X_test)
print(z)
print(predict[0:10])
print(y_test[0:10])

