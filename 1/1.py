#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#--------------------------------------------------------------------------------------------


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


my_housing = pd.read_csv('ex1data.csv')


# In[4]:


my_housing.head()


# In[5]:


my_housing.info()


# In[6]:


my_housing.describe()


# In[ ]:





# In[7]:


my_housing.columns


# In[12]:


sns.pairplot(my_housing)


# In[ ]:





# In[8]:


sns.displot(my_housing['Price'])


# In[ ]:





# In[9]:


sns.heatmap(my_housing.corr())


# In[ ]:





# In[10]:


X = my_housing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = my_housing['Price']


# In[ ]:





# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.4, random_state =101)


# In[13]:


from sklearn.linear_model import LinearRegression


# In[16]:


lm = LinearRegression()


# In[17]:


lm.fit(X_train,y_train)


# In[ ]:





# In[18]:


print(lm.intercept_)


# In[ ]:





# In[19]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[ ]:





# In[20]:


predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)


# In[36]:


sns.displot((y_test-predictions),bins=50);


# In[37]:


from sklearn import metrics


# In[38]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:


#---------------------------------------------------------------------------------------------------

