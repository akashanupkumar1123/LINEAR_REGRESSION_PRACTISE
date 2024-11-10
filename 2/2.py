#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[2]:


customers = pd.read_csv("Ecommerce Customers")


# In[3]:


customers.head()


# In[4]:


customers.describe()


# In[ ]:





# In[5]:


customers.info()


# In[9]:


sns.set_palette("GnBu_d")
sns.set_style('whitegrid')


# In[7]:


sns.jointplot(x = 'Time on Website', y = 'Yearly Amount Spent', data = customers)


# In[10]:


sns.jointplot(x='Time on App', y = 'Yearly Amount Spent', data= customers)


# In[14]:


sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=customers)


# In[ ]:





# In[15]:


sns.pairplot(customers)


# In[ ]:





# In[16]:


sns.lmplot(x='Length of Membership', y = 'Yearly Amount Spent', data = customers)


# In[ ]:





# In[22]:


y = customers['Yearly Amount Spent']

X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]


# In[ ]:





# In[23]:


from sklearn.model_selection import train_test_split


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=101)


# In[ ]:





# In[31]:


from sklearn.linear_model import LinearRegression


# In[32]:


lm = LinearRegression()


# In[33]:


lm.fit(X_train,y_train)


# In[34]:


print('Coefficients: \n', lm.coef_)


# In[ ]:





# In[35]:


predictions = lm.predict(X_test)


# In[37]:


plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[ ]:





# In[38]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:


#Residuals


# In[41]:


sns.histplot((y_test-predictions),bins=50);


# In[ ]:





# In[43]:


coeffecients = pd.DataFrame(lm.coef_, X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients


# In[ ]:


#__________________________________________-----------------------------------------


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




