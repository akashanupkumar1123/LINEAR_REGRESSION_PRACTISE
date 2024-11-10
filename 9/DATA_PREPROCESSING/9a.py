#!/usr/bin/env python
# coding: utf-8

# # Data Pre-Processing

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


print(np.__version__)
print(pd.__version__)
import sys
print(sys.version)
import sklearn
print(sklearn.__version__)


# In[ ]:


from sklearn.datasets import load_boston
boston_data = load_boston()
df = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
df.head()


# In[ ]:


X = df[['LSTAT']].values
y = boston_data.target


# In[ ]:


plt.figure(figsize=(8,6))
plt.scatter(X, y);


# ## Without Pre-processing

# In[ ]:


alpha = 0.0001
w_ = np.zeros(1 + X.shape[1])
cost_ = []
n_ = 100

for i in range(n_):
    y_pred = np.dot(X, w_[1:]) + w_[0]
    errors = (y - y_pred)
    
    w_[1:] += alpha * X.T.dot(errors)
    w_[0] += alpha * errors.sum()
    
    cost = (errors**2).sum() / 2.0
    cost_.append(cost)
    
plt.figure(figsize=(8,6))
plt.plot(range(1, n_ + 1), cost_);
plt.ylabel('SSE');
plt.xlabel('Epoch');    


# # With Pre-processing

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y.reshape(-1,1)).flatten()


# In[ ]:


alpha = 0.0001
w_ = np.zeros(1 + X_std.shape[1])
cost_ = []
n_ = 100

for i in range(n_):
    y_pred = np.dot(X_std, w_[1:]) + w_[0]
    errors = (y_std - y_pred)
    
    w_[1:] += alpha * X_std.T.dot(errors)
    w_[0] += alpha * errors.sum()
    
    cost = (errors**2).sum() / 2.0
    cost_.append(cost)
plt.figure(figsize=(8,6))
plt.plot(range(1, n_ + 1), cost_);
plt.ylabel('SSE');
plt.xlabel('Epoch');    


# ***

# Before Scaling

# In[ ]:


plt.figure(figsize=(8,6))
plt.hist(X);
plt.xlim(-40, 40);


# After Scaling

# In[ ]:


plt.figure(figsize=(8,6))
plt.hist(X_std);
plt.xlim(-4, 4);


# ***

# # Data Pre-processing

# * Standardization / Mean Removal
# 
# * Min-Max or Scaling Features to a Range
# 
# * Normalization
# 
# * Binarization

# **Assumptions**:
# 
# * Implicit/explicit assumption of machine learning algorithms: The features follow a normal distribution.
# * Most method are based on linear assumptions
# * Most machine learning requires the data to be standard normally distributed. Gaussian with zero mean and unit variance.
# 
# [scikit-learn:](http://scikit-learn.org/stable/modules/preprocessing.html) 
# 
# In practice we often ignore the shape of the distribution and just transform the data to center it by removing the mean value of each feature, then scale it by dividing non-constant features by their standard deviation.
# 
# For instance, many elements used in the objective function of a learning algorithm (such as the RBF kernel of Support Vector Machines or the l1 and l2 regularizers of linear models) **assume that all features are centered around zero and have variance in the same order**. If a feature has a variance that is orders of magnitude larger than others, it might dominate the objective function and make the estimator unable to learn from other features correctly as expected.
# 
# 

# In[ ]:


from sklearn import preprocessing


# In[ ]:


X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])


# In[ ]:


X_train.mean(axis=0)


# # Standardization / Mean Removal / Variance Scaling
# 
# [scikit Scale](http://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling)

# Mean is removed. Data is centered on zero. This is to remove bias.
# 
# Standardization of datasets is a common requirement for many machine learning estimators implemented in scikit-learn; they might behave badly if the individual features do not more or less look like standard normally distributed data: Gaussian with zero mean and unit variance. "standard normal" random variable with mean 0 and standard deviation 1.
# 
# $$X'=\frac{X-\bar{X}}{\sigma}$$

# In[ ]:


X_scaled = preprocessing.scale(X_train)
X_scaled


# Scaled data has zero mean and unit variance (unit variance means variance = 1):

# In[ ]:


X_scaled.mean(axis=0)


# In[ ]:


X_scaled.std(axis=0)


# Keeping in mind that if you have scaled your training data, you must do likewise with your test data as well. However, your assumption is that the mean and variance must be invariant between your train and test data. `scikit-learn` assists with a built-in utility function `StandardScaler`.

# In[ ]:


scaler = preprocessing.StandardScaler().fit(X_train)
scaler


# In[ ]:


scaler.mean_


# In[ ]:


scaler.scale_


# In[ ]:


scaler.transform(X_train)


# In[ ]:


plt.figure(figsize=(8,6))
plt.hist(X_train);


# You can now utilise the `transform` for new dataset

# In[ ]:


X_test = [[-1., 1., 0.]]


# In[ ]:


scaler.transform(X_test)


# ***

# # Min-Max or Scaling Features to a Range
# 
# Scaling features to lie between a given minimum and maximum value, often between zero and one, or so that the maximum absolute value of each feature is scaled to unit size.
# 
# The motivation to use this scaling include robustness to very small standard deviations of features and preserving zero entries in sparse data.

# ## MinMaxScaler
# 
# Scale a data to the `[0, 1]` range:

# In[ ]:


X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])


# In[ ]:


min_max_scaler = preprocessing.MinMaxScaler()


# In[ ]:


X_train_minmax = min_max_scaler.fit_transform(X_train)


# In[ ]:


X_train_minmax


# Now to unseen data

# In[ ]:


X_test = np.array([[-3., -1.,  0.], [2., 1.5, 4.]])


# In[ ]:


X_test_minmax = min_max_scaler.transform(X_test)


# In[ ]:


X_test_minmax


# doc:
# 
# Init signature: preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
# 
# Transforms features by scaling each feature to a given range.
# 
# This estimator scales and translates each feature individually such
# that it is in the given range on the training set, i.e. between
# zero and one.
# 
# The transformation is given by::
# 
#     X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
#     X_scaled = X_std * (max - min) + min
#     
# $$X_{std}=\frac{X-X_{min}}{X_{max}-X_{min}}$$
# 
# $$X'=X_{std} (\text{max} - \text{min}) + \text{min}$$

# In[ ]:


get_ipython().run_line_magic('pinfo', 'preprocessing.MinMaxScaler')


# ## MaxAbsScaler
# 
# Works in a very similar fashion, but scales in a way that the training data lies within the range `[-1, 1]` by dividing through the largest maximum value in each feature. It is meant for data that is already centered at zero or sparse data.

# In[ ]:


X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])


# In[ ]:


max_abs_scaler = preprocessing.MaxAbsScaler()
X_train_maxabs = max_abs_scaler.fit_transform(X_train)
X_train_maxabs 


# In[ ]:


X_test = np.array([[ -1., -0.5,  2.], [0., 0.5, -0.6]])
X_test_maxabs = max_abs_scaler.transform(X_test)
X_test_maxabs  


# ## Scaling sparse data

# Centering sparse data would destroy the sparseness structure in the data, and thus rarely is a sensible thing to do. 
# 
# However, it can make sense to scale sparse inputs, especially if features are on different scales.
# 
# `MaxAbsScaler` and `maxabs_scale` were specifically designed for scaling sparse data

# [Compare the effect of different scalers on data with outliers](http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py)

# ## Scaling vs Whitening
# 
# It is sometimes not enough to center and scale the features independently, since a downstream model can further make some assumption on the linear independence of the features.
# 
# To address this issue you can use `sklearn.decomposition.PCA` or `sklearn.decomposition.RandomizedPCA` with `whiten=True` to further remove the linear correlation across features.

# ***

# # Normalization

# Normalization is the process of scaling individual samples to have unit norm. 
# 
# This process can be useful if you plan to use a quadratic form such as the dot-product or any other kernel to quantify the similarity of any pair of samples.
# 
# $$X'=\frac{X-X_{mean}}{X_{max}-X_{min}}$$
# 
# This assumption is the base of the Vector Space Model often used in text classification and clustering contexts.
# 
# There are two types of Normalization
# 
#   1. **L1 normalization**, Least Absolute Deviations
# Ensure the sum of absolute values is 1 in each row. 
# 
#   2. **L2 normalization**, Least squares, 
# Ensure that the sum of squares is 1.

# In[ ]:


X = [[ 1., -1.,  2.],
     [ 2.,  0.,  0.],
     [ 0.,  1., -1.]]
X_normalized = preprocessing.normalize(X, norm='l2')

X_normalized    


# Alternatively
# 
# The `preprocessing` module further provides a utility class `Normalizer` that implements the same operation using the `Transformer` API.

# In[ ]:


normalizer = preprocessing.Normalizer().fit(X)  # fit does nothing
normalizer


# In[ ]:


normalizer.transform(X)


# In[ ]:


normalizer.transform([[-1.,  1., 0.]])  


# # Binarization
# 
# $$f(x)={0,1}$$
# 
# Feature binarization is the process of thresholding numerical features to get boolean values. This can be useful for downstream probabilistic estimators that make assumption that the input data is distributed according to a multi-variate Bernoulli distribution
# 
# 
# It is also common among the text processing community to use binary feature values (probably to simplify the probabilistic reasoning) even if normalized counts (a.k.a. term frequencies) or TF-IDF valued features often perform slightly better in practice.

# In[ ]:


X = [[ 1., -1.,  2.],
     [ 2.,  0.,  0.],
     [ 0.,  1., -1.]]

binarizer = preprocessing.Binarizer().fit(X)  # fit does nothing
binarizer


# In[ ]:


binarizer.transform(X)


# Modifying the threshold

# In[ ]:


binarizer = preprocessing.Binarizer(threshold=-0.5)


# In[ ]:


binarizer.transform(X)


# ***

# # Encoding categorical features

# [LabelEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)

# In[ ]:


source = ['australia', 'singapore', 'new zealand', 'hong kong']


# In[ ]:


label_enc = preprocessing.LabelEncoder()
src = label_enc.fit_transform(source)


# In[ ]:


print("country to code mapping:\n") 
for k, v in enumerate(label_enc.classes_): 
    print(v,'\t', k) 


# In[ ]:


test_data = ['hong kong', 'singapore', 'australia', 'new zealand']


# In[ ]:


result = label_enc.transform(test_data) 


# In[ ]:


print(result)


# ## One Hot / One-of-K Encoding
# 
# * Useful for dealing with sparse matrix
# * uses [one-of-k scheme](http://code-factor.blogspot.sg/2012/10/one-hotone-of-k-data-encoder-for.html)
# 
# 
# The process of turning a series of categorical responses into a set of binary result (0 or 1)

# [One Hot Encoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder)

# In[ ]:


source


# In[ ]:


src


# In[ ]:


from sklearn.preprocessing import OneHotEncoder


# In[ ]:


one_hot_enc = OneHotEncoder(sparse=False, categories='auto')
src = src.reshape(len(src), 1)
one_hot = one_hot_enc.fit_transform(src)
print(one_hot)


# In[ ]:


invert_res = label_enc.inverse_transform([np.argmax(one_hot[0, :])])
print(invert_res)


# In[ ]:


invert_res = label_enc.inverse_transform([np.argmax(one_hot[3, :])])
print(invert_res)


# # References
# 
# * [Section - Should I normalize/standardize/rescale the data?](http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html)
# * [Colleen Farrelly - Machine Learning by Analogy](https://www.slideshare.net/ColleenFarrelly/machine-learning-by-analogy-59094152)
# * [Lior Rokach - Introduction to Machine Learning](https://www.slideshare.net/liorrokach/introduction-to-machine-learning-13809045)
# * [Ritchie Ng](http://www.ritchieng.com/machinelearning-one-hot-encoding/)

# ***
