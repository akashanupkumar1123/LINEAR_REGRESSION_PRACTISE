Data Pre-Processing¶



Without Pre-processing¶




With Pre-processing¶


Data Pre-processing
Standardization / Mean Removal

Min-Max or Scaling Features to a Range

Normalization

Binarization

Assumptions:

Implicit/explicit assumption of machine learning algorithms: The features follow a normal distribution.
Most method are based on linear assumptions
Most machine learning requires the data to be standard normally distributed. Gaussian with zero mean and unit variance.
scikit-learn:

In practice we often ignore the shape of the distribution and just transform the data to center it by removing the mean value of each feature, then scale it by dividing non-constant features by their standard deviation.

For instance, many elements used in the objective function of a learning algorithm (such as the RBF kernel of Support Vector Machines or the l1 and l2 regularizers of linear models) assume that all features are centered around zero and have variance in the same order. If a feature has a variance that is orders of magnitude larger than others, it might dominate the objective function and make the estimator unable to learn from other features correctly as expected.



Standardization / Mean Removal / Variance Scaling
scikit Scale

Mean is removed. Data is centered on zero. This is to remove bias.

Standardization of datasets is a common requirement for many machine learning estimators implemented in scikit-learn; they might behave badly if the individual features do not more or less look like standard normally distributed data: Gaussian with zero mean and unit variance. "standard normal" random variable with mean 0 and standard deviation 1.

𝑋′=𝑋−𝑋¯𝜎

Keeping in mind that if you have scaled your training data, you must do likewise with your test data as well. However, your assumption is that the mean and variance must be invariant between your train and test data. scikit-learn assists with a built-in utility function StandardScaler.




Min-Max or Scaling Features to a Range
Scaling features to lie between a given minimum and maximum value, often between zero and one, or so that the maximum absolute value of each feature is scaled to unit size.

The motivation to use this scaling include robustness to very small standard deviations of features and preserving zero entries in sparse data.


Init signature: preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)

Transforms features by scaling each feature to a given range.

This estimator scales and translates each feature individually such that it is in the given range on the training set, i.e. between zero and one.

The transformation is given by::

X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
X_scaled = X_std * (max - min) + min
𝑋𝑠𝑡𝑑=𝑋−𝑋𝑚𝑖𝑛𝑋𝑚𝑎𝑥−𝑋𝑚𝑖𝑛
𝑋′=𝑋𝑠𝑡𝑑(max−min)+min
preprocessing.MinMaxScaler?



MaxAbsScaler
Works in a very similar fashion, but scales in a way that the training data lies within the range [-1, 1] by dividing through the largest maximum value in each feature. It is meant for data that is already centered at zero or sparse data.



Scaling sparse data
Centering sparse data would destroy the sparseness structure in the data, and thus rarely is a sensible thing to do.

However, it can make sense to scale sparse inputs, especially if features are on different scales.

MaxAbsScaler and maxabs_scale were specifically designed for scaling sparse data

Scaling vs Whitening
It is sometimes not enough to center and scale the features independently, since a downstream model can further make some assumption on the linear independence of the features.

To address this issue you can use sklearn.decomposition.PCA or sklearn.decomposition.RandomizedPCA with whiten=True to further remove the linear correlation across features.




Normalization
Normalization is the process of scaling individual samples to have unit norm.

This process can be useful if you plan to use a quadratic form such as the dot-product or any other kernel to quantify the similarity of any pair of samples.

𝑋′=𝑋−𝑋𝑚𝑒𝑎𝑛𝑋𝑚𝑎𝑥−𝑋𝑚𝑖𝑛
 
This assumption is the base of the Vector Space Model often used in text classification and clustering contexts.

There are two types of Normalization

L1 normalization, Least Absolute Deviations Ensure the sum of absolute values is 1 in each row.

L2 normalization, Least squares, Ensure that the sum of squares is 1.


Alternatively

The preprocessing module further provides a utility class Normalizer that implements the same operation using the Transformer API.



Binarization
𝑓(𝑥)=0,1
 
Feature binarization is the process of thresholding numerical features to get boolean values. This can be useful for downstream probabilistic estimators that make assumption that the input data is distributed according to a multi-variate Bernoulli distribution



It is also common among the text processing community to use binary feature values (probably to simplify the probabilistic reasoning) even if normalized counts (a.k.a. term frequencies) or TF-IDF valued features often perform slightly better in practice.


One Hot / One-of-K Encoding
Useful for dealing with sparse matrix
uses one-of-k scheme
The process of turning a series of categorical responses into a set of binary result (0 or 1)











