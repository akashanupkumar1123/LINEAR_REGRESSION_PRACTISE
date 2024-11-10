Regression
Boston Housing Price Prediction¶



Code	Description
CRIM	per capita crime rate by town
ZN	proportion of residential land zoned for  lots over 25,000 sq.ft.
INDUS	proportion of non-retail business acres per  town
CHAS	Charles River dummy variable (= 1 if tract  bounds river; 0 otherwise)
NOX	nitric oxides concentration (parts per 10 million)
RM	average number of rooms per dwelling
AGE	proportion of owner-occupied units built prior to 1940
DIS	weighted distances to five Boston employment centres
RAD	index of accessibility to radial highways
TAX	full-value property-tax rate per $10,000
PTRATIO	pupil-teacher ratio by town
B	1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
LSTAT	% lower status of the population
MEDV	Median value of owner-occupied homes in $1000's





.. _boston_dataset:

Boston house prices dataset
---------------------------

**Data Set Characteristics:**  

    :Number of Instances: 506 

    :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.

    :Attribute Information (in order):
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's

    :Missing Attribute Values: None

    :Creator: Harrison, D. and Rubinfeld, D.L.

This is a copy of UCI ML housing dataset.
https://archive.ics.uci.edu/ml/machine-learning-databases/housing/


This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.

The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
prices and the demand for clean air', J. Environ. Economics & Management,
vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
...', Wiley, 1980.   N.B. Various transformations are used in the table on
pages 244-261 of the latter.

The Boston house-price data has been used in many machine learning papers that address regression
problems.   
     
.. topic:: References

   - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.
   - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.


Basics of the API:

Most commonly, the steps in using the Scikit-Learn estimator API are as follows (we will step through a handful of detailed examples in the sections that follow):

Choose a class of model by importing the appropriate estimator class from Scikit- Learn.
Choose model hyperparameters by instantiating this class with desired values.
Arrange data into a features matrix and target vector.
Fit the model to your data by calling the fit() method of the model instance.
Apply the model to new data:
For supervised learning, often we predict labels for unknown data using the predict() method.
For unsupervised learning, we often transform or infer properties of the data using the transform() or predict() method.




Robust Regression¶

RANdom SAmple Consensus (RANSAC) Algorithm
link = http://scikit-learn.org/stable/modules/linear_model.html#ransac-regression

Each iteration performs the following steps:

Select min_samples random samples from the original data and check whether the set of data is valid (see is_data_valid).

Fit a model to the random subset (base_estimator.fit) and check whether the estimated model is valid (see is_model_valid).

Classify all data as inliers or outliers by calculating the residuals to the estimated model (base_estimator.predict(X) - y) - all data samples with absolute residuals smaller than the residual_threshold are considered as inliers.

Save fitted model as best model if number of inlier samples is maximal. In case the current estimated model has the same number of inliers, it is only considered as the best model if it has better score.




Performance Evaluation of Regression Model



---Method 1: Residual Analysis


----Method 2: Mean Squared Error (MSE)¶
        𝑀𝑆𝐸=1𝑛∑𝑖=1𝑛(𝑦𝑖−𝑦̂ 𝑖)2



----Method 3: Coefficient of Determination,  𝑅2        r2=1−𝑆𝑆𝐸𝑆𝑆𝑇




What does a Near Perfect Model Looks like?¶




------Method 1: Residual Analysis



-------Method 2: Mean Squared Error (MSE)




------Method 3: Coefficient of Determination,  𝑅2 













