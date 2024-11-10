Regularization with SciKit-Learn

--Previously we created a new polynomial feature set and then applied our standard linear regression on it, but we can be smarter about model   choice and utilize regularization.

-----Regularization attempts to minimize the RSS (residual sum of squares) and a penalty factor. This penalty factor will penalize models that have         coefficients that are too large. Some methods of regularization will actually cause non useful features to have a coefficient of zero, in which case        the model does not consider the feature.

-------Let's explore two methods of regularization, Ridge Regression and Lasso. We'll combine these with the polynomial feature set (it wouldn't be            as effective to perform regularization of a model on such a small original feature set of the original X).


Scaling the Data-----

    While our particular data set has all the values in the same order of magnitude ($1000s of dollars spent), typically that won't be the case on a     dataset, and since the mathematics behind regularized models will sum coefficients together, its important to standardize the features. Review     the theory videos for more info, as well as a discussion on why we only fit to the training data, and transform on both sets separately.


Performed RIDGE_REGRESSION


from sklearn.linear_model import Ridge


Calculate MAE, MSE, RMSE


--- Chossing an alpha calue with Cross- Validation


from sklearn.linear_model import RidgeCV

Choosing a scoring: https://scikit-learn.org/stable/modules/model_evaluation.html
# Negative RMSE so all metrics follow convention "Higher is better"

# See all options: sklearn.metrics.SCORERS.keys()
ridge_cv_model = RidgeCV(alphas=(0.1, 1.0, 10.0),scoring='neg_mean_absolute_error')



Calculated MAE, MSE, RMSE for the RidgeCV_model





Performed LASSO_REGRESSION


from sklearn.linear_model import LassoCV


calculate MAE, MSE, RMSE


choosing the best alpha to fit the parameters after fit fucntion

lasso_cv_model.alpha_




-- calculated MAE , MSE, RMSE


--finally caculated Coeffients of the model

lasso_cv_model.coef_





-----performed ELASTIC_NET

Elastic Net combines the penalties of ridge regression and lasso in an attempt to get the best of both worlds.





from sklearn.linear_model import ElasticNetCV


elastic_model = ElasticNetCV(l1_ratio=[.1, .5, .7,.9, .95, .99, 1],tol=0.01)


elastic_model.l1_ratio_


performed MAE, MSE,RMSE 



finally calculated coeffients of the model---

elastic_model.coef_
