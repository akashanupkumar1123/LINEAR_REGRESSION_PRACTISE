Regularized Method for Regression




Ridge Regression
Source: scikit-learn

Ridge regression addresses some of the problems of Ordinary Least Squares by imposing a penalty on the size of coefficients. The ridge coefficients minimize a penalized residual sum of squares,

min𝑤||||𝑋𝑤−𝑦||||22+𝛼||||𝑤||||22
𝛼>=0 is a complexity parameter that controls the amount of shrinkage: the larger the value of 𝛼, the greater the amount of shrinkage and thus the coefficients become more robust to collinearity.

Ridge regression is an L2 penalized model. Add the squared sum of the weights to the least-squares cost function.

Shows the effect of collinearity in the coefficients of an estimator.

Ridge Regression is the estimator used in this example. Each color represents a different feature of the coefficient vector, and this is displayed as a function of the regularization parameter.

This example also shows the usefulness of applying Ridge regression to highly ill-conditioned matrices. For such matrices, a slight change in the target variable can cause huge variances in the calculated weights. In such cases, it is useful to set a certain regularization (alpha) to reduce this variation (noise).



LASSO Regression
A linear model that estimates sparse coefficients.

Mathematically, it consists of a linear model trained with ℓ1 prior as regularizer. The objective function to minimize is:

min𝑤12𝑛𝑠𝑎𝑚𝑝𝑙𝑒𝑠||||𝑋𝑤−𝑦||||22+𝛼||||𝑤||||1
The lasso estimate thus solves the minimization of the least-squares penalty with 𝛼||||𝑤||||1 added, where 𝛼 is a constant and ||||𝑤||||1 is the ℓ1−𝑛𝑜𝑟𝑚 of the parameter vector.

Elastic Net
A linear regression model trained with L1 and L2 prior as regularizer.

This combination allows for learning a sparse model where few of the weights are non-zero like Lasso, while still maintaining the regularization properties of Ridge.

Elastic-net is useful when there are multiple features which are correlated with one another. Lasso is likely to pick one of these at random, while elastic-net is likely to pick both.

A practical advantage of trading-off between Lasso and Ridge is it allows Elastic-Net to inherit some of Ridge’s stability under rotation.

The objective function to minimize is in this case

min𝑤12𝑛𝑠𝑎𝑚𝑝𝑙𝑒𝑠||||𝑋𝑤−𝑦||||22+𝛼𝜌||||𝑤||||1+𝛼(1−𝜌)2||||𝑤||||22




When should I use Lasso, Ridge or Elastic Net?

Ridge regression can't zero out coefficients; You either end up including all the coefficients in the model, or none of them.

LASSO does both parameter shrinkage and variable selection automatically.

If some of your covariates are highly correlated, you may want to look at the Elastic Net instead of the LASSO.



