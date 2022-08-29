# Predicting-Bike-Sharing-Demand-
My project is solution for [prontonx kaggle 01 competition][1]
In this project, I did: 
* Processed data and feature engineering
* Build a linear regression, and a neural network regression using TensorFlow.
* Handled negative predict values.
* Created Random Forest, and CatBoost using scikit-learn, and CatBoost API.
* Grid search best params for CatBoost model.</br>
Detail in python notebook.</br>
Result: 

|                   | Private test | Public test |
|-------------------|--------------|-------------|
| Linear Regression | 30414.68756  | 31727.94172 |
| ANN               | 2866.88695   | 2572.43681  |
| Random Forest     | 2135.46164   | 2048.17455  |
| CatBoost          | 1400.04273   | 1288.63043  |

* Evaluate in RMSE loss

[1]: https://www.kaggle.com/competitions/protonx-tf04-linear-regression/overview
