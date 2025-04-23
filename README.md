# ANYL-605-Pre-Capstone-Project


## Data cleaning

* Standardize Columns: change to lower case and use _ between words instead of .
![image](https://github.com/user-attachments/assets/a79b026d-ef13-42d4-b68c-4d13a403aa83)
* Clean out NA values: 288 NA values. Total drop of 33 out of 53940 rows
![image](https://github.com/user-attachments/assets/fd0abfc2-76ac-4214-ad83-b3ae446cf2c9)
* Standardize the values for the categorical variables: ex promotion had values of "Yes", "No", "No ", and ""
![image](https://github.com/user-attachments/assets/706693c2-fdb5-468c-83c9-12ab0a1e09e2)
* Dropped rows where values were missing: out of 53,000 rows, the ones missing data was insignificant and therefore removed from analysis
* Removed Outliers as they skew the analysis

Data is now prepped and ready for Analysis
![image](https://github.com/user-attachments/assets/0068cfcc-b323-4b1f-91d6-24ef4b8188d3)



## Linear Regression

Multicollinearity exists in the linear regression model.

![image](https://github.com/user-attachments/assets/45aecd46-14d5-42f7-bc7f-a063e2688191)

It is unsurprising that carat, length_mm, width_mm, and depth_mm are all highly correlated as there is most likely a strong connection between the sizes. 

Possible solutions would be to combine length, width, and depth into a volume column or drop them completely and only use carat as that is usually the reference to diamond sizing. However, running a linear regression analysis with the volume variable still shows high VIF values and correlation with Carat, and the model with the variables dropped completely has an increase in the AIC value.

Instead, we will go ahead with Ridge Regression and Lasso Regression to address the multicollinearity and determine the best fit model. 

## Ridge Regression 

* best lambda for ridge regression<br>
![image](https://github.com/user-attachments/assets/8ba9e89e-9996-49f6-b1f1-65123f08a81f)

* coefficient values<br>
![image](https://github.com/user-attachments/assets/2b8b059c-71b6-4602-ba82-7f55d32e258e)

* RMSE for Model Comparison<br>
![image](https://github.com/user-attachments/assets/90ec3760-296d-4d00-9ab5-f6bc5e5c7150)

* R^2<br>
![image](https://github.com/user-attachments/assets/8d447a70-90c2-49ed-bce0-06c56230261d)


## Lasso Regression

* coefficient values<br>
![image](https://github.com/user-attachments/assets/68060a7e-fd2a-4474-85fe-922753fb0601)

* RMSE for Model Comparison<br>
![image](https://github.com/user-attachments/assets/302a425a-5fb9-4224-9be3-7849d0d62020)

* R^2<br>
![image](https://github.com/user-attachments/assets/465c0cf4-da3b-46f4-a73b-292804c5ece1)


## Best Model

![image](https://github.com/user-attachments/assets/c09d7ece-f22e-4fd2-a099-3e2a304c539e)

Given the RSMEs are incredible close as are the R^2 values of the Lasso, Ridge, and Linear Regression, we should select the model that handles the collinearity and provides the most insight in terms of significant variables. The ___ model delivers on both of those criteria. 

## Challenges:

Heterscedasticity occurs in the data. 
We have tried weighted least squares and log transformations without success. However, because the only real difference is seen with higher prices, $30,000 and above, and there are fewer data points at this extreme, we will continue with analysis.

![image](https://github.com/user-attachments/assets/38a7d90c-e69c-4120-92fc-6bec77481182)


## 2a. What drives consumers to pay a higher price for a specific piece of diamond?



## 2b. Can a specific characteristic of a diamond may have a negative impact on consumers’ willingness to play?



## 2c. Should they change geographical sourcing decisions?



## 2d. How should they plan for distribution in future?



## 2e. What is an optimal combination of carat, length (mm), width (mm), and depth (mm)?



## 2f. What specific strategies you suggest such that they can increase the sales?


