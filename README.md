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
![image](https://github.com/user-attachments/assets/8c586f0c-5853-4340-af16-320dea80c9f5)


## Linear Regression

Multicollinearity exists in the linear regression model.

![image](https://github.com/user-attachments/assets/45aecd46-14d5-42f7-bc7f-a063e2688191)

It is unsurprising that carat, length_mm, width_mm, and depth_mm are all highly correlated as there is most likely a strong connection between the sizes. 

Possible solutions would be to combine length, width, and depth into a volume column or drop them completely and only use carat as that is usually the reference to diamond sizing. However, running a linear regression analysis with the volume variable still shows high VIF values and correlation with Carat, and the model with the variables dropped completely has an increase in the AIC value.

Instead, we will go ahead with Ridge Regression and Lasso Regression to address the multicollinearity and determine the best fit model. 

## Ridge Regression 

* best lambda for ridge regression<br>
![image](https://github.com/user-attachments/assets/38cba595-21b5-4a51-93ae-b006dcb5d4a0)

* coefficient values
![image](https://github.com/user-attachments/assets/f68d8696-a6b9-4186-9964-cff72f6844f1)

* RMSE for Model Comparison
![image](https://github.com/user-attachments/assets/f94ec5b2-e87b-4eaa-9836-dadec2126f88)

* R^2<br>
![image](https://github.com/user-attachments/assets/6e65b24a-8df4-4d86-bb59-62e9544aba6e)


## Lasso Regression

* coefficient values
![image](https://github.com/user-attachments/assets/cf402559-cb69-4153-ab1f-b1da1a61da7d)

* RMSE for Model Comparison<br>
![image](https://github.com/user-attachments/assets/0f1fe3d4-5709-46c6-ad44-bbf31e060662)

* R^2<br>
![image](https://github.com/user-attachments/assets/85eacd07-096b-42d8-affe-c0528b1e85db)


## Best Model

![image](https://github.com/user-attachments/assets/3d5c4148-c66b-4df4-a397-b438ba7e2acf)


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


