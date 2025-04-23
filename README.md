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
![image](https://github.com/user-attachments/assets/aad82ca7-064c-47a9-9e99-4e4cb9b7bb6e)

* RMSE for Model Comparison
![image](https://github.com/user-attachments/assets/f94ec5b2-e87b-4eaa-9836-dadec2126f88)

* R^2<br>
![image](https://github.com/user-attachments/assets/6e65b24a-8df4-4d86-bb59-62e9544aba6e)


## Lasso Regression


## Best Model
![image](https://github.com/user-attachments/assets/c60fa380-c8d3-4196-a54d-b38a2c7c9242)


Given the RSMEs are incredible close as are the R^2 values of the Lasso and Linear Regression, we should select the model that handles the collinearity and provides the most insight in terms of significant variables. The Lasso model delivers on both of those criteria. 




