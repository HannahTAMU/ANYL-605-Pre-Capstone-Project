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

It is unsurprising that carat, length_mm, width_mm, and depth_mm are all highly correlated as there carat is a product of the dimensions.

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

Given the RSMEs are incredible close as are the R^2 values of the Lasso, Ridge, and Linear Regression, we should select the model that handles the collinearity and provides the most insight in terms of significant variables. The ridge model delivers on both of those criteria. 

## Challenges:

Heteroscedasticity is present in the data, and despite applying various corrective techniques, including weighted least squares, log transformations, and ridge regression with quadratic terms, it remains unresolved. Additionally, even when restricting the model to only carat while excluding width, length, and depth (mm), heteroscedasticity persisted. However, the primary variation occurs at higher price points, where data is sparse. Given that the model’s results align with logical expectations, showing that higher quality diamonds correspond to higher prices, we can confidently proceed with the analysis, as it still provides meaningful insights.<br><br>
It is important to note that higher-priced diamonds may be influenced by factors beyond the available data, such as brand recognition and consumer psychology. Clients purchasing high-value diamonds often belong to different demographic segments, where purchasing decisions are not made in the same way as those buying lower-priced diamonds. Their willingness to spend is likely driven by prestige, exclusivity, and personal or cultural significance rather than purely rational cost-benefit analysis. This difference in purchasing behavior may explain some of the variance in pricing at the higher end, which is not fully captured in the current model.

![image](https://github.com/user-attachments/assets/38a7d90c-e69c-4120-92fc-6bec77481182)


## 2a. What drives consumers to pay a higher price for a specific piece of diamond?

* Carat weight (11524.36) is the most influential factor
  * Carat has the strongest positive effect on sales price.
  * Larger diamonds command higher prices, although diminishing returns may exist at extreme carat values.
 
* Cut (base = fair)
  * Ideal (+772.37) > Premium (+722.41) > Very Good (+650.61) > Good (+514.04)
  * Higher-rated cutes increase price. The better the cut, the higher the price. Ideal cuts maximize brilliance and attract more consumer demand.

* Diamond Color (base = J (worst))
  * D (+2353.03) > E (+2152.12) > F (+2099.41) > G (+1893.37) > H (+1392.14) > I (+904.64)
  * Premimum colors significanly increase price. Consumers prefer colorless diamonds.

* Clarity (base = I1 (worst))
  * IF (Internally Flawless) (+5252.47) > VVS1 (+4933.09) > VVS2 (+4898.15) > VS1 (+4523.54) > VS2 (+4220.27) > SI1 (+3624.15) > SI2 (+2674.31)
  * Major Pricing Factor
  * Higher Clarity = Higher Price

* Dimensions Width mm (+1170.97)
  * Strong positive influence on price
  * Increasing width will increase price
 

## 2b. Can a specific characteristic of a diamond may have a negative impact on consumers’ willingness to play?

Yes. Certain characteristics of a diamond can have a negative impact on consumers' willingness to pay. Some have stronger and weaker impacts.

* Dimensions (mm)
  * Length mm (-1023.26)
    * Moderate negative influence on price. Excessive length is negatively associated with price.
  * Depth mm (-2102.15)
    * Strong negative influnce on price.
    * As depth increases by one unit, price will decrease by 2102.15

## 2c. Should they change geographical sourcing decisions?

The impact of country of origin on price is relatively small. However, there are countries with slightly negative effects compared to Brazil, like Cuba (-29.11) and Sweden (-1.35). Whereas, some countries have slightly positive effects compared to Brazil, like India (+11.22), Italy (+19.39), and the Netherlands (+16.59). Targeting regions with positive coefficients may yield better returns, however this is not a priority. 

## 2d. How should they plan for distribution in future?

* Optimize digital presence
  * Online Listings (+3.84) have a slight positive effect that suggests online sales do not hurt pricing.
* Be cautious with promotions
  * Promotional Pricing (-6.97) has a slight negative effect. Careful discounting strategies are needed to avoid undermining profit.

## 2e. What is an optimal combination of carat, length (mm), width (mm), and depth (mm)?

* Coefficients with quadratic terms<br>
![image](https://github.com/user-attachments/assets/541c7fd9-724c-49ff-b7c4-2dd1f7fe8b20)

* RMSE<br>
![image](https://github.com/user-attachments/assets/93448852-645b-45c6-b814-bcb16c588693)

* R-sq<br>
![image](https://github.com/user-attachments/assets/1e35098e-5c42-4135-b6e7-ca8f7be4136a)

* Optimal Values:<br>
![image](https://github.com/user-attachments/assets/55687034-f590-4131-aaf7-73b1ded02c6c)

* Carat (9653.46) and carat^2 (-2045.57):
  * Positive carat with negative carat^2 indicates there is an optimal carat weight. The optimal weight is 2.359606
  * Sales will increase as carat weight increases towards the optimal value but will decrease as it surpasses the optimal value.
* Length_mm (-4380.84) and length_mm^2 (275.24)
  * Negative length and positive length^2 indicates a minimum value of 7.96 mm.
  * Sales will decrease as length_mm increases to the minimum value but will increase once length_mm surpasses the minimum value.
* width_mm (-2457.99) and width_mm^2 (442.63)
  * Negative width and positive width^2 indicates a minimum value of 2.78 mm.
  * Sales will decrease as width_mm increases to the minimum value but will increase once width_mm surpasses the minimum value.
* depth_mm (-4544.11) and depth_mm^2 (532.33)
  * Negative depth and positive depth^2 indicates a minimum value of 4.27 mm.
  * Sales will decrease as depth_mm increases to the minimum value but will increase once depth_mm surpasses the minimum value.

Optimal combination: <br>
Carat = 2.357, length_mm > 7.96 mm, width_mm > 2.78 mm, depth_mm > 4.27 mm


## 2f. What specific strategies you suggest such that they can increase the sales?

1) Diamond Dimensions Matter​
Optimizing for carat, length, width, and depth can yield higher sales prices & profitable returns

2) Enhance digital presence ​
Online Listings (+3.84) have a slight positive effect.  Digital presence should utilized relative to marketing costs.

3) Source High Quality Colors and Clarity​
Color and Clarity can bring in high prices. Source these from pre-existing supply chains like Britain and the Netherlands. Analyze global diamond capacity & costs for strategic long-term sourcing.

4) Focus on Cut Quality ​
Higher grades of cut can bring high price returns for low cost. Invest in knowledgeable jewelers and technology for best yields.

## ADDITIONAL ANALYSIS - What is a realistic diamond dimension that maximizes carat size(the most important factor for sales price)?
Determining ideal round diamond sizes
Reference the file: Ideal_Diamond_Realistic.R

![image](https://github.com/user-attachments/assets/29f5da97-7ec7-46c3-88c8-d94f58447b33)

Given that most of the diamonds length and width are close to equal we can assume diamonds are typically round.
We need to determine the ideal size of a round cut diamond.
According to a diamond carat calculator app(Diamond Carat Calculator (Diamond Weight))
https://www.omnicalculator.com/other/diamond-carat
the formula for determining carat size from diamond dimensions for round cut diamonds is Diameter² × Depth × 0.0061 × (1 + GT). 
GT(Girdle Thickness Factor) usually lies between 1% (0.01) and 3% (0.03) for round diamonds. 
Ideal Diamond Cuts have an ideal depth percentage of 59-62.6%. This percentage is determined by dividing the diamond’s height by its width.
(Ideal Diamond Depth and Table by Cut | The Diamond Pro)
https://www.diamonds.pro/education/diamond-depth-and-table/.
Checking the GT range with the carat size set to the maximum carat value of 2.35
we can test for the diameter sizes between our minimum width(the smaller of the two diameter dimensions to capture all possibilites) that result in a 
depth that is above the minimum depth and within the ideal depth percentage of 59-62.6%:
![image](https://github.com/user-attachments/assets/954187f9-18b3-4cea-a654-f144851ab153)
  



