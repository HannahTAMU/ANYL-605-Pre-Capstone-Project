################################################################################

### Context 2 ###

################################################################################

### INSTALL PACKAGES ###
install.packages(c("tidyverse", "janitor", "dplyr", "car")) 

### LOAD LIBRARIES ###
library(tidyverse)
library(janitor)
library(car)
library(ggplot2)
library(glmnet)

################################################################################

### LOAD DATA ###

diamonds <- read.csv("C:/Users/rache/OneDrive - Texas A&M University/MS Analytics/Spring 2025/ANLY 605/pre-capstone/SD Data.csv", header = TRUE)

################################################################################

### DATA CLEANING ###

# Clean column names
names(diamonds)
diamonds <- clean_names(diamonds)

# Remove NA values
sum(is.na(diamonds)) #288 NAs
dim(diamonds) #53940 rows
diamonds <- na.omit(diamonds)
dim(diamonds) # 53907 - dropped 33 rows

### EXPLORING PROMOTION ###

unique(diamonds$promotion)
sum(diamonds$promotion == "") #3320 entries with ""

# Trim extra spaces and remove blank entries
diamonds$promotion <- trimws(diamonds$promotion)
diamonds <- diamonds %>% filter(promotion != "")
unique(diamonds$promotion) # only Yes and No now

### Exploring Dimensions ###

### REMOVING WIDTH OUTLIERS ###
width_outliers <- diamonds %>%
  filter(width_mm > length_mm * 1.50)
View(width_outliers)

diamonds <- diamonds %>% filter(width_mm <= length_mm * 1.5)

# Visualize width vs length, colored by sales price
ggplot(diamonds, aes(x = width_mm, y = length_mm, color = sales_price)) +
  geom_point(alpha = 0.5) +
  labs(
    title = "Scatter Plot of Width vs Length Colored by Sales Price",
    x = "Width (mm)",
    y = "Length (mm)",
    color = "Sales Price"
  ) +
  theme_minimal() +
  scale_color_gradient(low = "blue", high = "red")

### Remove Entries with Zero Dimensions ###

diamonds_missing_dimensions <- diamonds %>%
  filter(width_mm == 0 | length_mm == 0 | depth_mm == 0)
View(diamonds_missing_dimensions)
nrow(diamonds_missing_dimensions) # 18 rows with 0s in these dimension categories. Drop these

diamonds <- diamonds %>% filter(width_mm != 0 & length_mm != 0 & depth_mm != 0)

### Explore Country of Origin ###
ggplot(diamonds, aes(x = coo, y = carat)) +
  geom_boxplot(fill = "darkgreen") +
  labs(title = "Boxplot of Price by Country of Origin") +
  theme_minimal()

### Convert Character Columns to Factors ###
# Convert categorical variables to factors
diamonds$cut <- factor(diamonds$cut, levels = c("Fair", "Good", "Very Good", "Premium", "Ideal"))
diamonds$color <- factor(diamonds$color, levels = c("J", "I", "H", "G", "F", "E", "D"))
diamonds$clarity <- factor(diamonds$clarity, levels = c("I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"))
diamonds$coo <- factor(diamonds$coo)
diamonds$online <- factor(diamonds$online)
diamonds$promotion <- factor(diamonds$promotion)

summary(diamonds)
#We no longer have any variables with values of 0 so we can proceed

################################################################################

### LINEAR REGRESSION MODEL ###

# Fit the multiple linear regression model
model_price <- lm(sales_price ~ carat + cut + color + clarity + length_mm + 
                    width_mm + depth_mm + depth + table + coo + online + promotion, 
                  data = diamonds)
summary(model_price)
vif(model_price)
AIC(model_price)

# Print VIF values
print(vif_values)

### Highly correlated: width, depth, length with carat - drop length width and depth_mm

# Fit the multiple linear regression model with volume instead of individual dimensions
model_price <- lm(sales_price ~ carat + cut + color + clarity + depth + table + coo + online + promotion, data = diamonds)
vif(model_price)
AIC(model_price)

# AIC went up. let's proceed with ridge and lasso

################################################################################

### Ridge Regression ###
set.seed(123)
# Create model matrix to expand factor levels into dummy variables
x_var <- model.matrix(sales_price ~ .,data = diamonds)  # Remove intercept column
y_var <- diamonds$sales_price

# Define lambda range
lambda_seq <- 10^seq(2, -2, by = -0.1)

# Run ridge regression (alpha = 0 means ridge)
ridge_reg <- glmnet(x_var, y_var, alpha = 0, lambda = lambda_seq)
summary(ridge_reg)

# Cross Validation to find best lambda
ridge_cv <- cv.glmnet(x_var, y_var, alpha = 0, lambda = lambda_seq)
best_lambda <- ridge_cv$lambda.min 
best_lambda
# 0.01

#Best Model 
best_model <- ridge_cv$glmnet.fit 
head(best_model)

# Rebuilding the model with optimal lambda value 
best_ridge <- glmnet(x_var, y_var, alpha = 0, lambda =best_lambda) 
coef(best_ridge)

#Calculate RSME
# Make predictions using the fitted regression model
predicted_Y <- predict(best_ridge, newx = x_var, s = best_lambda)
# Compute RMSE
RMSE_value <- sqrt(mean((y_var - predicted_Y)^2))
# Print RMSE
print(RMSE_value) # RSME = 1117.64

# Compute R^2
SSE <- sum((Y - predicted_Y)^2)  # Sum of Squared Errors
SST <- sum((Y - mean(Y))^2)      # Total Sum of Squares
R2_value_ridge <- 1 - (SSE / SST)       # R^2 formula
print(R2_value_ridge)

################################################################################

### LASSO REGRESSION ###
set.seed(123)

X <- model.matrix(sales_price ~ .,data = diamonds)  # Remove intercept column
Y <- diamonds$sales_price

cv.lambda.lasso <- cv.glmnet(x=X, y=Y,  
                             alpha = 1)  
plot(cv.lambda.lasso)

#Lasso path 
plot(cv.lambda.lasso$glmnet.fit,  
     "lambda", label=FALSE)

l.lasso.min <- cv.lambda.lasso$lambda.min 
lasso.model <- glmnet(x=X, y=Y, 
                      alpha  = 1,  
                      lambda = l.lasso.min) 
lasso.model$beta # coefficients 

#Calculate RSME
# Make predictions using the fitted Lasso model
predicted_Y <- predict(lasso.model, newx = X, s = l.lasso.min)
# Compute RMSE
RMSE_value <- sqrt(mean((Y - predicted_Y)^2))
# Print RMSE
print(RMSE_value) # RSME = 1118.24





































