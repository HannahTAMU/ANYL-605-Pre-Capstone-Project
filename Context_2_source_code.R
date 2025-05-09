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
library(lmtest)

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

# Make predictions using the fitted model
predicted_values <- predict(model_price, newdata = diamonds)

# Compute RMSE
actual_values <- diamonds$sales_price
rmse_value <- sqrt(mean((actual_values - predicted_values)^2))

# Print RMSE
print(rmse_value)

plot(model_price$fitted.values, residuals(model_price), 
     main="Residuals vs Fitted", xlab="Fitted Values", ylab="Residuals")
abline(h=0, col="red")

### Highly correlated: width, depth, length with carat - drop length width and depth_mm

################################################################################

### Ridge Regression ###
set.seed(123)
# Create model matrix to expand factor levels into dummy variables
x_var <- model.matrix(sales_price ~ carat + cut + color + clarity + length_mm + 
                        width_mm + depth_mm + depth + table + coo + online + promotion,data = diamonds)  # Remove intercept column
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
RMSE_value_ridge <- sqrt(mean((y_var - predicted_Y)^2))
# Print RMSE
print(RMSE_value_ridge) 

# Compute R^2
SSE <- sum((y_var - predicted_Y)^2)  # Sum of Squared Errors
SST <- sum((y_var - mean(y_var))^2)      # Total Sum of Squares
R2_value_ridge <- 1 - (SSE / SST)       # R^2 formula
print(R2_value_ridge)

# Compute residuals
residuals_ridge <- y_var - predicted_Y

ggplot(data = data.frame(predicted_Y, residuals_ridge), aes(x = predicted_Y, y = residuals_ridge)) +
  geom_point() +
  geom_smooth(method = "lm", color = "red") +  # Linear smoothing instead of loess
  labs(title = "Ridge Residual Plot", x = "Predicted Sales Price", y = "Residuals") +
  theme_minimal()

################################################################################

### LASSO REGRESSION ###
set.seed(123)

X <- model.matrix(sales_price ~ carat + cut + color + clarity + length_mm + 
                    width_mm + depth_mm + depth + table + coo + online + promotion,data = diamonds)  # Remove intercept column
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
RMSE_value_lasso <- sqrt(mean((Y - predicted_Y)^2))
# Print RMSE
print(RMSE_value_lasso) 

SSE_lasso <- sum((Y - predicted_Y)^2)  
SST_lasso <- sum((Y - mean(Y))^2)  
R2_lasso <- 1 - (SSE_lasso / SST_lasso)  
print(R2_lasso)  

# Concatenate results into a table-like structure
ridge_output <- paste("Ridge Regression | RMSE:", RMSE_value_ridge, "| R²:", R2_value_ridge)
lasso_output <- paste("Lasso Regression | RMSE:", RMSE_value_lasso, "| R²:", R2_lasso)

# Print results
cat(ridge_output, "\n")
cat(lasso_output, "\n")

# Compute residuals
residuals_lasso <- Y - predicted_Y

# Create residual plot
ggplot(data = data.frame(predicted_Y, residuals_lasso), aes(x = predicted_Y, y = residuals_lasso)) +
  geom_point() +
  geom_smooth(method = "lm", color = "red") +  # Use linear smoothing
  labs(title = "Residual Plot", x = "Predicted Sales Price", y = "Residuals") +
  theme_minimal()

################################################################################

### Ridge Regression with quadratic terms ###

set.seed(123)
# Create model matrix to expand factor levels into dummy variables
x_var <- model.matrix(sales_price ~ carat + I(carat^2) + cut + color + clarity + length_mm + I(length_mm^2)+ 
                        width_mm + I(width_mm^2) + depth_mm + I(depth^2) + depth + table + coo + online + promotion,data = diamonds)
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
RMSE_value_ridge <- sqrt(mean((y_var - predicted_Y)^2))
# Print RMSE
print(RMSE_value_ridge) 

# Compute R^2
SSE <- sum((y_var - predicted_Y)^2)  # Sum of Squared Errors
SST <- sum((y_var - mean(y_var))^2)      # Total Sum of Squares
R2_value_ridge <- 1 - (SSE / SST)       # R^2 formula
print(R2_value_ridge)

# Compute residuals
residuals_ridge <- y_var - predicted_Y

ggplot(data = data.frame(predicted_Y, residuals_ridge), aes(x = predicted_Y, y = residuals_ridge)) +
  geom_point() +
  geom_smooth(method = "lm", color = "red") +  # Linear smoothing instead of loess
  labs(title = "Ridge Residual Plot", x = "Predicted Sales Price", y = "Residuals") +
  theme_minimal()

library(lmtest)

# Approximate model for heteroscedasticity testing
ridge_lm <- lm(sales_price ~ carat + I(carat^2) + cut + color + clarity + 
                 length_mm + I(length_mm^2) + width_mm + I(width_mm^2) + 
                 depth_mm + I(depth_mm^2) + table + coo + online + promotion, data = diamonds)

bptest(ridge_lm)  # p-value < 0.05 → heteroscedasticity exists

# get min and max values

# Extract coefficients from the ridge regression model
coefficients <- coef(best_ridge)
print(rownames(coefficients))  # View all coefficient names

# Identify quadratic terms
quadratic_terms <- coefficients[grep("\\^2", rownames(coefficients)), ]
print(quadratic_terms)

# Identify only linear terms explicitly (avoid quadratic terms)
linear_terms <- coefficients[rownames(coefficients) %in% c("carat", "length_mm", "width_mm", "depth_mm"), ]
print(linear_terms)

# Compute the optimal values for each quadratic equation
optimal_values <- -linear_terms / (2 * quadratic_terms)

# Print results
print("Optimal values for quadratic features:")
print(optimal_values)
