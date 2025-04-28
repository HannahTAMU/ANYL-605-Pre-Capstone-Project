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
library(dplyr)

################################################################################

### LOAD DATA ###

diamonds <- read.csv("C:/Users/Hanna/MSA/ANYL 605/PreCapstone/SD Data.csv")

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

############# QUESTION 3 #######################
### Explore Country of Origin ###
ggplot(diamonds, aes(x = coo, y = carat)) +
  geom_boxplot(fill = "darkgreen") +
  labs(title = "Boxplot of Price by Country of Origin") +
  theme_minimal()

# Load necessary libraries
library(ggplot2)
library(dplyr)

# Calculate proportions for each country
coo_proportion <- diamonds %>%
  group_by(coo) %>%
  summarise(count = n()) %>%
  mutate(proportion = count / sum(count)) %>%
  arrange(desc(proportion))

# Plot the proportions
ggplot(coo_proportion, aes(x = reorder(coo, proportion), y = proportion, fill = coo)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  coord_flip() +  # Flip for readability
  theme_minimal() +
  labs(title = "Proportion of Diamonds by Country of Origin",
       x = "Country",
       y = "Proportion of Total Diamonds") +
  scale_fill_brewer(palette = "Set3") +
  scale_y_continuous(labels = scales::percent)  # Format y-axis as percentage


# Load necessary libraries
library(ggplot2)
library(dplyr)

# Aggregate total diamond counts per country and clarity type
clarity_by_country <- diamonds %>%
  group_by(coo, clarity) %>%
  summarise(count = n(), .groups = "drop")

# Calculate proportion of highest clarity diamonds (IF, VVS1, VVS2) per country
high_clarity_proportion <- diamonds %>%
  filter(clarity %in% c("IF", "VVS1", "VVS2")) %>%
  group_by(coo) %>%
  summarise(high_clarity_count = n()) %>%
  mutate(high_clarity_proportion = round((high_clarity_count / sum(diamonds$clarity %in% c("IF", "VVS1", "VVS2"))) * 100))

# Merge the proportions into the dataset
clarity_data <- left_join(clarity_by_country, high_clarity_proportion, by = "coo")

# Reverse order: Sort countries by total diamond count (highest at the top)
ggplot(clarity_data, aes(x = reorder(coo, count), y = count, fill = clarity)) + 
  geom_bar(stat = "identity") + 
  geom_text(data = distinct(clarity_data, coo, .keep_all = TRUE), 
            aes(x = coo, y = sum(count), label = paste0(high_clarity_proportion, "%")), 
            vjust = -0.5, size = 5, color = "black", fontface = "bold") +  # Bold label without decimals
  coord_flip() +  # Flip for readability
  theme_minimal() +
  labs(title = "Diamond Clarity Distribution by Country (Sorted by Total Diamonds)",
       x = "Country",
       y = "Number of Diamonds",
       fill = "Clarity") +
  scale_fill_brewer(palette = "Set3")  # Color-coded clarity levels


############# BY COLOR 
# Load necessary libraries
library(ggplot2)
library(dplyr)

# Aggregate total diamond counts per country and color type
color_by_country <- diamonds %>%
  group_by(coo, color) %>%
  summarise(count = n(), .groups = "drop")

# Calculate proportion of highest color diamonds (D, E, F) per country
high_color_proportion <- diamonds %>%
  filter(color %in% c("D", "E", "F")) %>%
  group_by(coo) %>%
  summarise(high_color_count = n()) %>%
  mutate(high_color_proportion = round((high_color_count / sum(diamonds$color %in% c("D", "E", "F"))) * 100))

# Merge the proportions into the dataset
color_data <- left_join(color_by_country, high_color_proportion, by = "coo")

# Reverse order: Sort countries by total diamond count (highest at the top)
ggplot(color_data, aes(x = reorder(coo, count), y = count, fill = color)) + 
  geom_bar(stat = "identity") + 
  geom_text(data = distinct(color_data, coo, .keep_all = TRUE), 
            aes(x = coo, y = sum(count), label = paste0(high_color_proportion, "%")), 
            vjust = -0.5, size = 5, color = "black", fontface = "bold") +  # Bold proportion label
  coord_flip() +  # Flip for readability
  theme_minimal() +
  labs(title = "Diamond Color Distribution by Country (Sorted by Total Diamonds)",
       x = "Country",
       y = "Number of Diamonds",
       fill = "Color") +
  scale_fill_brewer(palette = "Set3")  # Color-coded diamond colors



##########MAP
# Load necessary libraries
install.packages(c("sf", "rnaturalearth", "rnaturalearthdata", "RColorBrewer"))
# Load necessary libraries
library(ggplot2)
library(dplyr)
library(sf)
library(rnaturalearth)
library(rnaturalearthdata)
library(RColorBrewer)

# Create a data frame with country names and coefficients
coeff_data <- data.frame(
  country = c("United Kingdom", "Cuba", "India", "Italy", "Jamaica", 
              "Netherlands", "Sweden", "United States", "Brazil"), 
  coefficient = c(0.0859567, -29.1052501, 11.2248246, 19.3870277, 
                  4.6079159, 16.5946924, -1.3543338, 0.7830533, 0) # Brazil = Baseline (0)
)
# Fix country name issue
coeff_data$country[coeff_data$country == "United States"] <- "United States of America"
# Load world map data
world <- ne_countries(scale = "medium", returnclass = "sf")

# Merge coefficients with world map data
world_data <- left_join(world, coeff_data, by = c("name" = "country"))

# Define color palette: Bright turquoise for positive, bright reddish-purple for negative, grey for Brazil
ggplot(world_data) +
  geom_sf(fill = "white", color = "black") +  # Outline for non-included countries
  geom_sf(data = world_data %>% filter(!is.na(coefficient)), aes(fill = coefficient)) +
  scale_fill_gradient2(low = "#D42E92", mid = "grey", high = "#00F5D4", midpoint = 0) +  # Brighter colors
  theme_minimal() +
  labs(title = "World Map of Coefficients (Baseline: Brazil)",
       fill = "Coefficient Value")


# Define custom colors to match the map
coeff_data$color_group <- case_when(
  coeff_data$coefficient > 0 ~ "High Coefficient",
  coeff_data$coefficient < 0 ~ "Low Coefficient",
  TRUE ~ "Baseline (Brazil)"
)

# Create bar chart with same colors as the map
ggplot(coeff_data, aes(x = reorder(country, coefficient), y = coefficient, fill = coefficient)) +
  geom_bar(stat = "identity") +
  scale_fill_gradient2(low = "#D42E92", mid = "grey", high = "#00F5D4", midpoint = 0) +  # Matching map colors
  theme_minimal() +
  labs(title = "Coefficients by Country",
       x = "Country",
       y = "Coefficient Value",
       fill = "Coefficient")




##################################################


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
