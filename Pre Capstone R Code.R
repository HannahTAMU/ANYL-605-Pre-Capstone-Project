
install.packages("tidyverse")
library(tidyverse)
install.packages("janitor" ) 
library ("janitor")     
install.packages("dplyr") 
library (dplyr)
install.packages("car") 
library (car)
install.packages("caret")
library(caret)
install.packages("glmnet")
library(glmnet)
install.packages("ggplot2")
libarary(ggplot2)

diamonds <- read.csv("C:/Users/nlwym/OneDrive/Desktop/A&M MS ANALYTICS/605 Data Visualization/PreCapstone/SD Data.csv", header = TRUE)

#explore df
diamonds <- clean_names(diamonds)
View(diamonds)
str(diamonds)
names(diamonds)

#Handle NA values
sum(is.na(diamonds)) #288 NAs
dim(diamonds) #53940 rows
diamonds<-na.omit(diamonds)
dim(diamonds)


# Check their unique values
for (col in char_cols) {
  cat("\nLevels of", col, ":\n")
  print(levels(diamonds[[col]]))
}

#EXPLORING PROMOTION
print(levels(diamonds$promotion))
unique(diamonds$promotion)
levels(diamonds$promotion)
str(diamonds$promotion)
sum(diamonds$promotion == "") #3320 entries with ""


#PROMOTION has some entries with "NO " instead of just "NO"
diamonds$promotion <- trimws(diamonds$promotion)

#drop rows with diamonds$promotion == ""
diamonds <- diamonds %>%
filter(promotion != "")

width_outliers <- diamonds %>%
    filter(width_mm > length_mm * 1.50)
View(width_outliers)

#removed 2 extreme outliers
diamonds <- diamonds %>%
    filter(width_mm <= length_mm *1.5)

#Visualize width_mm v length_mm with sales_price as color
library(ggplot2)

###Discovered entries where both width and length are coded as 0.00. These have prices 
#and table and depth measurements however
diamonds_missing_dimensions <- diamonds %>%
  filter(width_mm == 0 | length_mm == 0 | depth_mm == 0)
View(diamonds_missing_dimensions)
nrow(diamonds_missing_dimensions) # 18 rows with 0s in these dimension categories. Drop these

#remove entries that have 0 for the dimensions
diamonds <- diamonds %>%
  filter(width_mm != 0 & length_mm != 0 & depth_mm != 0)

summary(diamonds)
#We no longer have any variables with values of 0 so we can proceed

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

#boxplots Carat by Country of Origin
ggplot(diamonds, aes(x = coo, y = carat)) +
  geom_boxplot(fill = "darkgreen") +
  labs(title = "Boxplot of Price by County of Origin") +
  theme_minimal()


#Convert Char Types to Factors
diamonds <- diamonds %>% 
  mutate( 
    cut = factor(cut, levels= c("Fair","Good","Very Good","Premium","Ideal")),
    color = factor(color, levels= c("J","I","H","G","F","E","D")),
    clarity = factor(clarity, levels=c("I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF")),
    promotion = as.factor(promotion),
    online = as.factor(online),
    coo = as.factor(coo)
                 )

#Linear Regression
linear_model <- lm(sales_price~carat+cut+color+clarity+depth+table+length_mm+width_mm+depth_mm+coo+online+promotion, data=diamonds)
summary(linear_model)
AIC(linear_model) 
vif(linear_model)
r2_lm <- summary(linear_model)$r.squared
rmse_lm <- RMSE(pred=predict(linear_model), obs=diamonds$sales_price)
print(rmse_lm)

#Ridge to handle colinearity
# Create model matrix to expand factor levels into dummy variables
x_var <- model.matrix(sales_price ~ carat + cut + color + clarity + depth + table +
                        length_mm + width_mm + depth_mm + coo + online + promotion,
                      data = diamonds)[, -1]  # Remove intercept column
y_var <- diamonds[,"sales_price"]

# Define lambda range
lambda_seq <- 10^seq(2, -2, by = -0.1)

# Run ridge regression (alpha = 0 means ridge)
ridge_reg <- glmnet(x_var, y_var, alpha = 0, lambda = lambda_seq)

# Cross Validation to find best lambda
cv_ridge <- cv.glmnet(x_var, y_var, alpha = 0)
best_lambda <- cv_ridge$lambda.min
coef(ridge_reg, s = best_lambda)

# Predicted values at best lambda
pred_ridge <- predict(ridge_reg, s = best_lambda, newx = x_var)

# Using cross validation glmnet
ridge_cv <- cv.glmnet(x_var, y_var, alpha = 0, lambda = lambda_seq)
# Best lambda value
best_lambda <- ridge_cv$lambda.min
best_lambda

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
SSE <- sum((Y - predicted_Y)^2)  # Sum of Squared Errors
SST <- sum((Y - mean(Y))^2)      # Total Sum of Squares
R2_value_ridge <- 1 - (SSE / SST)       # R^2 formula
print(R2_value_ridge)

#Lasso to check collinearity and pick significant variables
#Define the model equation and remove the 7th column Sales Price from matrix 
X <- model.matrix(sales_price ~ carat + cut + color + clarity + depth + table +
                    length_mm + width_mm + depth_mm + coo + online + promotion,
                  data = diamonds)
#Convert Outcome to Matrix form
Y <- as.matrix(diamonds$sales_price)
#Penalty type (alpha=1 is lasso and alpha=0 is the ridge)
cv.lambda.lasso <- cv.glmnet(x=X, y=Y,
                             alpha = 1)
#Obtain the estimates
l.lasso.min <- cv.lambda.lasso$lambda.min
lasso.model <- glmnet(x=X, y=Y,
                      alpha = 1,
                      lambda = l.lasso.min)
summary(lasso.model)
lasso.model$beta #Coefficients

#Plot
plot(cv.lambda.lasso)
log_lambda_min <- log(l.lasso.min) 
log_lambda_min

cv.lambda.lasso 
#Lasso path
plot(cv.lambda.lasso$glmnet.fit,
     "lambda", label=FALSE)
#Obtain the lasso estimates
l.lasso.min <- cv.lambda.lasso$lambda.min
lasso.model <- glmnet(x=X, y=Y,
                      alpha = 1,
                      lambda = l.lasso.min)
summary(lasso.model)
lasso.model$beta #Coefficients
names(lasso.model)


#Calculate RSME
# Make predictions using the fitted Lasso model
predicted_Y <- predict(lasso.model, newx = X, s = l.lasso.min)
# Compute RMSE
RMSE_value_lasso <- sqrt(mean((Y - predicted_Y)^2))
# Print RMSE
print(RMSE_value_lasso)
# Compute R^2
SSE <- sum((Y - predicted_Y)^2)  # Sum of Squared Errors
SST <- sum((Y - mean(Y))^2)      # Total Sum of Squares
R2_value_lasso <- 1 - (SSE / SST)       # R^2 formula
print(R2_value_lasso)

#Compare Models 
models_compared <- # Create the dataframe
  results_df <- data.frame(
    Metric = c("R^2", "RMSE"),
    `Linear Regression` = c(r2_lm, rmse_lm),
    `Ridge Regression` = c(R2_value_ridge, RMSE_value_ridge),
    `Lasso Regression` = c(R2_value_lasso, RMSE_value_lasso),
    check.names = FALSE  # Prevents conversion of column names with spaces
  )
print(models_compared)


#Investigate the relative strength of Italy, Netherlands, India
#Distribution of carat size by country
install.packages("viridis")
library(viridis)

ggplot(diamonds %>% filter(coo %in% c("ITALY", "NETHERLAND", "INDIA")), aes(x = carat, fill = coo)) +
  geom_density(alpha = 0.8) +
  labs(title = "Carat Distribution by Country", x = "Carat", fill = "Country") +
  scale_fill_viridis_d(option = "D") +
  theme_minimal() +
  theme(
    axis.text = element_text(size = 20),
    axis.title = element_text(size = 20),
    plot.title = element_text(size = 22),
    legend.text = element_text(size = 14),
    legend.title = element_text(size = 14)
  )



#Frequency of top 3 colors by country
ggplot(diamonds %>% 
         filter(coo %in% c("ITALY", "NETHERLAND", "INDIA"), 
                color %in% c("D", "E", "F")), 
       aes(x = color, fill = coo)) +
  geom_bar(position = "dodge") +
  labs(title = "Color Distribution (D, E, F) by Country", x = "Color", y = "Count", fill = "Country") +
  scale_fill_viridis_d(option = "D") +
  theme_minimal() +
  theme(
    axis.text = element_text(size = 20),
    axis.title = element_text(size = 20),
    plot.title = element_text(size = 22),
    legend.text = element_text(size = 14),
    legend.title = element_text(size = 14)
  )


#Frequency of top 3 clairities by country
ggplot(diamonds %>% 
         filter(coo %in% c("ITALY", "NETHERLAND", "INDIA"), 
                clarity %in% c("VVS2", "VVS1", "IF")), 
       aes(x = clarity, fill = coo)) +
  geom_bar(position = "dodge") +
  labs(title = "Clarity Distribution (VVS2, VVS1, IF) by Country", x = "Clarity", y = "Count", fill = "Country") +
  scale_fill_viridis_d(option = "D") +
  theme_minimal() +
  theme(
    axis.text = element_text(size = 20),
    axis.title = element_text(size = 20),
    plot.title = element_text(size = 22),
    legend.text = element_text(size = 14),
    legend.title = element_text(size = 14)
  )



# Optimality for Significant Factors
diamonds$carat_sq <- diamonds$carat^2
diamonds$depth_mm_sq <- diamonds$depth_mm^2
diamonds$length_mm_sq <- diamonds$length_mm^2
diamonds$width_mm_sq <- diamonds$width_mm^2

#Required variables
optimality_model <- lm(sales_price ~ carat + carat_sq + depth_mm + depth_mm_sq + 
                         width_mm + width_mm_sq + length_mm + length_mm_sq, data=diamonds)
summary(optimality_model)


# Optimality calculations
max_carat <- -5888.91 / (2 * -2214.74)
min_depth_mm <- 3547.43 / (2 * 594.64)
min_width_mm <- 20754.2 / (2 * 2034.69)
max_length_mm <- -10700.58 / (2 * -1015.59)

# Create a one-row data frame with named columns
results_table <- data.frame(
  max_carat = max_carat,
  min_depth_mm = min_depth_mm,
  min_width_mm = min_width_mm,
  max_length_mm = max_length_mm
)
print(results_table)


#Determine ideal round diamond sizes
# Constants
ideal_depth_percent_low <- 0.59
ideal_depth_percent_high <- 0.626

min_depth_mm <- 3547.43 / (2 * 594.64)
min_width_mm <- 20754.2 / (2 * 2034.69)
max_length_mm <- -10700.58 / (2 * -1015.59)

# Generate sequences
diameter_range <- seq(min_width_mm, max_length_mm, by = 0.01)
gt_range <- seq(0.01, 0.03, by = 0.005)

# Initialize an empty list to store results
results_list <- list()

# Loop over gt values
for (gt in gt_range) {
  # Initialize temporary storage vectors
  valid_diameters <- c()
  valid_depths <- c()
  valid_percent_depths <- c()
  valid_gt <- c()
  
  # Loop over diameters
  for (diameter in diameter_range) {
    depth <- 1.329461 / (diameter^2 * 0.0061 * (1 + gt))
    
    if (depth >= min_depth_mm) {
      percent_depth <- diameter/depth
      
      if (percent_depth >= ideal_depth_percent_low && percent_depth <= ideal_depth_percent_high) {
        valid_diameters <- c(valid_diameters, diameter)
        valid_depths <- c(valid_depths, depth)
        valid_percent_depths <- c(valid_percent_depths, percent_depth)
        valid_gt <- c(valid_gt, gt)
      }
    }
  }
  
  # Combine results for this gt
  if (length(valid_diameters) > 0) {
    temp_df <- data.frame(
      Diameter_mm = valid_diameters,
      Depth_mm = valid_depths,
      Depth_Percentage = valid_percent_depths,
      GT = valid_gt,
      carat = 1.329461
    )
    results_list[[as.character(gt)]] <- temp_df
  }
}

# Combine all gt result dataframes into one
final_results <- do.call(rbind, results_list)

#clean up rownames
rownames(final_results) <- NULL

# View results
print(final_results)




#Graph Optimality Curves
#NOTE:
#I can’t get my carat optimality graph to be a maximum or positive and I can’t get 
#Length to show as a maximum instead of a minimum

# Carat Model
carat_fit <- lm(sales_price ~ carat + carat_sq, data=diamonds)
#summary(carat_fit)
carat_new <- seq(-10, 50, 0.05)
predictedsales <- predict(carat_fit, list(carat=carat_new, carat_sq=carat_new^2))
plot(carat_new, predictedsales, pch=16, xlab="Carat Size", ylab="Sales Price", cex.lab=1.3, col="green")
lines(carat_new, predictedsales, col="darkgreen", lwd=3)

# Depth Model
depth_fit <- lm(sales_price ~ depth_mm + depth_mm_sq, data=diamonds)
summary(depth_fit)
depth_mm_new <- seq(0, 6, 0.05)
predicted_sales_depth <- predict(depth_fit, list(depth_mm=depth_mm_new, depth_mm_sq=depth_mm_new^2))
plot(depth_mm_new, predicted_sales_depth, pch=16, xlab="Depth (mm)", ylab="Sales Price", cex.lab=1.3, col="blue")
lines(depth_mm_new, predicted_sales_depth, col="darkblue", lwd=3)

# Width Model
width_fit <- lm(sales_price ~ width_mm + width_mm_sq, data=diamonds)
#summary(width_fit)
width_mm_new <- seq(0, 10, 0.05)
predicted_sales_width <- predict(width_fit, list(width_mm=width_mm_new, width_mm_sq=width_mm_new^2))
plot(width_mm_new, predicted_sales_width, pch=16, xlab="Width (mm)", ylab="Sales Price", cex.lab=1.3, col="purple")
lines(width_mm_new, predicted_sales_width, col="darkpurple", lwd=3)


# Length Model
length_fit <- lm(sales_price ~ length_mm + length_mm_sq, data=diamonds)
#summary(length_fit)
length_mm_new <- seq(0, 12, 0.05)
predicted_sales_length <- predict(length_fit, list(length_mm=length_mm_new, length_mm_sq=length_mm_new^2))
plot(length_mm_new, predicted_sales_length, pch=16, xlab="Length (mm)", ylab="Sales Price", cex.lab=1.3, col="red")
lines(length_mm_new, predicted_sales_length, col="darkred", lwd=3)


