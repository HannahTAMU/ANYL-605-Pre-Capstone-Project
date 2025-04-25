#v3 precapstone


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
library(ggplot2)
install.packages("gridExtra")
library(gridExtra) # For arranging plots in a grid


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
#Visualize width_mm v length_mm with sales_price as color


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



#####ANOTHER ITERATION WHERE WE SEPARATE THE CLEANED DATASET TO TEST IN OTHER FUNCTIONS
library(dplyr)
library(ggplot2)
library(car)
library(glmnet)


# Define a simple RMSE function
RMSE <- function(pred, obs){
  sqrt(mean((pred - obs)^2, na.rm = TRUE))
}

# Define the list of models
models_to_test <- list(
  linear_model = lm(sales_price ~ carat + cut + color + clarity + depth + table + length_mm + width_mm + depth_mm + coo + online + promotion, data = diamonds),
  lm_dims_removed = lm(sales_price ~ carat + cut + color + clarity + depth + table + coo + online + promotion, data = diamonds),
  log_lm_dims_removed = lm(log(sales_price) ~ carat + cut + color + clarity + depth + table + coo + online + promotion, data = diamonds)
)

# Function to process each model and return cleaned model info
process_model_for_reuse <- function(model, model_name, data) {
  cat(paste("Processing model:", model_name, "\n"))
  
  # Summarize the initial model and print metrics
  summary(model)
  r_squared <- summary(model)$r.squared
  aic_value <- AIC(model)
  vif_values <- car::vif(model)
  rmse <- RMSE(pred=predict(model), obs=model.response(model.frame(model)))
  cat(paste("Initial R-squared:", round(r_squared, 4), "\n"))
  cat(paste("Initial RSME:", round(rmse, 4), "\n"))
  cat(paste("Initial AIC:", round(aic_value, 2), "\n"))
  cat("Initial VIF values:\n")
  print(vif_values)
  cat("\n")
  
  # Calculate Cook's Distance and Leverage
  cooks_d <- cooks.distance(model)
  leverage <- hatvalues(model)
  
  # Define thresholds (using a percentile-based method)
  n <- nrow(data)
  p <- length(coefficients(model))
  cooks_thresh <- quantile(cooks_d, 0.99)
  leverage_thresh <- quantile(leverage, 0.99)
  
  # Identify influential points
  influential_points <- which(cooks_d > cooks_thresh | leverage > leverage_thresh)
  cat(paste("Number of influential points identified:", length(influential_points), "\n"))
  
  cleaned_model <- NULL
  cleaned_data <- data
  
  if (length(influential_points) > 0) {
    # Identify the rows to remove
    rows_to_remove <- which(rownames(data) %in% rownames(data[influential_points, ]))
    
    # Create cleaned dataset
    cleaned_data <- data[-rows_to_remove, ]
    cat(paste("Dataset size after removing influential points:", nrow(cleaned_data), "\n"))
    
    # Refit the model with cleaned data
    cleaned_model <- update(model, data = cleaned_data)
    cat(paste("Model refitted for:", model_name, "with cleaned data\n"))
    
    # Summarize the cleaned model and print metrics
    summary(cleaned_model)
    r_squared_clean <- summary(cleaned_model)$r.squared
    rmse_clean <- RMSE(pred=predict(cleaned_model, newdata = cleaned_data), obs=model.response(model.frame(cleaned_model, data = cleaned_data)))
    aic_value_clean <- AIC(cleaned_model)
    vif_values_clean <- car::vif(cleaned_model)
    cat(paste("Cleaned R-squared:", round(r_squared_clean, 4), "\n"))
    cat(paste("Cleaned RSME:", round(rmse_clean, 4), "\n"))
    cat(paste("Cleaned AIC:", round(aic_value_clean, 2), "\n"))
    cat("Cleaned VIF values:\n")
    print(vif_values_clean)
    cat("\n")
    
  } else {
    cat("No influential points found based on the thresholds. Returning original model.\n")
    cleaned_model <- model
  }
  
  return(list(original_model = model, cleaned_model = cleaned_model, cleaned_data = cleaned_data, influential_points = influential_points))
}

# Function to plot Cook's Distance and Residuals using the output of process_model_for_reuse
plot_cleaned_model_results <- function(cleaned_model_output, model_name, original_data) {
  original_model <- cleaned_model_output$original_model
  cleaned_model <- cleaned_model_output$cleaned_model
  cleaned_data <- cleaned_model_output$cleaned_data
  influential_points <- cleaned_model_output$influential_points
  n_original <- nrow(original_data)
  
  # Plot Cook's Distance (using the original model)
  cooks_d <- cooks.distance(original_model)
  cooks_thresh <- quantile(cooks_d, 0.99)
  influence_data <- data.frame(
    Observation = 1:n_original,
    CooksDistance = cooks_d,
    Influential = 1:n_original %in% influential_points
  )
  
  cooks_plot <- ggplot(influence_data, aes(x = Observation, y = CooksDistance, color = Influential)) +
    geom_point(size = 2, alpha = 0.7) +
    geom_hline(yintercept = cooks_thresh, linetype = "dashed", color = "red") +
    scale_color_manual(values = c("FALSE" = "steelblue", "TRUE" = "red")) +
    labs(
      title = paste("Cook's Distance for", model_name),
      subtitle = paste("Threshold:", round(cooks_thresh, 5)),
      x = "Observation Index",
      y = "Cook's Distance",
      color = "Influential?"
    ) +
    theme_minimal()
  print(cooks_plot)
  cooks_plot
  
  # Plot Predicted vs Standardized Residuals for the cleaned model
  predictions_clean <- predict(cleaned_model, newdata = cleaned_data)
  standardized_residuals_clean <- rstandard(cleaned_model)
  
  pred_vs_residuals_df_clean <- data.frame(
    Predicted = predictions_clean,
    StdResiduals = standardized_residuals_clean
  )
  
  residuals_plot_clean <- ggplot(pred_vs_residuals_df_clean, aes(x = Predicted, y = StdResiduals)) +
    geom_point(color = "steelblue") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "darkred") +
    labs(
      title = paste("Residuals (Cleaned) for", model_name), # Shorter title
      x = "Predicted Values",
      y = "Standardized Residuals"
    ) +
    theme_minimal()
  print(residuals_plot_clean)
  residuals_plot_clean
}


# Function to perform Ridge Regression
perform_ridge <- function(model, data, model_name, is_cleaned = FALSE) {
  cat(paste("Performing Ridge Regression for:", model_name, ifelse(is_cleaned, " (Cleaned Data)", " (Original Data)"), "\n"))
  
  # Create model matrix
  x_var <- model.matrix(formula(model), data = data)[, -1, drop = FALSE] # Remove intercept column
  y_var <- model.response(model.frame(formula(model), data = data))
  
  # Define lambda range for cross-validation
  lambda_seq <- 10^seq(2, -2, by = -0.1)
  
  # Cross Validation to find best lambda
  cv_ridge <- cv.glmnet(x_var, y_var, alpha = 0, lambda = lambda_seq)
  best_lambda <- cv_ridge$lambda.min
  cat(paste("Best Lambda (via CV):", best_lambda, "\n"))
  print(coef(cv_ridge, s = best_lambda))
  cat("\n")
  
  # Rebuilding the model with optimal lambda value
  best_ridge <- glmnet(x_var, y_var, alpha = 0, lambda = best_lambda)
  
  # Calculate RMSE
  predicted_Y <- predict(best_ridge, newx = x_var, s = best_lambda)
  rmse_value <- sqrt(mean((y_var - predicted_Y)^2))
  cat(paste("RMSE:", rmse_value, "\n"))
  
  # Calculate R-squared
  sse <- sum((y_var - predicted_Y)^2)
  sst <- sum((y_var - mean(y_var))^2)
  r2_value <- 1 - (sse / sst)
  cat(paste("R-squared:", r2_value, "\n"))
  cat("\n")
  
  # Visual Ridge residuals
  # Standardized residuals
  residuals_ridge <- y_var - predicted_Y
  standardized_residuals_ridge <- scale(residuals_ridge)
  
  # DataFrame: Predicted vs Standardized Residuals
  ridge_pred_vs_resid <- data.frame(
    Predicted = as.vector(predicted_Y),
    StdResiduals = as.vector(standardized_residuals_ridge)
  )
  
  # Plot Predicted vs Standardized Residuals
  resid_plot <- ggplot(ridge_pred_vs_resid, aes(x = Predicted, y = StdResiduals)) +
    geom_point(color = "darkgreen") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
    labs(title = paste("Ridge:", model_name, ifelse(is_cleaned, "(Cleaned)", "(Original)")), # Shorter title
         x = "Predicted Values",
         y = "Standardized Residuals") +
    theme_minimal()
  print(resid_plot)
  resid_plot
}
# Function to perform Lasso Regression
perform_lasso <- function(model, data, model_name, is_cleaned = FALSE) {
  cat(paste("Performing Lasso Regression for:", model_name, ifelse(is_cleaned, " (Cleaned Data)", " (Original Data)"), "\n"))
  
  # Define the formula
  model_formula <- formula(model)
  
  # Create model matrix and outcome variable
  x_var <- model.matrix(model_formula, data = data)[, -1, drop = FALSE] # Remove intercept
  y_var <- model.response(model.frame(model_formula, data = data))
  
  # Penalty type (alpha=1 is lasso)
  cv.lambda.lasso <- cv.glmnet(x = x_var, y = y_var, alpha = 1)
  
  # Obtain the best lambda (lambda.min)
  best_lambda_lasso <- cv.lambda.lasso$lambda.min
  cat(paste("Best Lambda (via CV):", best_lambda_lasso, "\n"))
  plot(cv.lambda.lasso, main = paste("Lasso CV Lambda for", model_name, ifelse(is_cleaned, " (Cleaned)", " (Original)")))
  
  # Fit the Lasso model with the best lambda
  lasso_model <- glmnet(x = x_var, y = y_var, alpha = 1, lambda = best_lambda_lasso)
  
  cat("\nLasso Coefficients:\n")
  print(coef(lasso_model))
  cat("\n")
  
  # Make predictions
  predicted_y_lasso <- predict(lasso_model, newx = x_var, s = best_lambda_lasso)
  
  # Calculate RMSE
  rmse_value_lasso <- sqrt(mean((y_var - predicted_y_lasso)^2))
  cat(paste("RMSE (Lasso):", rmse_value_lasso, "\n"))
  
  # Calculate R-squared
  sse_lasso <- sum((y_var - predicted_y_lasso)^2)
  sst_lasso <- sum((y_var - mean(y_var))^2)
  r2_value_lasso <- 1 - (sse_lasso / sst_lasso)
  cat(paste("R-squared (Lasso):", r2_value_lasso, "\n"))
  cat("\n---------------------------------------------------\n")
  
  # Residual Plots
  residuals_lasso <- y_var - predicted_y_lasso
  standardized_residuals_lasso <- scale(residuals_lasso)
  
  lasso_pred_vs_actual <- data.frame(
    Actual = y_var,
    Predicted = as.vector(predicted_y_lasso)
  )
  
  lasso_pred_vs_resid <- data.frame(
    Predicted = as.vector(predicted_y_lasso),
    StdResiduals = as.vector(standardized_residuals_lasso)
  )
  
  # Plot: Predicted vs Actual
  p1 <- ggplot(lasso_pred_vs_actual, aes(x = Actual, y = Predicted)) +
    geom_point(color = "darkred") +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "blue") +
    labs(title = paste("Lasso:", model_name, ifelse(is_cleaned, "(Cleaned)", "(Original)")), # Even shorter title
         x = "Actual Values",
         y = "Predicted Values") +
    theme_minimal()
  print(p1)
  
  # Plot: Predicted vs Standardized Residuals
  p2 <- ggplot(lasso_pred_vs_resid, aes(x = Predicted, y = StdResiduals)) +
    geom_point(color = "darkred") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "blue") +
    labs(title = paste("Lasso:", model_name, ifelse(is_cleaned, "(Cleaned)", "(Original)")),
         x = "Predicted Values",
         y = "Standardized Residuals") +
    theme_minimal()
  print(p2)
  
  return(list(actual_vs_predicted = p1, residuals = p2)) # Return a list of plots
}


# Loop through the list of models and process each one to get cleaned output
cleaned_models_output <- list()
for (name in names(models_to_test)) {
  cleaned_models_output[[name]] <- process_model_for_reuse(models_to_test[[name]], name, diamonds)
}

# Initialize the plot lists in the global environment
cooks_plots <- list()
residuals_plots <- list()

# Now, loop through the cleaned model output and perform the plotting
for (name in names(cleaned_models_output)) {
  cat(paste("\n--- Plotting Cook's and Residuals for model:", name, "---\n"))
  results <- plot_cleaned_model_results(cleaned_models_output[[name]], name, diamonds)
  print(str(results)) # Inspect the structure of what's being returned
  cooks_plots[[paste0(name, "-cooks")]] <- results # Let's see if this captures anything
  residuals_plots[[paste0(name, "-residuals")]] <- results # And this
}


# Loop through the cleaned model output and perform Ridge Regression on both original and cleaned (if available)
ridge_results <- list()
for (name in names(cleaned_models_output)) {
  output <- cleaned_models_output[[name]]
  ridge_results[[paste0("original_", name)]] <- perform_ridge(output$original_model, diamonds, name, is_cleaned = FALSE)
  if (!is.null(output$cleaned_model)) {
    ridge_results[[paste0("cleaned_", name)]] <- perform_ridge(output$cleaned_model, output$cleaned_data, name, is_cleaned = TRUE)
  }
}

####LASSO
lasso_results <- list()
for (name in names(cleaned_models_output)) {
  output <- cleaned_models_output[[name]]
  lasso_results[[paste0("original_", name)]] <- perform_lasso(output$original_model, diamonds, name, is_cleaned = FALSE)
  if (!is.null(output$cleaned_model)) {
    lasso_results[[paste0("cleaned_", name)]] <- perform_lasso(output$cleaned_model, output$cleaned_data, name, is_cleaned = TRUE)
  }
}



assemble_plots_grid(cooks_plots, residuals_plots, ridge_results, lasso_results, title_size = 8)
# Function to assemble plots in a grid organized by model with improved title handling
assemble_plots_grid <- function(cooks_plots, residuals_plots, ridge_results, lasso_results, title_size = 9, title_lineheight = 0.8, title_margin = margin(b = 5, r = 5, l = 5)) { # Added left and right margin
  cat("\n--- Starting assemble_plots_grid ---\n")
  all_model_names <- unique(gsub("-(cooks|residuals|ridge-original|ridge-cleaned|lasso-original|lasso-cleaned)$", "", names(cooks_plots)))
  cat(paste("Unique model names found:", paste(all_model_names, collapse = ", "), "\n"))
  grid_plots <- list()
  
  for (model_name in all_model_names) {
    cat(paste("\n--- Processing model:", model_name, "---\n"))
    plots_for_model <- list()
    
    # Define a common theme for titles with wrapping and smaller size
    wrapped_title_theme <- theme(plot.title = element_text(size = title_size, lineheight = title_lineheight, margin = title_margin))
    
    # Add Cook's Distance plot
    cooks_name <- paste0(model_name, "-cooks")
    if (cooks_name %in% names(cooks_plots) && !is.null(cooks_plots[[cooks_name]]) && inherits(cooks_plots[[cooks_name]], "ggplot")) {
      plots_for_model[[cooks_name]] <- cooks_plots[[cooks_name]] + wrapped_title_theme
      cat(paste("Added Cook's plot:", cooks_name, "\n"))
    } else {
      cat(paste("Cook's plot not found or invalid:", cooks_name, "\n"))
    }
    
    # Add Residuals plot
    residuals_name <- paste0(model_name, "-residuals")
    if (residuals_name %in% names(residuals_plots) && !is.null(residuals_plots[[residuals_name]]) && inherits(residuals_plots[[residuals_name]], "ggplot")) {
      plots_for_model[[residuals_name]] <- residuals_plots[[residuals_name]] + wrapped_title_theme
      cat(paste("Added Residuals plot:", residuals_name, "\n"))
    } else {
      cat(paste("Residuals plot not found or invalid:", residuals_name, "\n"))
    }
    
    # Add Ridge Regression Residuals plot (Original)
    ridge_original_name <- paste0("original_", model_name)
    if (ridge_original_name %in% names(ridge_results) && !is.null(ridge_results[[ridge_original_name]]) && inherits(ridge_results[[ridge_original_name]], "ggplot")) {
      plots_for_model[[paste0(model_name, "-ridge-original")]] <- ridge_results[[ridge_original_name]] + ggtitle(paste(model_name, "Original - Ridge Residuals")) + wrapped_title_theme
      cat(paste("Added Ridge (Original) plot:", paste0(model_name, "-ridge-original"), "\n"))
    } else {
      cat(paste("Ridge (Original) plot not found or invalid:", ridge_original_name, "\n"))
    }
    
    # Add Ridge Regression Residuals plot (Cleaned)
    ridge_cleaned_name <- paste0("cleaned_", model_name)
    if (ridge_cleaned_name %in% names(ridge_results) && !is.null(ridge_results[[ridge_cleaned_name]]) && inherits(ridge_results[[ridge_cleaned_name]], "ggplot")) {
      plots_for_model[[paste0(model_name, "-ridge-cleaned")]] <- ridge_results[[ridge_cleaned_name]] + ggtitle(paste(model_name, "Cleaned - Ridge Residuals")) + wrapped_title_theme
      cat(paste("Added Ridge (Cleaned) plot:", paste0(model_name, "-ridge-cleaned"), "\n"))
    } else {
      cat(paste("Ridge (Cleaned) plot not found or invalid:", ridge_cleaned_name, "\n"))
    }
    
    # Add Lasso Regression Plots (Original)
    lasso_original_name <- paste0("original_", model_name)
    if (lasso_original_name %in% names(lasso_results) && !is.null(lasso_results[[lasso_original_name]])) {
      if (!is.null(lasso_results[[lasso_original_name]]$actual_vs_predicted) && inherits(lasso_results[[lasso_original_name]]$actual_vs_predicted, "ggplot")) {
        plots_for_model[[paste0(model_name, "-lasso-original-actual")]] <- lasso_results[[lasso_original_name]]$actual_vs_predicted + ggtitle(paste(model_name, "Original - Lasso Pred vs Actual")) + wrapped_title_theme
        cat(paste("Added Lasso (Original) Actual vs Predicted plot:", paste0(model_name, "-lasso-original-actual"), "\n"))
      } else {
        cat(paste("Lasso (Original) Actual vs Predicted plot not found or invalid:", lasso_original_name, "$actual_vs_predicted\n"))
      }
      if (!is.null(lasso_results[[lasso_original_name]]$residuals) && inherits(lasso_results[[lasso_original_name]]$residuals, "ggplot")) {
        plots_for_model[[paste0(model_name, "-lasso-original-residuals")]] <- lasso_results[[lasso_original_name]]$residuals + ggtitle(paste(model_name, "Original - Lasso Residuals")) + wrapped_title_theme
        cat(paste("Added Lasso (Original) Residuals plot:", paste0(model_name, "-lasso-original-residuals"), "\n"))
      } else {
        cat(paste("Lasso (Original) Residuals plot not found or invalid:", lasso_original_name, "$residuals\n"))
      }
    } else {
      cat(paste("Lasso (Original) results not found or invalid:", lasso_original_name, "\n"))
    }
    
    # Add Lasso Regression Plots (Cleaned)
    lasso_cleaned_name <- paste0("cleaned_", model_name)
    if (lasso_cleaned_name %in% names(lasso_results) && !is.null(lasso_results[[lasso_cleaned_name]])) {
      if (!is.null(lasso_results[[lasso_cleaned_name]]$actual_vs_predicted) && inherits(lasso_results[[lasso_cleaned_name]]$actual_vs_predicted, "ggplot")) {
        plots_for_model[[paste0(model_name, "-lasso-cleaned-actual")]] <- lasso_results[[lasso_cleaned_name]]$actual_vs_predicted + ggtitle(paste(model_name, "Cleaned - Lasso Pred vs Actual")) + wrapped_title_theme
        cat(paste("Added Lasso (Cleaned) Actual vs Predicted plot:", paste0(model_name, "-lasso-cleaned-actual"), "\n"))
      } else {
        cat(paste("Lasso (Cleaned) Actual vs Predicted plot not found or invalid:", lasso_cleaned_name, "$actual_vs_predicted\n"))
      }
      if (!is.null(lasso_results[[lasso_cleaned_name]]$residuals) && inherits(lasso_results[[lasso_cleaned_name]]$residuals, "ggplot")) {
        plots_for_model[[paste0(model_name, "-lasso-cleaned-residuals")]] <- lasso_results[[lasso_cleaned_name]]$residuals + ggtitle(paste(model_name, "Cleaned - Lasso Residuals")) + wrapped_title_theme
        cat(paste("Added Lasso (Cleaned) Residuals plot:", paste0(model_name, "-lasso-cleaned-residuals"), "\n"))
      } else {
        cat(paste("Lasso (Cleaned) Residuals plot not found or invalid:", lasso_cleaned_name, "$residuals\n"))
      }
    } else {
      cat(paste("Lasso (Cleaned) results not found or invalid:", lasso_cleaned_name, "\n"))
    }
    
    if (length(plots_for_model) > 0) {
      grid_plots[[model_name]] <- plots_for_model
      cat(paste("Number of plots for", model_name, ":", length(plots_for_model), "\n"))
    } else {
      cat(paste("No valid plots found for", model_name, "\n"))
    }
  }
  
  cat("\n--- Arranging plots in the grid ---\n")
  final_grid <- list()
  for (model in names(grid_plots)) {
    n_plots <- length(grid_plots[[model]])
    cat(paste("Arranging", n_plots, "plots for model:", model, "\n"))
    if (n_plots > 0) {
      final_grid[[model]] <- do.call(grid.arrange, c(grid_plots[[model]], ncol = n_plots, padding = unit(5, "mm"))) # Added padding
      cat(paste("Arranged plots for model:", model, "\n"))
    } else {
      cat(paste("No plots to arrange for model:", model, "\n"))
    }
  }
  
  n_model_rows <- length(final_grid)
  cat(paste("Number of model rows in final_grid:", n_model_rows, "\n"))
  if (n_model_rows > 0) {
    do.call(grid.arrange, c(final_grid, nrow = n_model_rows)) # Arrange model rows
    cat("Final grid arranged and (hopefully) printed with better title handling.\n")
  } else {
    cat("No model rows to arrange in the final grid.\n")
  }
  cat("--- Finished assemble_plots_grid ---\n")
}









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



