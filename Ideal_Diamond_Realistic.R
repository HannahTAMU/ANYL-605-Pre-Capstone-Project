
#Determine ideal round diamond sizes

#Given that most of the diamonds length and width are close to equal we can assume diamonds are typically round.
#We need to determine the ideal size of a round cut diamond.
#According to a diamond carat calculator app(Diamond Carat Calculator (Diamond Weight))
#https://www.omnicalculator.com/other/diamond-carat
#the formula for determining carat size from diamond dimensions for round cut diamonds is Diameter² × Depth × 0.0061 × (1 + GT). 
#GT(Girdle Thickness Factor) usually lies between 1% (0.01) and 3% (0.03) for round diamonds. 
#Ideal Diamond Cuts have an ideal depth percentage of 59-62.6%. This percentage is determined by dividing the diamond’s height by its width.
#(Ideal Diamond Depth and Table by Cut | The Diamond Pro)
#https://www.diamonds.pro/education/diamond-depth-and-table/.
#Checking the GT range with the carat size set to the maximum carat value of 2.35
#we can test for the diameter sizes between our minimum width that result in a 
#depth that is above the minimum depth and within the ideal depth percentage of 59-62.6%:
  

# Constants based on real world gem cutting considerations
ideal_depth_percent_low <- 0.59
ideal_depth_percent_high <- 0.626

max_carat = 2.35
min_depth_mm <- 4.27
min_width_mm <- 2.78
min_length_mm <- 7.96

# Generate sequences; 50 is arbitrary
diameter_range <- seq(min_width_mm, 50, by = 0.01)
#Based on real world ranges of GT
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
  
  # Loop over diameters; 2.35 is max carat size
  for (diameter in diameter_range) {
    depth <- max_carat / (diameter^2 * 0.0061 * (1 + gt))
    
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
      carat = max_carat
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


#Plot Final Results
# Load ggplot2
library(ggplot2)

# Create the scatterplot with purple color mapping based on GT
ggplot(final_results, aes(x = Diameter_mm, y = Depth_mm, color = GT)) +
  geom_point() +
  scale_color_gradient(low = "#e0bbff", high = "#6a0dad") +  # light to dark purple
  labs(
    title = "Ideal Round Cut Diamond Dimensions",
    subtitle = "Carat = 2.35, Depth% = 0.59-0.626, GT = 0.01-0.03",
    x = "Diameter (mm)",
    y = "Depth (mm)",
    color = "GT"
  ) +
  theme_minimal()





