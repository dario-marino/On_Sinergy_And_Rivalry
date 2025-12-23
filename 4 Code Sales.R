
#############################
#### GAM Sales for BIG and SMALL
#############################

library(ggplot2)
library(dplyr)

pair_product <- read.csv("/mnt/ide0/home/dariomarino/pair_product.csv")

pair_product <- pair_product %>%
  filter(sale_start >= 0)

# Load necessary packages
library(mgcv)
library(dplyr)
library(ggplot2)
library(tidyr)

# --- STEP 1: Prepare data ----

pair_product <- pair_product %>%
  filter(sale_start > 0) %>%
  mutate(
    # Remove log transformation - use actual sales values
    fyear_start = factor(fyear_start),
    sic_start   = factor(sic_start)
  )

controls <- c(
  "at_start", "emp_start", "ebitda_start", "xrd_recv",
  "at_recv", "emp_recv", "fyear_recv", "concentration_start",
  "concentration_recv"
)

# --- STEP 2: Fit GAM models with bam() ----

fit_model <- function(formula_text) {
  bam(
    as.formula(formula_text),
    data = pair_product,
    discrete = TRUE,
    nthreads = 30,
    # Add gamma distribution for positive continuous data
    family = Gamma(link = "log")
  )
}

smooth_term     <- "s(product_similarity, technology_similarity)"
control_formula <- paste(controls, collapse = " + ")
# Change dependent variable to sale_start
base_formula    <- paste("sale_start ~", smooth_term, "+", control_formula)

formulas <- list(
  controls_only     = base_formula,
  year_fe           = paste0(base_formula, " + fyear_start"),
  sector_fe         = paste0(base_formula, " + sic_start"),
  year_sector_fe    = paste0(base_formula, " + fyear_start + sic_start")
)

model_controls_only  <- fit_model(formulas$controls_only);  gc()
model_year_fe        <- fit_model(formulas$year_fe);         gc()
model_sector_fe      <- fit_model(formulas$sector_fe);       gc()
model_year_sector_fe <- fit_model(formulas$year_sector_fe);  gc()

models <- list(
  model_controls_only,
  model_year_fe,
  model_sector_fe,
  model_year_sector_fe
)

# --- STEP 3: Create prediction grid ----

control_medians <- pair_product %>%
  summarise(across(all_of(controls), ~median(., na.rm = TRUE)))

grid_data <- expand.grid(
  product_similarity = seq(0, 1, length.out = 100),
  technology_similarity = seq(0, 1, length.out = 100)
)
grid_data <- bind_cols(grid_data, control_medians[rep(1, nrow(grid_data)), ])

grid_data$fyear_start <- factor(rep(levels(pair_product$fyear_start)[1], nrow(grid_data)),
                                levels = levels(pair_product$fyear_start))
grid_data$sic_start <- factor(rep(levels(pair_product$sic_start)[1], nrow(grid_data)),
                              levels = levels(pair_product$sic_start))

# --- NEW: Create density map to mask low-observation areas ----

# Round similarity values to 2 digits for matching
pair_product_density <- pair_product %>%
  mutate(
    ps_bin = round(product_similarity, 2),
    ts_bin = round(technology_similarity, 2)
  ) %>%
  count(ts_bin, ps_bin, name = "density")

grid_data <- grid_data %>%
  mutate(
    ps_bin = round(product_similarity, 2),
    ts_bin = round(technology_similarity, 2)
  ) %>%
  left_join(pair_product_density, by = c("ts_bin", "ps_bin"))

# Mask areas with low density (< threshold)
density_threshold <- 10  # Set as desired
grid_data <- grid_data %>%
  mutate(use_point = ifelse(is.na(density) | density < density_threshold, FALSE, TRUE))


# --- STEP 4: Plotting and saving ----
plot_heatmap <- function(model, title) {
  # No need to exponentiate since we're already working with actual sale values
  grid_data$predicted_sales <- predict(model, newdata = grid_data, type = "response")
  
  ggplot(grid_data %>% filter(use_point), aes(x = technology_similarity, y = product_similarity, fill = predicted_sales)) +
    geom_tile() +
    scale_fill_viridis_c(option = "magma") +
    labs(
      title = title,
      x = "Technology Similarity",
      y = "Product Similarity",
      fill = "Predicted Sales"
    ) +
    theme_minimal()
}

titles <- c(
  "GAM: Controls Only",
  "GAM: Controls + Year Fixed Effects",
  "GAM: Controls + Sector Fixed Effects",
  "GAM: Controls + Year + Sector Fixed Effects"
)

plots <- Map(plot_heatmap, models, titles)
for (p in plots) print(p)

filenames <- c(
  "heatmap_sales_controls_only.png",
  "heatmap_sales_year_fe.png",
  "heatmap_sales_sector_fe.png",
  "heatmap_sales_year_sector_fe.png"
)

Map(ggsave, filename = filenames, plot = plots, width = 8, height = 6)


# --- DENSITY PLOT (unchanged) ----

library(ggplot2)
library(viridis)
library(scales) # Essential for label_number()

# Ensure scientific notation is suppressed for axis labels if desired
options(scipen = 999)

ggplot(pair_product, aes(x = technology_similarity, y = product_similarity)) +
  geom_bin2d(bins = 50) +
  scale_fill_gradient(
    low = "lightblue",  # Starting color for low values (lighter blue)
    high = "darkblue", # Ending color for high values (darker blue)
    trans = "log10",
    # Manually define a set of breaks that are well-spaced on a log scale
    # but also include more intermediate values at the higher end.
    # Adjust these values based on the actual range of your 'count' data.
    breaks = c(10, 100, 1000, 10000, 100000, 1000000),
    # Use scales::label_number to format the breaks as full numbers
    # without scientific notation or "K" abbreviations.
    labels = scales::label_number(big.mark = ",", accuracy = 1), # Adds comma for thousands
    # Optional: Set limits if you want to ensure the scale spans a precise range
    # limits = c(min_actual_count, max_actual_count)
  ) +
  labs(title = "Density of Observations in Similarity Space",
       fill = "Count (Log10)") # Keep legend title consistent


# Modified GAM Analysis - Direct Log(1+x) Predictions
# Instead of transforming back to original scale, we'll work directly with log values

library(mgcv)
library(dplyr)
library(ggplot2)
library(viridis)
library(tidyr)

# --- STEP 1: Prepare data ----
pair_product <- pair_product %>%
  filter(sale_start >= 0) %>%
  mutate(
    log1p_sale_start = log(1 + sale_start),
    fyear_start = factor(fyear_start),
    sic_start   = factor(sic_start)
  )

controls <- c(
  "at_start", "emp_start", "ebitda_start", "xrd_recv",
  "at_recv", "emp_recv", "fyear_recv", "concentration_start",
  "concentration_recv"
)

# --- STEP 2: Fit GAM models with bam() ----
fit_model <- function(formula_text) {
  bam(
    as.formula(formula_text),
    data = pair_product,
    discrete = TRUE,
    nthreads = 30
  )
}

smooth_term     <- "s(product_similarity, technology_similarity)"
control_formula <- paste(controls, collapse = " + ")
base_formula    <- paste("log1p_sale_start ~", smooth_term, "+", control_formula)

formulas <- list(
  controls_only     = base_formula,
  year_fe           = paste0(base_formula, " + fyear_start"),
  sector_fe         = paste0(base_formula, " + sic_start"),
  year_sector_fe    = paste0(base_formula, " + fyear_start + sic_start")
)

model_controls_only  <- fit_model(formulas$controls_only);  gc()
model_year_fe        <- fit_model(formulas$year_fe);         gc()
model_sector_fe      <- fit_model(formulas$sector_fe);       gc()
model_year_sector_fe <- fit_model(formulas$year_sector_fe);  gc()

models <- list(
  model_controls_only,
  model_year_fe,
  model_sector_fe,
  model_year_sector_fe
)

# --- STEP 3: Create prediction grid ----
control_medians <- pair_product %>%
  summarise(across(all_of(controls), ~median(., na.rm = TRUE)))

grid_data <- expand.grid(
  product_similarity = seq(0, 1, length.out = 100),
  technology_similarity = seq(0, 1, length.out = 100)
)
grid_data <- bind_cols(grid_data, control_medians[rep(1, nrow(grid_data)), ])

grid_data$fyear_start <- factor(rep(levels(pair_product$fyear_start)[1], nrow(grid_data)),
                                levels = levels(pair_product$fyear_start))
grid_data$sic_start <- factor(rep(levels(pair_product$sic_start)[1], nrow(grid_data)),
                              levels = levels(pair_product$sic_start))

# --- Create density map to mask low-observation areas ----
pair_product_density <- pair_product %>%
  mutate(
    ps_bin = round(product_similarity, 2),
    ts_bin = round(technology_similarity, 2)
  ) %>%
  count(ts_bin, ps_bin, name = "density")

grid_data <- grid_data %>%
  mutate(
    ps_bin = round(product_similarity, 2),
    ts_bin = round(technology_similarity, 2)
  ) %>%
  left_join(pair_product_density, by = c("ts_bin", "ps_bin"))

# Mask areas with low density
density_threshold <- 10
grid_data <- grid_data %>%
  mutate(use_point = ifelse(is.na(density) | density < density_threshold, FALSE, TRUE))

# --- STEP 4: Modified plotting function for direct log predictions ----
plot_log_heatmap <- function(model, title) {
  # CHANGED: Predict log(1+sales) directly without transformation
  grid_data$predicted_log_sales <- predict(model, newdata = grid_data, type = "response")
  
  ggplot(grid_data %>% filter(use_point), 
         aes(x = technology_similarity, y = product_similarity, fill = predicted_log_sales)) +
    geom_tile() +
    scale_fill_viridis_c(
      option = "magma",
      name = "Predicted\nLog(1+Sales)"  # Updated legend title
    ) +
    labs(
      title = title,
      x = "Technology Similarity",
      y = "Product Similarity",
      fill = "Predicted Log(1+Sales)"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 12, face = "bold"),
      axis.title = element_text(size = 11),
      legend.title = element_text(size = 10)
    )
}

# Updated titles to reflect log predictions
titles <- c(
  "GAM: Controls Only - Log(1+Sales) Predictions",
  "GAM: Controls + Year FE - Log(1+Sales) Predictions", 
  "GAM: Controls + Sector FE - Log(1+Sales) Predictions",
  "GAM: Controls + Year + Sector FE - Log(1+Sales) Predictions"
)

plots <- Map(plot_log_heatmap, models, titles)

# Display plots
for (p in plots) print(p)

# Save plots with updated filenames
filenames <- c(
  "heatmap_controls_only_log_sales_direct.png",
  "heatmap_year_fe_log_sales_direct.png", 
  "heatmap_sector_fe_log_sales_direct.png",
  "heatmap_year_sector_fe_log_sales_direct.png"
)

Map(ggsave, filename = filenames, plot = plots, width = 8, height = 6)

# --- STEP 5: Additional analysis - Summary statistics of log predictions ----
cat("=== SUMMARY OF LOG(1+SALES) PREDICTIONS ===\n")
for(i in 1:length(models)) {
  model <- models[[i]]
  title <- titles[[i]]
  
  grid_data$predicted_log_sales <- predict(model, newdata = grid_data, type = "response")
  filtered_predictions <- grid_data$predicted_log_sales[grid_data$use_point]
  
  cat("\n", title, ":\n")
  cat("Min predicted log(1+sales):", round(min(filtered_predictions, na.rm = TRUE), 3), "\n")
  cat("Max predicted log(1+sales):", round(max(filtered_predictions, na.rm = TRUE), 3), "\n")
  cat("Mean predicted log(1+sales):", round(mean(filtered_predictions, na.rm = TRUE), 3), "\n")
  cat("Range:", round(max(filtered_predictions, na.rm = TRUE) - min(filtered_predictions, na.rm = TRUE), 3), "\n")
}

# --- STEP 6: Compare actual vs predicted log values ----
cat("\n=== ACTUAL vs PREDICTED LOG(1+SALES) COMPARISON ===\n")
cat("Actual log(1+sales) in data:\n")
cat("Min:", round(min(pair_product$log1p_sale_start, na.rm = TRUE), 3), "\n")
cat("Max:", round(max(pair_product$log1p_sale_start, na.rm = TRUE), 3), "\n") 
cat("Mean:", round(mean(pair_product$log1p_sale_start, na.rm = TRUE), 3), "\n")
cat("Median:", round(median(pair_product$log1p_sale_start, na.rm = TRUE), 3), "\n")

cat("\nNote: Predictions are now in log(1+sales) scale directly.\n")
cat("To convert back to original sales scale: exp(predicted_log_sales) - 1\n")



library(ggplot2)
library(viridis)
library(scales) # Essential for label_number()

# Ensure scientific notation is suppressed for axis labels if desired
options(scipen = 999)

ggplot(pair_product, aes(x = technology_similarity, y = product_similarity)) +
  geom_bin2d(bins = 50) +
  scale_fill_gradient(
    low = "lightblue",  # Starting color for low values (lighter blue)
    high = "darkblue", # Ending color for high values (darker blue)
    trans = "log10",
    # Manually define a set of breaks that are well-spaced on a log scale
    # but also include more intermediate values at the higher end.
    # Adjust these values based on the actual range of your 'count' data.
    breaks = c(10, 100, 1000, 10000, 100000, 1000000),
    # Use scales::label_number to format the breaks as full numbers
    # without scientific notation or "K" abbreviations.
    labels = scales::label_number(big.mark = ",", accuracy = 1), # Adds comma for thousands
    # Optional: Set limits if you want to ensure the scale spans a precise range
    # limits = c(min_actual_count, max_actual_count)
  ) +
  labs(title = "Density of Observations in Similarity Space",
       fill = "Count (Log10)") # Keep legend title consistent




# Load required libraries
library(ggplot2)
library(dplyr)
library(viridis)

# Create bins for technology_similarity and product_similarity with 0.01 dimension
pair_product <- pair_product %>%
  mutate(
    tech_sim_bin = round(technology_similarity / 0.01) * 0.01,
    prod_sim_bin = round(product_similarity / 0.01) * 0.01
  )

# Calculate average sale_start for each bin combination
heatmap_data <- pair_product %>%
  group_by(tech_sim_bin, prod_sim_bin) %>%
  summarise(
    avg_sale_start = mean(sale_start, na.rm = TRUE),
    count = n(),
    .groups = 'drop'
  ) %>%
  filter(!is.na(avg_sale_start) & avg_sale_start > 0) %>%  # Remove NA and zero/negative values
  mutate(log_avg_sale_start = log(avg_sale_start))  # Add log transformation

# Create the heatmap
heatmap_plot <- ggplot(heatmap_data, aes(x = tech_sim_bin, y = prod_sim_bin, fill = log_avg_sale_start)) +
  geom_tile() +
  scale_fill_viridis_c(
    option = "viridis",
    name = "Log(Average\nsale_start)"
  ) +
  labs(
    title = "Heatmap of Log(Average sale_start)",
    subtitle = "By Technology Similarity (x-axis) and Product Similarity (y-axis)",
    x = "Technology Similarity",
    y = "Product Similarity"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12),
    axis.text = element_text(size = 10),
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 9)
  ) +
  coord_fixed(ratio = 1)  # Keep square aspect ratio

# Display the plot
print(heatmap_plot)

# Optional: Save the plot
# ggsave("sale_start_heatmap.png", plot = heatmap_plot, width = 10, height = 8, dpi = 300)