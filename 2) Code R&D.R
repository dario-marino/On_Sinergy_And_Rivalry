# ============================================================================
# PART 0: LINEAR FIXED EFFECTS MODELS — NO INTERACTION (LEVELS)
# ============================================================================

cat("\n=== LINEAR FIXED EFFECTS MODELS — NO INTERACTION (LEVELS) ===\n")

controls <- c(
  "at_start", "emp_start", "at_recv", "emp_recv",
  "xrd_recv", "concentration_start", "concentration_recv",
  "ebitda_start"
)

control_formula <- paste(controls, collapse = " + ")

base_formula_levels <- paste(
  "xrd_start ~ product_similarity + technology_similarity +",
  control_formula
)

lin_lvl_1 <- feols(as.formula(base_formula_levels), data = pair_product)

lin_lvl_2 <- feols(
  as.formula(paste(base_formula_levels, "| fyear_start")),
  data = pair_product
)

lin_lvl_3 <- feols(
  as.formula(paste(base_formula_levels, "| sic_start")),
  data = pair_product,
  cluster = ~ sic_start
)

lin_lvl_4 <- feols(
  as.formula(paste(base_formula_levels, "| fyear_start + sic_start")),
  data = pair_product,
  cluster = ~ sic_start
)

etable(
  lin_lvl_1, lin_lvl_2, lin_lvl_3, lin_lvl_4,
  title = "Linear Effects on R&D (Levels, No Interaction)",
  headers = c("Controls", "Time FE", "Sector FE", "Time + Sector FE")
)


# ============================================================================
# PART 0: LINEAR FIXED EFFECTS MODELS — NO INTERACTION (LOG(1+R&D))
# ============================================================================

cat("\n=== LINEAR FIXED EFFECTS MODELS — NO INTERACTION (LOG(1+R&D)) ===\n")

controls_log <- c(
  "log1p_at_start", "log1p_emp_start",
  "log1p_at_recv", "log1p_emp_recv",
  "log1p_xrd_recv",
  "concentration_start", "concentration_recv",
  "log1p_ebitda_start"
)

control_formula_log <- paste(controls_log, collapse = " + ")

base_formula_log <- paste(
  "log1p_xrd_start ~ product_similarity + technology_similarity +",
  control_formula_log
)

lin_log_1 <- feols(as.formula(base_formula_log), data = pair_product)

lin_log_2 <- feols(
  as.formula(paste(base_formula_log, "| fyear_start")),
  data = pair_product
)

lin_log_3 <- feols(
  as.formula(paste(base_formula_log, "| sic_start")),
  data = pair_product,
  cluster = ~ sic_start
)

lin_log_4 <- feols(
  as.formula(paste(base_formula_log, "| fyear_start + sic_start")),
  data = pair_product,
  cluster = ~ sic_start
)

etable(
  lin_log_1, lin_log_2, lin_log_3, lin_log_4,
  title = "Linear Effects on R&D (log(1+R&D), No Interaction)",
  headers = c("Controls", "Time FE", "Sector FE", "Time + Sector FE")
)




###########REGRESSION 1 (to ask to see the effect like in bloom also on sales, quality citation, and our measure
# Comprehensive R&D Investment Analysis
# Effect of Product and Technology Similarity on R&D Investment (Normal Values)
# pair_product <- read.csv("/mnt/ide0/home/dariomarino/pair_product.csv")

library(fixest)
library(mgcv)
library(parallel)
library(dplyr)
library(ggplot2)
library(viridis)
library(gridExtra)

# Set up parallel processing
cl <- makeCluster(30)
clusterEvalQ(cl, library(mgcv))

# ============================================================================
# DATA PREPARATION: CLEAN R&D DATA (NORMAL VALUES)
# ============================================================================

# Remove negative R&D values and ensure all variables are in normal (non-log) form
pair_product <- pair_product %>%
  filter(xrd_start >= 0, xrd_recv >= 0) %>%  # Remove negative R&D values
  mutate(
    # Ensure all variables are in normal values (remove any existing log transformations)
    # R&D variables (dependent and control)
    xrd_start = as.numeric(xrd_start),
    xrd_recv = as.numeric(xrd_recv),
    
    # Other financial variables in normal values
    at_start = as.numeric(at_start),
    at_recv = as.numeric(at_recv),
    emp_start = as.numeric(emp_start),
    emp_recv = as.numeric(emp_recv),
    ebitda_start = as.numeric(ebitda_start)
  )

# Check the distribution of normal R&D values
cat("=== NORMAL R&D DISTRIBUTION SUMMARY (AFTER REMOVING NEGATIVE VALUES) ===\n")
cat("R&D start (xrd_start) summary:\n")
print(summary(pair_product$xrd_start))
cat("\nR&D recv (xrd_recv) summary:\n")
print(summary(pair_product$xrd_recv))
cat("\nNumber of zero/missing R&D observations:\n")
cat("Zeros in xrd_start:", sum(pair_product$xrd_start == 0, na.rm = TRUE), "\n")
cat("NAs in xrd_start:", sum(is.na(pair_product$xrd_start)), "\n")

# Check other control variables
cat("\n=== CONTROL VARIABLES SUMMARY (NORMAL VALUES) ===\n")
cat("Assets start (at_start) summary:\n")
print(summary(pair_product$at_start))
cat("\nEmployees start (emp_start) summary:\n")
print(summary(pair_product$emp_start))
cat("\nEBITDA start (ebitda_start) summary:\n")
print(summary(pair_product$ebitda_start))

# ============================================================================
# PART 1: LINEAR FIXED EFFECTS MODELS WITH INTERACTIONS (NORMAL VALUES)
# ============================================================================
cat("\n=== LINEAR FIXED EFFECTS MODELS WITH INTERACTIONS (NORMAL VALUES) ===\n")

# Define control variables using normal (non-log) values
controls <- c("at_start", "emp_start", "at_recv", "emp_recv", "xrd_recv", 
              "concentration_start", "concentration_recv", "ebitda_start")

# Create formula components
control_formula <- paste(controls, collapse = " + ")
base_formula <- paste("xrd_start ~", control_formula)

# Add the main variables of interest and interaction term
interaction_vars <- " + product_similarity + technology_similarity + product_similarity:technology_similarity"

# Interaction models with different fixed effects specifications
interaction_model1 <- feols(as.formula(paste(base_formula, interaction_vars)), 
                            data = pair_product)

interaction_model2 <- feols(as.formula(paste(base_formula, interaction_vars, "| fyear_start")), 
                            data = pair_product)

interaction_model3 <- feols(as.formula(paste(base_formula, interaction_vars, "| sic_start")), 
                            data = pair_product, 
                            cluster = ~sic_start)

interaction_model4 <- feols(as.formula(paste(base_formula, interaction_vars, "| fyear_start + sic_start")), 
                            data = pair_product, 
                            cluster = ~sic_start)

# Display results - Interaction Models Only
cat("\n=== INTERACTION MODELS RESULTS ===\n")
cat("\nInteraction Model 1 - Controls Only (Normal Values):\n")
print(summary(interaction_model1))

cat("\nInteraction Model 2 - Controls + Time FE (Normal Values):\n")
print(summary(interaction_model2))

cat("\nInteraction Model 3 - Controls + Sector FE (Clustered SE, Normal Values):\n")
print(summary(interaction_model3))

cat("\nInteraction Model 4 - Controls + Time FE + Sector FE (Clustered SE, Normal Values):\n")
print(summary(interaction_model4))

# Create comparison table for interaction models
cat("\n=== INTERACTION EFFECTS COMPARISON ===\n")
etable(interaction_model1, interaction_model2, interaction_model3, interaction_model4,
       title = "R&D Investment Regression Results - Normal Values (With Interactions)",
       headers = c("Controls", "Controls + Time FE", "Controls + Sector FE", "Full Model"))

# Additional diagnostics for normal values
cat("\n=== MODEL DIAGNOSTICS FOR NORMAL VALUES ===\n")
cat("Sample sizes:\n")
cat("Model 1:", nobs(interaction_model1), "observations\n")
cat("Model 2:", nobs(interaction_model2), "observations\n")
cat("Model 3:", nobs(interaction_model3), "observations\n")
cat("Model 4:", nobs(interaction_model4), "observations\n")

# Check for potential issues with normal values (e.g., extreme values)
cat("\n=== POTENTIAL DATA ISSUES CHECK ===\n")
cat("Extreme values in dependent variable (xrd_start):\n")
cat("99th percentile:", quantile(pair_product$xrd_start, 0.99, na.rm = TRUE), "\n")
cat("Maximum value:", max(pair_product$xrd_start, na.rm = TRUE), "\n")
cat("Number of observations > 99th percentile:", 
    sum(pair_product$xrd_start > quantile(pair_product$xrd_start, 0.99, na.rm = TRUE), na.rm = TRUE), "\n")





###########REGRESSION 1 (to ask to see the effect like in bloom also on sales, quality citation, and our measure
# Comprehensive R&D Investment Analysis
# Effect of Product and Technology Similarity on R&D Investment (Log(1+x) Values)
# pair_product <- read.csv("/mnt/ide0/home/dariomarino/pair_product.csv")

library(fixest)
library(mgcv)
library(parallel)
library(dplyr)
library(ggplot2)
library(viridis)
library(gridExtra)

# Set up parallel processing
pair_product <- pair_product %>% filter(xrd_start >= 0)

cl <- makeCluster(30)
clusterEvalQ(cl, library(mgcv))

# ============================================================================
# DATA PREPARATION: CLEAN R&D DATA AND CREATE LOG(1+X) VARIABLES
# ============================================================================

# Remove negative R&D values and create log(1+x) R&D variables
pair_product <- pair_product %>%
  filter(xrd_start >= 0, xrd_recv >= 0) %>%  # Remove negative R&D values
  mutate(
    # Log(1+x) of R&D start (dependent variable)
    log1p_xrd_start = log(1 + xrd_start),
    
    # Log(1+x) of R&D recv (control variable)
    log1p_xrd_recv = log(1 + xrd_recv),
    
    # Log(1+x) of other financial variables for consistency
    log1p_at_start = log(1 + pmax(0, at_start)),      # Ensure non-negative before log
    log1p_at_recv = log(1 + pmax(0, at_recv)),
    log1p_emp_start = log(1 + pmax(0, emp_start)),
    log1p_emp_recv = log(1 + pmax(0, emp_recv)),
    log1p_ebitda_start = log(1 + pmax(0, ebitda_start))
  )

# Check the distribution of log(1+x) R&D values
cat("=== LOG(1+X) R&D DISTRIBUTION SUMMARY (AFTER REMOVING NEGATIVE VALUES) ===\n")
cat("Original xrd_start summary:\n")
print(summary(pair_product$xrd_start))
cat("\nLog(1+x) xrd_start summary:\n")
print(summary(pair_product$log1p_xrd_start))
cat("\nNumber of zero R&D observations (now handled by log(1+x)):\n")
cat("Zeros in xrd_start:", sum(pair_product$xrd_start == 0, na.rm = TRUE), "\n")
cat("NAs in xrd_start:", sum(is.na(pair_product$xrd_start)), "\n")
cat("NAs in log1p_xrd_start:", sum(is.na(pair_product$log1p_xrd_start)), "\n")

# ============================================================================
# PART 1: LINEAR FIXED EFFECTS MODELS WITH INTERACTIONS (LOG(1+X) R&D)
# ============================================================================
cat("\n=== LINEAR FIXED EFFECTS MODELS WITH INTERACTIONS (LOG(1+X) R&D) ===\n")

# Define control variables using log(1+x)-transformed versions
controls <- c("log1p_at_start", "log1p_emp_start", "log1p_at_recv", "log1p_emp_recv", 
              "log1p_xrd_recv", "concentration_start", "concentration_recv", "log1p_ebitda_start")

# Create formula components
control_formula <- paste(controls, collapse = " + ")
base_formula <- paste("log1p_xrd_start ~", control_formula)

# Add the main variables of interest and interaction term
interaction_vars <- " + product_similarity + technology_similarity + product_similarity:technology_similarity"

# Interaction models with different fixed effects specifications
interaction_model1 <- feols(as.formula(paste(base_formula, interaction_vars)), 
                            data = pair_product)

interaction_model2 <- feols(as.formula(paste(base_formula, interaction_vars, "| fyear_start")), 
                            data = pair_product)

interaction_model3 <- feols(as.formula(paste(base_formula, interaction_vars, "| sic_start")), 
                            data = pair_product, 
                            cluster = ~sic_start)

interaction_model4 <- feols(as.formula(paste(base_formula, interaction_vars, "| fyear_start + sic_start")), 
                            data = pair_product, 
                            cluster = ~sic_start)

# Display results - Interaction Models Only
cat("\n=== INTERACTION MODELS RESULTS ===\n")
cat("\nInteraction Model 1 - Controls Only (Log(1+x) R&D):\n")
print(summary(interaction_model1))

cat("\nInteraction Model 2 - Controls + Time FE (Log(1+x) R&D):\n")
print(summary(interaction_model2))

cat("\nInteraction Model 3 - Controls + Sector FE (Clustered SE, Log(1+x) R&D):\n")
print(summary(interaction_model3))

cat("\nInteraction Model 4 - Controls + Time FE + Sector FE (Clustered SE, Log(1+x) R&D):\n")
print(summary(interaction_model4))

# Create comparison table for interaction models
cat("\n=== INTERACTION EFFECTS COMPARISON ===\n")
etable(interaction_model1, interaction_model2, interaction_model3, interaction_model4,
       title = "Log(1+x) R&D Investment Regression Results (With Interactions)",
       headers = c("Controls", "Controls + Time FE", "Controls + Sector FE", "Full Model"))

# Additional diagnostics for log(1+x) transformation
cat("\n=== LOG(1+X) TRANSFORMATION DIAGNOSTICS ===\n")
cat("Distribution of log(1+x) transformed variables:\n")
sapply(pair_product[c("log1p_xrd_start", "log1p_xrd_recv", "log1p_at_start", 
                      "log1p_emp_start", "log1p_ebitda_start")], 
       function(x) c(mean = mean(x, na.rm = TRUE), 
                     median = median(x, na.rm = TRUE),
                     sd = sd(x, na.rm = TRUE),
                     min = min(x, na.rm = TRUE),
                     max = max(x, na.rm = TRUE)))




#############################################
# GLM / PPML ROBUSTNESS (THESIS TABLES 5 & 6)
#############################################

library(fixest)
library(dplyr)

# ------------------------------------------------------------------
# DATA (same sample as thesis: keep zeros, drop negatives)
# ------------------------------------------------------------------

pp <- pair_product %>%
  filter(xrd_start >= 0, xrd_recv >= 0) %>%
  mutate(
    log1p_xrd_start = log(1 + xrd_start)
  )

# Controls exactly as in thesis
controls <- c(
  "at_start", "emp_start",
  "at_recv", "emp_recv",
  "xrd_recv",
  "concentration_start", "concentration_recv",
  "ebitda_start"
)

control_formula <- paste(controls, collapse = " + ")

# RHS with interaction
rhs <- paste(
  "product_similarity + technology_similarity +",
  "product_similarity:technology_similarity +",
  control_formula
)

# ================================================================
# TABLE 5: PPML — LEVELS OF R&D (log link)
# ================================================================

# (1) No FE
ppml_1 <- fepois(
  as.formula(paste("xrd_start ~", rhs)),
  data = pp,
  cluster = ~ sic_start + fyear_start
)

# (2) Year FE
ppml_2 <- fepois(
  as.formula(paste("xrd_start ~", rhs, "| fyear_start")),
  data = pp,
  cluster = ~ sic_start + fyear_start
)

# (3) Sector FE
ppml_3 <- fepois(
  as.formula(paste("xrd_start ~", rhs, "| sic_start")),
  data = pp,
  cluster = ~ sic_start + fyear_start
)

# (4) Year + Sector FE
ppml_4 <- fepois(
  as.formula(paste("xrd_start ~", rhs, "| fyear_start + sic_start")),
  data = pp,
  cluster = ~ sic_start + fyear_start
)

etable(
  ppml_1, ppml_2, ppml_3, ppml_4,
  title = "R&D — PPML (Levels) with FE (Two-Way Clustered SEs)",
  headers = c("No FE", "Year FE", "Sector FE", "Year + Sector FE")
)

# ================================================================
# TABLE 6: GAUSSIAN GLM — log(1 + R&D)
# ================================================================

# (1) No FE
glm_1 <- feols(
  as.formula(paste("log1p_xrd_start ~", rhs)),
  data = pp,
  cluster = ~ sic_start + fyear_start
)

# (2) Year FE
glm_2 <- feols(
  as.formula(paste("log1p_xrd_start ~", rhs, "| fyear_start")),
  data = pp,
  cluster = ~ sic_start + fyear_start
)

# (3) Sector FE
glm_3 <- feols(
  as.formula(paste("log1p_xrd_start ~", rhs, "| sic_start")),
  data = pp,
  cluster = ~ sic_start + fyear_start
)

# (4) Year + Sector FE
glm_4 <- feols(
  as.formula(paste("log1p_xrd_start ~", rhs, "| fyear_start + sic_start")),
  data = pp,
  cluster = ~ sic_start + fyear_start
)

etable(
  glm_1, glm_2, glm_3, glm_4,
  title = "R&D — Gaussian on log(1 + R&D) with FE (Two-Way Clustered SEs)",
  headers = c("No FE", "Year FE", "Sector FE", "Year + Sector FE")
)






################
GAM
################


library(ggplot2)
library(dplyr)

pair_product <- pair_product %>%
  filter(xrd_start >= 0)

# Load necessary packages
library(mgcv)
library(dplyr)
library(ggplot2)
library(tidyr)

# --- STEP 1: Prepare data ----
pair_product <- pair_product %>%
  filter(xrd_start >= 0) %>%
  mutate(
    # Remove log transformation - use actual R&D values
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
# Change dependent variable from log_xrd_start to xrd_start
base_formula    <- paste("xrd_start ~", smooth_term, "+", control_formula)

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
  # No need to exponentiate since we're already working with actual R&D values
  grid_data$predicted_rnd <- predict(model, newdata = grid_data, type = "response")
  
  ggplot(grid_data %>% filter(use_point), aes(x = technology_similarity, y = product_similarity, fill = predicted_rnd)) +
    geom_tile() +
    scale_fill_viridis_c() +
    labs(
      title = title,
      x = "Technology Similarity",
      y = "Product Similarity",
      fill = "Predicted R&D"
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
  "heatmap_controls_only.png",
  "heatmap_year_fe.png",
  "heatmap_sector_fe.png",
  "heatmap_year_sector_fe.png"
)

Map(ggsave, filename = filenames, plot = plots, width = 8, height = 6)



#########
##density
####



plot_heat <- function(data, value_col, title, palette = "viridis") {
  ggplot(data, aes(x = t_mid, y = p_mid, fill = .data[[value_col]])) +
    geom_tile() +
    coord_fixed(expand = FALSE) +
    scale_x_continuous("technology_similarity", limits = c(0,1), breaks = seq(0,1,0.1)) +
    scale_y_continuous("product_similarity", limits = c(0,1), breaks = seq(0,1,0.1)) +
    scale_fill_viridis_c(na.value = NA, option = palette) +
    labs(title = title, fill = "avg") +
    theme_minimal(base_size = 12) +
    theme(panel.grid = element_blank())
}

# R&D (xrd_start) → viridis
p1 <- plot_heat(agg, "mean_xrd",        "Average xrd_start",        palette = "viridis")
p3 <- plot_heat(agg, "mean_log1p_xrd",  "Average log(1 + xrd_start)",palette = "viridis")

# Sales → magma
p2 <- plot_heat(agg, "mean_sale",       "Average sale_start",       palette = "magma")
p4 <- plot_heat(agg, "mean_log1p_sale", "Average log(1 + sale_start)",palette = "magma")

print(p1); print(p2); print(p3); print(p4)







pair_product <- read.csv("/mnt/ide0/home/dariomarino/pair_product.csv")
################
# OPTIMIZED XGBOOST (Tweedie) version with FAST TRAINING + PARALLELIZATION
# - Option 1: Faster parameters and aggressive early stopping
# - Option 2: Parallel training of all 4 specifications
# - Optimized for 32-core system
################

# Packages
library(dplyr)
library(tidyr)
library(ggplot2)
library(xgboost)
library(Matrix)
library(viridis)
library(future)     # For parallel processing
library(furrr)      # For functional parallel programming
library(purrr)      # For map functions

gc()  # initial cleanup

# SOLUTION 1: Increase the global size limit to accommodate large datasets
options(future.globals.maxSize = 15 * 1024^3)  # 15 GB limit (increased further)

# Set up parallel processing for 32 cores
# Reserve some cores for system (use 28 out of 32)
plan(multisession, workers = 4)  # 4 model specs in parallel, each using 7 cores

# -----------------------------
# STEP 0: Data prep (match GAM)
# -----------------------------
pair_product <- pair_product %>%
  filter(xrd_start >= 0) %>%  # Tweedie expects positive outcomes
  mutate(
    fyear_start = factor(fyear_start),
    sic_start   = factor(sic_start)
  )

controls <- c(
  "at_start", "emp_start", "ebitda_start", "xrd_recv",
  "at_recv", "emp_recv", "fyear_recv", "concentration_start",
  "concentration_recv"
)

gc()  # after data prep

set.seed(42)

# Train/valid split for early stopping
n <- nrow(pair_product)
idx <- sample.int(n, size = floor(0.8 * n))
train_df <- pair_product[idx, ]
valid_df <- pair_product[-idx, ]
rm(idx, n); gc()

# Levels present in TRAIN
year_levels_train   <- levels(droplevels(train_df$fyear_start))
sector_levels_train <- levels(droplevels(train_df$sic_start))

# Most common levels for grid reference
most_common_year_tr <- names(sort(table(train_df$fyear_start), decreasing = TRUE))[1]
most_common_sic_tr  <- names(sort(table(train_df$sic_start), decreasing = TRUE))[1]
gc()

# ------------------------------------------
# Helpers: features, matrix build, alignment
# ------------------------------------------
rhs_for_spec <- function(include_year = FALSE, include_sector = FALSE) {
  rhs <- c("product_similarity", "technology_similarity", controls)
  if (include_year)   rhs <- c(rhs, "fyear_start")
  if (include_sector) rhs <- c(rhs, "sic_start")
  rhs
}

# Add/drop/reorder columns to match training feature names
align_matrix <- function(mm_new, target_cols) {
  new_cols <- colnames(mm_new)
  if (is.null(new_cols)) new_cols <- character(0)
  
  missing <- setdiff(target_cols, new_cols)
  extra   <- setdiff(new_cols, target_cols)
  
  # add missing zero columns
  if (length(missing) > 0) {
    add <- Matrix(0, nrow = nrow(mm_new), ncol = length(missing), sparse = TRUE)
    colnames(add) <- missing
    mm_new <- cbind(mm_new, add)
  }
  # drop extras
  if (length(extra) > 0) {
    keep <- setdiff(colnames(mm_new), extra)
    mm_new <- mm_new[, keep, drop = FALSE]
  }
  # reorder to target
  mm_new <- mm_new[, target_cols, drop = FALSE]
  mm_new
}

# Build sparse matrix + DMatrix for TRAIN/VALID (label required)
prep_matrix <- function(data, rhs, terms_obj = NULL, target_cols = NULL) {
  keep_vars <- c("xrd_start", rhs)
  data2 <- tidyr::drop_na(data, dplyr::all_of(keep_vars))
  
  f <- as.formula(paste("~", paste(rhs, collapse = " + "), "- 1"))
  if (is.null(terms_obj)) {
    terms_obj <- terms(f)
  }
  mm <- Matrix::sparse.model.matrix(terms_obj, data = data2, na.action = stats::na.pass)
  
  # Align to training columns if provided (for VALID)
  if (!is.null(target_cols)) {
    mm <- align_matrix(mm, target_cols)
  }
  
  stopifnot(nrow(mm) == nrow(data2))  # sanity check
  cols <- colnames(mm)
  dmat <- xgboost::xgb.DMatrix(data = mm, label = data2$xrd_start)
  
  rm(mm); gc()
  
  list(
    dmat  = dmat,
    terms = terms_obj,
    cols  = cols,
    n     = nrow(data2)
  )
}

# Build sparse matrix for NEWDATA (no label)
prep_matrix_newdata <- function(newdata, terms_obj, target_cols) {
  mm <- Matrix::sparse.model.matrix(terms_obj, data = newdata, na.action = stats::na.pass)
  mm <- align_matrix(mm, target_cols)
  dnew <- xgboost::xgb.DMatrix(mm)
  rm(mm); gc()
  dnew
}

# ------------------------------
# OPTIMIZED XGBoost configuration (OPTION 1: FAST TRAINING)
# ------------------------------
xgb_params_fast <- list(
  objective = "reg:tweedie",
  tweedie_variance_power = 1.4,
  eta = 0.1,                    # Increased from 0.05 (faster convergence)
  max_depth = 4,                # Reduced from 6 (simpler, faster trees)
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 10,
  lambda = 1.0,
  alpha = 0.0,
  eval_metric = "rmse",
  nthread = 7                   # 32 cores / 4 parallel jobs = 8 cores per job, reserve 1
)

fit_xgb_fast <- function(dtrain, watchlist, params,
                         nrounds = 1000,           # Reduced from 3000
                         early_stopping_rounds = 50, # Reduced from 100
                         nthread = 7) {            # Cores per model
  bst <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = nrounds,
    watchlist = watchlist,
    early_stopping_rounds = early_stopping_rounds,
    print_every_n = 25,         # Print less frequently
    nthread = nthread,
    maximize = FALSE,
    verbose = 1
  )
  gc()
  bst
}

# -----------------------------------
# STEP 1: Prediction grid (100 x 100) - SHARED ACROSS ALL MODELS
# -----------------------------------
control_medians <- pair_product %>%
  summarise(across(all_of(controls), ~median(., na.rm = TRUE)))
gc()

grid_data <- expand.grid(
  product_similarity    = seq(0, 1, length.out = 100),
  technology_similarity = seq(0, 1, length.out = 100)
) %>%
  bind_cols(control_medians[rep(1, nrow(.)), ])
gc()

# Fix FEs to the most common TRAIN level
grid_data$fyear_start <- factor(rep(most_common_year_tr, nrow(grid_data)),
                                levels = year_levels_train)
grid_data$sic_start   <- factor(rep(most_common_sic_tr, nrow(grid_data)),
                                levels = sector_levels_train)
gc()

# ---------------------------------------------
# STEP 2: Density mask (same as your GAM code)
# ---------------------------------------------
pair_product_density <- pair_product %>%
  mutate(
    ps_bin = round(product_similarity, 2),
    ts_bin = round(technology_similarity, 2)
  ) %>%
  count(ts_bin, ps_bin, name = "density")
gc()

grid_data <- grid_data %>%
  mutate(
    ps_bin = round(product_similarity, 2),
    ts_bin = round(technology_similarity, 2)
  ) %>%
  left_join(pair_product_density, by = c("ts_bin", "ps_bin"))
rm(pair_product_density); gc()

density_threshold <- 10
grid_data <- grid_data %>%
  mutate(use_point = ifelse(is.na(density) | density < density_threshold, FALSE, TRUE))
gc()

# -----------------------------------
# STEP 3: Helper functions for parallel execution
# -----------------------------------
predict_grid <- function(model, terms_obj, train_cols, newdata) {
  dnew <- prep_matrix_newdata(newdata, terms_obj, train_cols)
  preds <- as.numeric(predict(model, newdata = dnew))
  rm(dnew); gc()
  preds
}

plot_heatmap <- function(preds, title, newdata) {
  plot_df <- newdata %>%
    mutate(predicted_rnd = preds) %>%
    filter(use_point)
  
  p <- ggplot(plot_df, aes(x = technology_similarity, y = product_similarity, fill = predicted_rnd)) +
    geom_tile() +
    scale_fill_viridis_c() +
    labs(
      title = title,
      x = "Technology Similarity",
      y = "Product Similarity",
      fill = "Predicted R&D"
    ) +
    theme_minimal()
  
  rm(plot_df); gc()
  p
}

# ORIGINAL: PARALLEL TRAINING FUNCTION
train_single_spec <- function(spec_params) {
  include_year <- spec_params$include_year
  include_sector <- spec_params$include_sector
  title <- spec_params$title
  filename <- spec_params$filename
  
  cat("Starting spec:", title, "\n")
  
  # Build rhs and training/validation matrices for this spec
  rhs <- rhs_for_spec(include_year, include_sector)
  tr  <- prep_matrix(train_df, rhs, terms_obj = NULL, target_cols = NULL)
  va  <- prep_matrix(valid_df, rhs, terms_obj = tr$terms, target_cols = tr$cols)
  
  cat(sprintf(
    "Spec[%s]: train=%d, valid=%d, features=%d\n",
    title, tr$n, va$n, length(tr$cols)
  ))
  gc()
  
  # Train model
  watch <- list(train = tr$dmat, valid = va$dmat)
  model <- fit_xgb_fast(tr$dmat, watch, xgb_params_fast)
  
  # Store terms and cols before removing tr and va objects
  terms_obj <- tr$terms
  train_cols <- tr$cols
  
  # Cleanup training objects
  rm(watch, tr, va); gc()
  
  # Predict on grid (use shared grid_data)
  preds <- predict_grid(model, terms_obj, train_cols, grid_data)
  p <- plot_heatmap(preds, title, grid_data)
  
  # Save plot
  ggsave(filename, p, width = 8, height = 6)
  cat("Completed and saved:", filename, "\n")
  
  # Cleanup
  rm(model, preds, terms_obj, train_cols); gc()
  
  # Return results
  list(
    plot = p,
    title = title,
    filename = filename
  )
}

# SOLUTION 2: Sequential processing with explicit parallelization
train_single_spec_sequential <- function(spec_params) {
  include_year <- spec_params$include_year
  include_sector <- spec_params$include_sector
  title <- spec_params$title
  filename <- spec_params$filename
  
  cat("Starting spec:", title, "\n")
  
  # Build rhs and training/validation matrices for this spec
  rhs <- rhs_for_spec(include_year, include_sector)
  tr  <- prep_matrix(train_df, rhs, terms_obj = NULL, target_cols = NULL)
  va  <- prep_matrix(valid_df, rhs, terms_obj = tr$terms, target_cols = tr$cols)
  
  cat(sprintf(
    "Spec[%s]: train=%d, valid=%d, features=%d\n",
    title, tr$n, va$n, length(tr$cols)
  ))
  gc()
  
  # Use more cores per model since we're not running in parallel
  xgb_params_sequential <- xgb_params_fast
  xgb_params_sequential$nthread <- 28  # Use most of the 32 cores
  
  # Train model
  watch <- list(train = tr$dmat, valid = va$dmat)
  model <- fit_xgb_fast(tr$dmat, watch, xgb_params_sequential)
  
  # Store terms and cols before removing tr and va objects
  terms_obj <- tr$terms
  train_cols <- tr$cols
  
  # Cleanup training objects
  rm(watch, tr, va); gc()
  
  # Predict on grid
  preds <- predict_grid(model, terms_obj, train_cols, grid_data)
  p <- plot_heatmap(preds, title, grid_data)
  
  # Save plot
  ggsave(filename, p, width = 8, height = 6)
  cat("Completed and saved:", filename, "\n")
  
  # Cleanup
  rm(model, preds, terms_obj, train_cols); gc()
  
  # Return results
  list(
    plot = p,
    title = title,
    filename = filename
  )
}

# Define all specs
spec_list <- list(
  list(include_year = FALSE, include_sector = FALSE, 
       title = "XGBoost (Tweedie): Controls Only", 
       filename = "xgb_heatmap_controls_only.png"),
  list(include_year = TRUE, include_sector = FALSE, 
       title = "XGBoost (Tweedie): Controls + Year Fixed Effects", 
       filename = "xgb_heatmap_year_fe.png"),
  list(include_year = FALSE, include_sector = TRUE, 
       title = "XGBoost (Tweedie): Controls + Sector Fixed Effects", 
       filename = "xgb_heatmap_sector_fe.png"),
  list(include_year = TRUE, include_sector = TRUE, 
       title = "XGBoost (Tweedie): Controls + Year + Sector Fixed Effects", 
       filename = "xgb_heatmap_year_sector_fe.png")
)

# ------------------------------
# STEP 4A: TRY PARALLEL EXECUTION FIRST (SOLUTION 1)
# ------------------------------
cat("Attempting parallel training of all 4 model specifications...\n")
cat("Using 4 workers with", xgb_params_fast$nthread, "cores each\n")

# Try parallel training first
tryCatch({
  start_time <- Sys.time()
  results <- future_map(spec_list, train_single_spec, .options = furrr_options(seed = 42))
  end_time <- Sys.time()
  
  cat("Parallel training completed successfully in:", round(as.numeric(end_time - start_time), 2), "minutes\n")
  
  # Extract plots for display/further use
  plots <- purrr::map(results, ~.x$plot)
  names(plots) <- purrr::map_chr(results, ~.x$title)
  
  # Display completion summary
  cat("\nParallel Training Summary:\n")
  for(i in seq_along(results)) {
    cat(sprintf("  %d. %s -> %s\n", i, results[[i]]$title, results[[i]]$filename))
  }
  
}, error = function(e) {
  cat("Parallel training failed with error:", e$message, "\n")
  cat("Falling back to sequential training with high parallelization...\n")
  
  # ------------------------------
  # STEP 4B: FALLBACK TO SEQUENTIAL EXECUTION (SOLUTION 2)
  # ------------------------------
  
  # Clean up parallel workers before switching to sequential
  plan(sequential)
  gc()  # Force garbage collection
  
  cat("Starting sequential training with high parallelization per model...\n")
  cat("Using 28 cores per model\n")
  
  start_time <- Sys.time()
  results <<- purrr::map(spec_list, train_single_spec_sequential)
  end_time <- Sys.time()
  
  cat("Sequential training completed in:", round(as.numeric(end_time - start_time), 2), "minutes\n")
  
  # Extract plots for display/further use
  plots <<- purrr::map(results, ~.x$plot)
  names(plots) <<- purrr::map_chr(results, ~.x$title)
  
  # Display completion summary
  cat("\nSequential Training Summary:\n")
  for(i in seq_along(results)) {
    cat(sprintf("  %d. %s -> %s\n", i, results[[i]]$title, results[[i]]$filename))
  }
})

# Clean up parallel workers
plan(sequential)

gc()  # final sweep

cat("Done! All 4 models trained and saved.\n")



################
####GAM
################


# Load necessary packages
library(mgcv)
library(dplyr)
library(ggplot2)
library(tidyr)
library(viridis)

# --- STEP 1: Prepare data ----

pair_product <- pair_product %>%
  filter(xrd_start >= 0) %>%  # Include zeros
  mutate(
    log1p_xrd_start = log(1 + xrd_start),  # log(1 + x) transformation
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
base_formula    <- paste("log1p_xrd_start ~", smooth_term, "+", control_formula)

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

density_threshold <- 10
grid_data <- grid_data %>%
  mutate(use_point = ifelse(is.na(density) | density < density_threshold, FALSE, TRUE))

# --- STEP 4: Plotting with DIRECT log predictions ----

plot_heatmap_log <- function(model, title) {
  # CHANGED: Direct prediction of log(1+R&D) values without transformation
  grid_data$predicted_log_rnd <- predict(model, newdata = grid_data, type = "response")
  
  ggplot(grid_data %>% filter(use_point), aes(x = technology_similarity, y = product_similarity, fill = predicted_log_rnd)) +
    geom_tile() +
    scale_fill_viridis_c() +
    labs(
      title = title,
      x = "Technology Similarity",
      y = "Product Similarity",
      fill = "Predicted\nlog(1+R&D)"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 12, face = "bold"),
      axis.title = element_text(size = 11),
      legend.title = element_text(size = 10)
    )
}

titles <- c(
  "GAM: Controls Only - Direct log(1+R&D) Predictions",
  "GAM: Controls + Year FE - Direct log(1+R&D) Predictions", 
  "GAM: Controls + Sector FE - Direct log(1+R&D) Predictions",
  "GAM: Controls + Year + Sector FE - Direct log(1+R&D) Predictions"
)

plots_log <- Map(plot_heatmap_log, models, titles)

# Display all plots
for (p in plots_log) print(p)

# Save plots with new naming convention
filenames_log <- c(
  "heatmap_controls_only_direct_log.png",
  "heatmap_year_fe_direct_log.png", 
  "heatmap_sector_fe_direct_log.png",
  "heatmap_year_sector_fe_direct_log.png"
)

Map(ggsave, filename = filenames_log, plot = plots_log, width = 8, height = 6)

# --- OPTIONAL: Compare log predictions vs original scale predictions side by side ----

# Function for original scale predictions (your previous approach)
plot_heatmap_original <- function(model, title) {
  grid_data$predicted_rnd <- exp(predict(model, newdata = grid_data, type = "response")) - 1
  
  ggplot(grid_data %>% filter(use_point), aes(x = technology_similarity, y = product_similarity, fill = predicted_rnd)) +
    geom_tile() +
    scale_fill_viridis_c() +
    labs(
      title = paste(title, "(Original Scale)"),
      x = "Technology Similarity", 
      y = "Product Similarity",
      fill = "Predicted R&D"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 12, face = "bold"),
      axis.title = element_text(size = 11),
      legend.title = element_text(size = 10)
    )
}

# Create comparison plots for the full model
comparison_log <- plot_heatmap_log(model_year_sector_fe, "Full Model")
comparison_original <- plot_heatmap_original(model_year_sector_fe, "Full Model")

# Display comparison
print(comparison_log)
print(comparison_original)

# Summary statistics of predictions
cat("\n=== PREDICTION SUMMARY STATISTICS ===\n")
log_preds <- predict(model_year_sector_fe, newdata = grid_data %>% filter(use_point), type = "response")
original_preds <- exp(log_preds) - 1

cat("Direct log(1+R&D) predictions:\n")
cat("Min:", round(min(log_preds, na.rm = TRUE), 4), "\n")
cat("Max:", round(max(log_preds, na.rm = TRUE), 4), "\n") 
cat("Mean:", round(mean(log_preds, na.rm = TRUE), 4), "\n")
cat("Median:", round(median(log_preds, na.rm = TRUE), 4), "\n")

cat("\nTransformed to original scale predictions:\n")
cat("Min:", round(min(original_preds, na.rm = TRUE), 4), "\n")
cat("Max:", round(max(original_preds, na.rm = TRUE), 4), "\n")
cat("Mean:", round(mean(original_preds, na.rm = TRUE), 4), "\n") 
cat("Median:", round(median(original_preds, na.rm = TRUE), 4), "\n")

cat("\nNote: Direct log predictions show the log(1+R&D) values directly from the model\n")
cat("These represent the natural log of (1 + R&D investment)\n")




plot_heat <- function(data, value_col, title, palette = "viridis") {
  ggplot(data, aes(x = t_mid, y = p_mid, fill = .data[[value_col]])) +
    geom_tile() +
    coord_fixed(expand = FALSE) +
    scale_x_continuous("technology_similarity", limits = c(0,1), breaks = seq(0,1,0.1)) +
    scale_y_continuous("product_similarity", limits = c(0,1), breaks = seq(0,1,0.1)) +
    scale_fill_viridis_c(na.value = NA, option = palette) +
    labs(title = title, fill = "avg") +
    theme_minimal(base_size = 12) +
    theme(panel.grid = element_blank())
}

# R&D (xrd_start) → viridis
p1 <- plot_heat(agg, "mean_xrd",        "Average xrd_start",        palette = "viridis")
p3 <- plot_heat(agg, "mean_log1p_xrd",  "Average log(1 + xrd_start)",palette = "viridis")

# Sales → magma
p2 <- plot_heat(agg, "mean_sale",       "Average sale_start",       palette = "magma")
p4 <- plot_heat(agg, "mean_log1p_sale", "Average log(1 + sale_start)",palette = "magma")

print(p1); print(p2); print(p3); print(p4)

