#!/usr/bin/env Rscript
# Optional ML/GAM heatmaps for R&D and Sales (compact, flat).

suppressPackageStartupMessages({
  library(dplyr); library(readr); library(ggplot2); library(viridis)
  library(mgcv);  library(xgboost); library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript 24_ml_gam_heatmaps.R pair_product.csv out_dir/")
}
inp <- args[1]; outdir <- args[2]; dir.create(outdir, showWarnings=FALSE, recursive=TRUE)

pp_raw <- read_csv(inp, show_col_types = FALSE)

# ---------- COMMON SETUP ----------
controls <- c("at_start","emp_start","ebitda_start","xrd_recv",
              "at_recv","emp_recv","concentration_start","concentration_recv")

make_grid <- function(df, add_factors=TRUE){
  g <- expand.grid(
    product_similarity    = seq(0,1,length.out=100),
    technology_similarity = seq(0,1,length.out=100)
  )
  meds <- df %>% summarise(across(all_of(controls), ~median(., na.rm=TRUE)))
  g <- dplyr::bind_cols(g, meds[rep(1, nrow(g)), ])
  if (add_factors){
    g$fyear_start <- factor(levels(df$fyear_start)[1], levels = levels(df$fyear_start))
    g$sic_start   <- factor(levels(df$sic_start)[1],   levels = levels(df$sic_start))
  }
  # density mask
  dens <- df %>%
    mutate(ps=round(product_similarity,2), ts=round(technology_similarity,2)) %>%
    count(ts, ps, name="n")
  g %>%
    mutate(ps=round(product_similarity,2), ts=round(technology_similarity,2)) %>%
    left_join(dens, by=c("ts","ps")) %>%
    mutate(use = ifelse(is.na(n) | n < 10, FALSE, TRUE))
}

plot_map <- function(df_plot, val, ttl, file){
  p <- ggplot(dplyr::filter(df_plot, use),
              aes(technology_similarity, product_similarity, fill = .data[[val]])) +
    geom_tile() +
    scale_fill_viridis_c() +
    labs(x="Technology similarity", y="Product similarity", fill="Predicted", title=ttl) +
    theme_minimal()
  ggsave(file, p, width=8, height=6, dpi=300)
}

# ---------- (A) R&D: GAM ----------
pp_rnd <- pp_raw %>%
  filter(xrd_start > 0) %>%
  mutate(fyear_start=factor(fyear_start), sic_start=factor(sic_start))

gam_rnd <- bam(
  as.formula(paste(
    "xrd_start ~ s(product_similarity, technology_similarity) +",
    paste(controls, collapse=" + "),
    "+ fyear_start + sic_start"
  )),
  data = pp_rnd, family = Gamma(link="log"), discrete = TRUE
)

g_rnd <- make_grid(pp_rnd, add_factors = TRUE)
g_rnd$pred_rnd <- predict(gam_rnd, newdata = g_rnd, type = "response")
plot_map(g_rnd, "pred_rnd", "GAM: Predicted R&D over (δ, ω)", file.path(outdir, "rnd_gam_heatmap.png"))

# ---------- (B) SALES: GAM ----------
pp_sales <- pp_raw %>%
  filter(sale_start > 0) %>%
  mutate(fyear_start=factor(fyear_start), sic_start=factor(sic_start))

gam_sales <- bam(
  as.formula(paste(
    "sale_start ~ s(product_similarity, technology_similarity) +",
    paste(controls, collapse=" + "),
    "+ fyear_start + sic_start"
  )),
  data = pp_sales, family = Gamma(link="log"), discrete = TRUE
)

g_sales <- make_grid(pp_sales, add_factors = TRUE)
g_sales$pred_sales <- predict(gam_sales, newdata = g_sales, type = "response")
plot_map(g_sales, "pred_sales", "GAM: Predicted Sales over (δ, ω)", file.path(outdir, "sales_gam_heatmap.png"))

# ---------- (C) Optional: fast XGBoost Tweedie (skippable if no xgboost) ----------
safe_xgb <- function(df, ycol, title, file){
  if (!requireNamespace("xgboost", quietly = TRUE)) return(invisible(NULL))
  df2 <- df %>% select(all_of(c(ycol, "product_similarity","technology_similarity", controls))) %>% na.omit()
  y   <- df2[[ycol]]
  mm  <- sparse.model.matrix(~ . -1, data = df2 %>% select(-all_of(ycol)))
  dtr <- xgboost::xgb.DMatrix(data = mm, label = y)
  bst <- xgboost::xgb.train(
    params = list(objective="reg:tweedie", tweedie_variance_power=1.4,
                  eta=0.1, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                  eval_metric="rmse"),
    data = dtr, nrounds = 500, verbose = 0
  )
  g <- expand.grid(product_similarity=seq(0,1,length.out=100),
                   technology_similarity=seq(0,1,length.out=100))
  meds <- df2 %>% summarise(across(all_of(controls), ~median(., na.rm=TRUE)))
  g <- dplyr::bind_cols(g, meds[rep(1, nrow(g)), ])
  mmg <- sparse.model.matrix(~ . -1, data = g)
  g$pred <- as.numeric(predict(bst, newdata = mmg))
  p <- ggplot(g, aes(technology_similarity, product_similarity, fill = pred)) +
    geom_tile() + scale_fill_viridis_c() +
    labs(x="Technology similarity", y="Product similarity", fill="Predicted", title=title) +
    theme_minimal()
  ggsave(file, p, width=8, height=6, dpi=300)
}

# R&D XGB (optional)
try(safe_xgb(pp_raw %>% filter(xrd_start > 0), "xrd_start",
             "XGBoost: Predicted R&D over (δ, ω)", file.path(outdir, "rnd_xgb_heatmap.png")), silent = TRUE)

# Sales XGB (optional)
try(safe_xgb(pp_raw %>% filter(sale_start > 0), "sale_start",
             "XGBoost: Predicted Sales over (δ, ω)", file.path(outdir, "sales_xgb_heatmap.png")), silent = TRUE)

message("Saved: rnd_gam_heatmap.png, sales_gam_heatmap.png (and XGB maps if xgboost available)")
