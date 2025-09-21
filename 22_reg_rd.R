#!/usr/bin/env Rscript
# R&D regressions (FE + interaction) and GAM surface.
suppressPackageStartupMessages({ library(fixest); library(mgcv); library(dplyr); library(ggplot2); library(readr) })

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Usage: Rscript 22_reg_rd.R pair_product.csv out_dir/")
inp <- args[1]; outdir <- args[2]; dir.create(outdir, showWarnings=FALSE, recursive=TRUE)

pp <- read_csv(inp, show_col_types = FALSE) %>% filter(xrd_start >= 0)
controls <- c("at_start","emp_start","at_recv","emp_recv","xrd_recv","concentration_start","concentration_recv","ebitda_start")
base <- as.formula(paste("xrd_start ~", paste(controls, collapse=" + "),
                         "+ product_similarity + technology_similarity + product_similarity:technology_similarity"))

m1 <- feols(base, data=pp)
m2 <- feols(update(base, . ~ . | fyear_start), data=pp)
m3 <- feols(update(base, . ~ . | sic_start),  data=pp, cluster=~sic_start)
m4 <- feols(update(base, . ~ . | fyear_start + sic_start), data=pp, cluster=~sic_start)
capture.output(etable(m1,m2,m3,m4, title="R&D Investment Regression Results (levels, with interaction)"),
               file=file.path(outdir,"rd_fe_models.txt"))

pp_pos <- pp %>% filter(xrd_start > 0) %>% mutate(fyear_start=factor(fyear_start), sic_start=factor(sic_start))
gam_form <- as.formula(paste("xrd_start ~ s(product_similarity, technology_similarity) +", paste(controls, collapse=" + "),
                             "+ fyear_start + sic_start"))
gm <- bam(gam_form, data=pp_pos, family=Gamma(link="log"), discrete=TRUE)

grid <- expand.grid(product_similarity=seq(0,1,length.out=100), technology_similarity=seq(0,1,length.out=100))
for (v in controls) grid[[v]] <- median(pp_pos[[v]], na.rm=TRUE)
grid$fyear_start <- levels(pp_pos$fyear_start)[1]; grid$sic_start <- levels(pp_pos$sic_start)[1]

dens <- pp_pos %>% mutate(ps=round(product_similarity,2), ts=round(technology_similarity,2)) %>% count(ts,ps, name="n")
grid2 <- grid %>% mutate(ps=round(product_similarity,2), ts=round(technology_similarity,2)) %>% left_join(dens, by=c("ts","ps"))
grid2$use <- ifelse(is.na(grid2$n) | grid2$n < 10, FALSE, TRUE)

grid2$pred <- predict(gm, newdata=grid2, type="response")
p <- ggplot(dplyr::filter(grid2, use), aes(technology_similarity, product_similarity, fill=pred)) +
  geom_tile() + scale_fill_viridis_c() + labs(x="Technology similarity", y="Product similarity", fill="Pred. R&D",
                                              title="GAM: Predicted R&D over (δ, ω)") + theme_minimal()
ggsave(file.path(outdir,"rd_gam_heatmap.png"), p, width=8, height=6, dpi=300)
