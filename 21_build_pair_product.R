#!/usr/bin/env Rscript
# Build pair_product from final_data + salescompustat (adds Sales/EBITDA & concentration).
suppressPackageStartupMessages({ library(dplyr); library(readr); library(lubridate) })

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("Usage: Rscript 21_build_pair_product.R final_data.csv salescompustat.csv out_pair_product.csv")
final_path <- args[1]; sales_path <- args[2]; out_path <- args[3]

final_data <- read_csv(final_path, show_col_types = FALSE)
salescomp  <- read_csv(sales_path,  show_col_types = FALSE)

salescompustat_combined <- salescomp %>%
  mutate(fyear = year(datadate),
         gvkey_original = gvkey,
         gvkey_numeric  = as.character(as.numeric(gvkey))) %>%
  select(gvkey_original, gvkey_numeric, fyear, sale, ebitda) %>%
  bind_rows(., . %>% select(gvkey = gvkey_numeric, fyear, sale, ebitda)) %>%
  select(gvkey = gvkey_original, fyear, sale, ebitda) %>%
  bind_rows(., salescomp %>% transmute(gvkey = as.character(as.numeric(gvkey)), fyear = year(datadate), sale, ebitda)) %>%
  distinct(gvkey, fyear, .keep_all = TRUE)

pair_product <- final_data %>%
  rename(gvkey_start = gvkey1, gvkey_recv = gvkey2, fyear_start = year, technology_similarity = tech_similarity) %>%
  select(-curcd_starting, -curcd_receiving, -sich_receiving, -sich_starting, -conm1, -conm2) %>%
  rename(conm_start=conm_starting, at_start=at_starting, emp_start=emp_starting, xrd_start=xrd_starting, sic_start=sic_starting,
         conm_recv=conm_receiving, at_recv=at_receiving, emp_recv=emp_receiving, xrd_recv=xrd_receiving, sic_recv=sic_receiving) %>%
  mutate(fyear_recv = fyear_start)

herf <- pair_product %>% group_by(sic_start) %>% summarise(total_at = sum(at_start, na.rm=TRUE), .groups='drop')
conc <- pair_product %>% left_join(herf, by = "sic_start") %>% mutate(ms = at_start/total_at) %>% 
  group_by(sic_start) %>% summarise(concentration = sum(ms^2, na.rm=TRUE), .groups='drop')

pair_product <- pair_product %>%
  left_join(conc %>% rename(concentration_start = concentration), by = "sic_start") %>%
  left_join(conc %>% rename(concentration_recv = concentration), by = c("sic_recv" = "sic_start"))

write_csv(pair_product %>% select(
  gvkey_start, fyear_start, conm_start, at_start, emp_start, xrd_start, sic_start,
  gvkey_recv, fyear_recv, conm_recv, at_recv, emp_recv, xrd_recv, sic_recv,
  concentration_start, concentration_recv, product_similarity, technology_similarity,
  sale_start, ebitda_start
), out_path)
cat("Wrote", out_path, "\n")
