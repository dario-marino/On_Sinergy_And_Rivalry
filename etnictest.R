
### Test if ETNIC data is related to SIC

# 📦 Load data.table and stats
library(data.table)

# 1. Load ETNIC
etnic_data <- fread("/home/dariomarino/Thesis/ETNIC2_data.txt")
company_data <- fread("/home/dariomarino/Thesis/companydata.csv", select = c("gvkey", "sich"))

# 2. Format types
company_data[, gvkey := as.character(gvkey)]
company_data[, sich := sprintf("%04s", as.character(sich))]
etnic_data[, `:=`(gvkey1 = as.character(gvkey1), gvkey2 = as.character(gvkey2))]

# 3. SIC assignment
sic_lookup <- setNames(company_data$sich, company_data$gvkey)
etnic_data[, SIC1 := sic_lookup[gvkey1]]
etnic_data[, SIC2 := sic_lookup[gvkey2]]
etnic_data <- etnic_data[!is.na(SIC1) & !is.na(SIC2)]

# 4. Dummies
etnic_data[, `:=`(
  exact_match   = SIC1 == SIC2,
  first3_match  = substr(SIC1, 1, 3) == substr(SIC2, 1, 3) & SIC1 != SIC2,
  first2_match  = substr(SIC1, 1, 2) == substr(SIC2, 1, 2) & substr(SIC1, 1, 3) != substr(SIC2, 1, 3),
  first1_match  = substr(SIC1, 1, 1) == substr(SIC2, 1, 1) & substr(SIC1, 1, 2) != substr(SIC2, 1, 2)
)]

# 5. Run with industry/year FE
etnic_data[, industry := substr(SIC1, 1, 3)]  # use 3-digit for fewer levels
reg_etnic_fe <- lm(score ~ exact_match + first3_match + first2_match + first1_match +
                     factor(industry) + factor(year), data = etnic_data)
summary(reg_etnic_fe)
