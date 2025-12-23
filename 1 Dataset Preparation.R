library(dplyr)

#select only the necessary columns, enough for HHI
#this is the data you downloaded from Compustat
companydata_selected <- companydata %>%
  select(
    gvkey,       # Identifier
    fyear,       # Fiscal year
    conm,        # Company name
    curcd,       # Currency code
    at,          # Total assets
    emp,         # Number of employees
    xrd,         # Research & Development expenses
    sich         # SIC code
  )

head(companydata_selected)

# now merging technology and product similarity
names(tnic2_data) <- c("year", "gvkey1", "gvkey2", "score")

merged_data <- merge(
  tech_similarity_doc2vec,
  tnic2_data,
  by = c("gvkey1", "gvkey2", "year"),  # Adding year to ensure proper matching
  all = FALSE
)

names(merged_data)[names(merged_data) == "similarity_doc2vec"] <- "tech_similarity"
names(merged_data)[names(merged_data) == "score"] <- "product_similarity"

cat("Number of rows in tech similarity dataset:", nrow(tech_similarity_doc2vec), "\n")
cat("Number of rows in tnic2 dataset:", nrow(tnic2_data), "\n")
cat("Number of rows in merged dataset:", nrow(merged_data), "\n")

write.csv(merged_data, "tech_prod.csv", row.names = FALSE)

tech_products <- merged_data
remove(merged_data)


"""""""""""""""
filtered_company data has info for each company, and as you can see techproducts 
has 2 gvkey and a year for each row. We match gvkey and year twice, 
once to gvkey1 and year, and the other one to gvkey2 and year. 
All the columns of the row filtered_company data twice, the column should
obviously with renamed with a "_starting" at the end for the match gvkey1 and
year to gvkey and year in filtered and with a "_receiving" at the end for the 
match gvkey2 and year. We need the information for both companies
that are connected in every row and I need them in that same row.
"""""""""""""""

filtered_companydata <- companydata_selected %>%
  
  # Ensure identifiers are character and year is numeric/integer
  mutate(
    gvkey = as.character(gvkey),
    fyear = as.integer(fyear)
  ) %>%
  
  # Keep only observations with valid firm-year info
  filter(
    !is.na(gvkey),
    !is.na(fyear),
    
    # Core economic variables required later
    !is.na(at), at > 0,
    !is.na(emp), emp > 0,
    !is.na(xrd), xrd >= 0,
    
    # Industry classification (use SICH consistently)
    !is.na(sich)
  ) %>%
  
  # Remove duplicate firm-year observations if any
  distinct(gvkey, fyear, .keep_all = TRUE)

# Quick diagnostic
cat("Filtered company data rows:", nrow(filtered_companydata), "\n")
cat("Unique firm-year pairs:", 
    nrow(distinct(filtered_companydata, gvkey, fyear)), "\n")

head(filtered_companydata)



# Convert gvkey1 and gvkey2 to character type in tech_products
tech_products <- tech_products %>%
  mutate(
    gvkey1 = as.character(gvkey1),
    gvkey2 = as.character(gvkey2)
  )

# First join for the starting company (gvkey1)
merged_data <- tech_products %>%
  left_join(
    filtered_companydata,
    by = c("gvkey1" = "gvkey", "year" = "fyear"),
    suffix = c("", "_starting"),
    relationship = "many-to-many"
  ) %>%
  rename_with(
    ~paste0(., "_starting"),
    c("conm", "curcd", "at", "emp", "xrd", "sich")
  )

# Second join for the receiving company (gvkey2)
final_data <- merged_data %>%
  left_join(
    filtered_companydata,
    by = c("gvkey2" = "gvkey", "year" = "fyear"),
    suffix = c("", "_receiving"),
    relationship = "many-to-many"
  ) %>%
  rename_with(
    ~paste0(., "_receiving"),
    c("conm", "curcd", "at", "emp", "xrd", "sich")
  )



final_data <- final_data %>%
  mutate(
    gvkey1 = as.character(gvkey1),
    gvkey2 = as.character(gvkey2)
  )

# First join for the starting company (gvkey1) - for "sic"
merged_data <- final_data %>%
  left_join(
    companydata %>% select(gvkey, fyear, sic),  # Only keep the columns needed for the join
    by = c("gvkey1" = "gvkey", "year" = "fyear"),
    suffix = c("", "_starting"),
    relationship = "many-to-many"
  ) %>%
  rename(
    sic_starting = sic
  )

# Second join for the receiving company (gvkey2) - for "sic"
final_data <- merged_data %>%
  left_join(
    companydata %>% select(gvkey, fyear, sic),  # Only keep the columns needed for the join
    by = c("gvkey2" = "gvkey", "year" = "fyear"),
    suffix = c("", "_receiving"),
    relationship = "many-to-many"
  ) %>%
  rename(
    sic_receiving = sic
  )


#write.csv(final_data, "fin.csv", row.names = FALSE)




final_data <- final_data[!is.na(final_data$xrd_starting), ]
final_data <- final_data[!is.na(final_data$at_starting), ]
final_data <- final_data[!is.na(final_data$emp_starting), ]
final_data <- final_data[!is.na(final_data$sich_starting), ]

final_data <- final_data[!is.na(final_data$at_receiving), ]
final_data <- final_data[!is.na(final_data$emp_receiving), ]
final_data <- final_data[!is.na(final_data$sich_receiving), ]
final_data <- final_data[!is.na(final_data$xrd_receiving), ]


library(plm)
library(gmm)
library(dplyr)

# Calculate concentration ratios for starting and receiving firms
final_data <- final_data %>%
  group_by(sic_starting) %>%
  mutate(total_assets_starting = sum(at_starting),
         concentration_starting = at_starting / total_assets_starting) %>%
  ungroup() %>%
  group_by(sic_receiving) %>%
  mutate(total_assets_receiving = sum(at_receiving),
         concentration_receiving = at_receiving / total_assets_receiving) %>%
  ungroup()

# Create interaction term
final_data$tech_prod_interaction <- final_data$tech_similarity * final_data$product_similarity

# Prepare the panel data structure
# Using both firm identifiers and year to create the panel
pdata <- pdata.frame(final_data, 
                     index = c("gvkey1", "gvkey2", "year"))

###########REGRESSION

# Create sector variables
# 3-digit sector variables
pdata$sector3_starting <- substr(as.character(pdata$sic_starting), 1, 3)
pdata$sector3_receiving <- substr(as.character(pdata$sic_receiving), 1, 3)

# 4-digit sector variables
pdata$sector4_starting <- substr(as.character(pdata$sic_starting), 1, 4)
pdata$sector4_receiving <- substr(as.character(pdata$sic_receiving), 1, 4)

# Convert sectors to factors
pdata$sector3_starting <- as.factor(pdata$sector3_starting)
pdata$sector3_receiving <- as.factor(pdata$sector3_receiving)
pdata$sector4_starting <- as.factor(pdata$sector4_starting)
pdata$sector4_receiving <- as.factor(pdata$sector4_receiving)

# Ensure year is a factor for fixed effects
pdata$year <- as.factor(pdata$year)



# Product Similarity Matching Script
# This script matches pairs from df with annual product similarity files
# rm(list = setdiff(ls(), c("df", "final_matched")))
# df <- df[, -ncol(df)]


library(arrow)

# Create a connection to the parquet file
ds <- open_dataset("/home/dariomarino/Thesis/pairwise_network_final.parquet")

# Get all column names first
all_columns <- names(ds)

# Filter out columns that start with "patent"
columns_to_keep <- all_columns[!grepl("^patent", all_columns, ignore.case = TRUE)]

# Load only the columns that don't start with "patent"
df <- ds %>%
  select(all_of(columns_to_keep)) %>%
  collect()

# Check the result
dim(df)
head(df)



# Memory-Efficient Product Similarity Matching Script
# Avoids duplicating the 40GB df across cores

library(data.table)
library(parallel)

# Convert to data.table for speed
setDT(df)

# Get unique years to process
years_to_process <- unique(df$fyear_start)
years_to_process <- years_to_process[years_to_process >= 1988 & years_to_process <= 2023]
years_to_process <- years_to_process[!is.na(years_to_process)]

cat("Processing years:", paste(years_to_process, collapse=", "), "\n")
cat("Total rows in df:", nrow(df), "\n")

# Function to process a single year - takes only the year subset as input
process_year_subset <- function(year_data, year) {
  cat("Processing year", year, "with", nrow(year_data), "pairs...\n")
  
  # Construct file path
  file_path <- paste0("/home/dariomarino/Thesis/ETNIC/tnicall", year, ".txt")
  
  # Check if file exists
  if(!file.exists(file_path)) {
    cat("Warning: File", file_path, "does not exist. Skipping.\n")
    return(NULL)
  }
  
  # Load the year file as data.table
  yearly_data <- fread(file_path, header=TRUE, sep="\t")
  
  # Create lookup table with both orders
  # Order 1: gvkey1-gvkey2
  lookup1 <- yearly_data[, .(gvkey1, gvkey2, score)]
  setkey(lookup1, gvkey1, gvkey2)
  
  # Order 2: gvkey2-gvkey1 (reversed)
  lookup2 <- yearly_data[, .(gvkey1 = gvkey2, gvkey2 = gvkey1, score)]
  setkey(lookup2, gvkey1, gvkey2)
  
  # Combine both lookup tables
  lookup_combined <- rbind(lookup1, lookup2)
  setkey(lookup_combined, gvkey1, gvkey2)
  
  # Remove duplicates (in case a pair appears in both orders in original data)
  lookup_combined <- unique(lookup_combined)
  
  # Prepare year_data for matching
  setkey(year_data, gvkey_start, gvkey_recv)
  
  # Perform the lookup join
  matched <- lookup_combined[year_data, on = .(gvkey1 = gvkey_start, gvkey2 = gvkey_recv)]
  
  # Return only the matched rows with ALL original columns plus product_similarity
  result <- matched[!is.na(score)]
  
  # Add the product_similarity column and clean up only the extra join columns
  result[, product_similarity := score]
  result[, c("gvkey1", "gvkey2", "score") := NULL]  # Remove only the extra join helper columns
  
  cat("Matched", nrow(result), "pairs for year", year, "\n")
  
  # Clean up
  rm(yearly_data, lookup1, lookup2, lookup_combined, matched)
  gc()
  
  return(result)
}

# Pre-split df by year to avoid memory duplication
year_subsets <- split(df, by = "fyear_start", keep.by = TRUE)

# Filter to only years we have files for
year_subsets <- year_subsets[names(year_subsets) %in% as.character(years_to_process)]

cat("Found", length(year_subsets), "years to process\n")

# Process years in smaller parallel batches to avoid memory issues
batch_size <- 10  # Process 10 years at a time
year_names <- names(year_subsets)
year_batches <- split(year_names, ceiling(seq_along(year_names) / batch_size))

cat("Processing", length(year_names), "years in", length(year_batches), "batches of", batch_size, "\n")

all_results <- list()

for(batch_idx in seq_along(year_batches)) {
  cat("Processing batch", batch_idx, "of", length(year_batches), "\n")
  
  batch_years <- year_batches[[batch_idx]]
  
  # Set up parallel processing for this batch
  n_cores_batch <- min(10, length(batch_years))  # Use fewer cores per batch
  cl <- makeCluster(n_cores_batch)
  clusterEvalQ(cl, {
    library(data.table)
  })
  
  # Export the function to all workers
  clusterExport(cl, "process_year_subset")
  
  # Create process_args for this batch only
  batch_process_args <- list()
  for(year_name in batch_years) {
    year_num <- as.numeric(year_name)
    batch_process_args[[year_name]] <- list(year_data = year_subsets[[year_name]], year = year_num)
  }
  
  # Process this batch in parallel
  batch_results <- parLapply(cl, batch_process_args, function(args) {
    process_year_subset(args$year_data, args$year)
  })
  
  # Stop cluster for this batch
  stopCluster(cl)
  
  # Add batch results to overall results
  batch_results <- batch_results[!sapply(batch_results, is.null)]
  if(length(batch_results) > 0) {
    all_results <- c(all_results, batch_results)
  }
  
  # Clean up batch data
  rm(batch_process_args, batch_results)
  gc()
  
  cat("Completed batch", batch_idx, "\n")
}

# Filter out NULL results
all_results <- all_results[!sapply(all_results, is.null)]

# Combine all results
if(length(all_results) > 0) {
  final_matched <- rbindlist(all_results)
  
  cat("Total matched pairs:", nrow(final_matched), "\n")
  
  # Save the matched dataset
  fwrite(final_matched, "/home/dariomarino/Thesis/pair_product.csv")
  cat("Saved matched dataset to /home/dariomarino/Thesis/pair_product.csv\n")
  
  # Save the pairs dataset (just the three key columns)
  pairs_df <- final_matched[, .(gvkey_start, gvkey_recv, fyear_start)]
  fwrite(pairs_df, "/home/dariomarino/Thesis/pairs.csv")
  cat("Saved pairs dataset to /home/dariomarino/Thesis/pairs.csv\n")
  
  cat("Processing complete!\n")
  cat("Final dataset has", nrow(final_matched), "rows\n")
  
} else {
  cat("No matches found across all years!\n")
}

# Clean up
rm(year_subsets, process_args, all_results)
gc()




#### now python code server to get back technology similarity, 
#here we match it back and finally start fixest

### this is an interaction regression, but is it on the paper?


library(plm)
library(gmm)
library(dplyr)

# Calculate concentration ratios for starting and receiving firms
final_data <- final_data %>%
  group_by(sic_starting) %>%
  mutate(total_assets_starting = sum(at_starting),
         concentration_starting = at_starting / total_assets_starting) %>%
  ungroup() %>%
  group_by(sic_receiving) %>%
  mutate(total_assets_receiving = sum(at_receiving),
         concentration_receiving = at_receiving / total_assets_receiving) %>%
  ungroup()

# Create interaction term
final_data$tech_prod_interaction <- final_data$tech_similarity * final_data$product_similarity

# Prepare the panel data structure
pdata <- pdata.frame(final_data, index = c("gvkey1", "gvkey2", "year"))


## now we have to add sales which we didn't have before
## you should just download the compustat data with sales included
## but i put it here for completeness


######To make final data like pair product and run the following:

# Load required libraries
library(dplyr)
library(readr)

# Read the datasets
final_data <- read_csv("/home/dariomarino/Thesis/final_data.csv")
salescompustat <- read_csv("/home/dariomarino/Thesis/salescompustat.csv")

# Step 1: Match gvkey1 and year with gvkey and year from datadate in salescompustat
# Extract year from datadate and match on both gvkey and year for unique matching
final_data <- final_data %>%
  left_join(salescompustat %>% 
              mutate(gvkey = as.integer(gvkey),
                     year = as.integer(format(datadate, "%Y"))) %>%
              select(gvkey, year, ebitda, sale), 
            by = c("gvkey1" = "gvkey", "year" = "year")) %>%
  rename(ebitda_start = ebitda,
         sale_start = sale)

# Step 2: Rename columns as requested
final_data <- final_data %>%
  rename(
    gvkey_start = gvkey1,
    gvkey_recv = gvkey2,
    fyear_start = year,
    technology_similarity = tech_similarity
  )

# Step 3: Remove unwanted columns
final_data <- final_data %>%
  select(-curcd_starting, -curcd_receiving, -sich_receiving, -sich_starting, -conm1, -conm2)

# Step 4: Rename *starting to *start and *receiving to *recv
final_data <- final_data %>%
  rename(
    conm_start = conm_starting,
    at_start = at_starting,
    emp_start = emp_starting,
    xrd_start = xrd_starting,
    sic_start = sic_starting,
    conm_recv = conm_receiving,
    at_recv = at_receiving,
    emp_recv = emp_receiving,
    xrd_recv = xrd_receiving,
    sic_recv = sic_receiving
  )

# Step 5: Calculate Herfindahl Index for concentration
# First, calculate market shares by sector (sic_start)
herfindahl_data <- final_data %>%
  group_by(sic_start) %>%
  summarise(
    total_at = sum(at_start, na.rm = TRUE),
    .groups = 'drop'
  )

# Calculate individual firm shares and Herfindahl index
concentration_by_sector <- final_data %>%
  left_join(herfindahl_data, by = "sic_start") %>%
  mutate(market_share = at_start / total_at) %>%
  group_by(sic_start) %>%
  summarise(
    concentration = sum(market_share^2, na.rm = TRUE),
    .groups = 'drop'
  )

# Add concentration_start based on sic_start
final_data <- final_data %>%
  left_join(concentration_by_sector %>% rename(concentration_start = concentration), 
            by = "sic_start")

# Add concentration_recv based on sic_recv
final_data <- final_data %>%
  left_join(concentration_by_sector %>% rename(concentration_recv = concentration), 
            by = c("sic_recv" = "sic_start"))

# Step 6: Create fyear_recv (assuming it should match fyear_start)
final_data <- final_data %>%
  mutate(fyear_recv = fyear_start)

# Step 7: Final column selection and ordering
pair_product <- final_data %>%
  select(
    gvkey_start, fyear_start, conm_start, at_start, emp_start, xrd_start, sic_start,
    gvkey_recv, fyear_recv, conm_recv, at_recv, emp_recv, xrd_recv, sic_recv,
    concentration_start, concentration_recv, product_similarity, technology_similarity,
    sale_start, ebitda_start
  )

# Clean up - remove intermediate objects, keep only pair_product
rm(final_data, salescompustat, herfindahl_data, concentration_by_sector)

# Display summary of the transformed dataset
cat("Dataset transformation completed!\n")
cat("Final dataset dimensions:", dim(pair_product)[1], "rows,", dim(pair_product)[2], "columns\n")
cat("\nFinal column names:\n")
print(names(pair_product))

# Display first few rows
cat("\nFirst 5 rows of the transformed dataset:\n")
print(head(pair_product, 5))



###### To give to final data its sales

pair_product <- read.csv("/mnt/ide0/home/dariomarino/Thesis/pair_product.csv")

library(dplyr)
library(readr)
library(lubridate)

# Load the salescompustat dataset
salescompustat <- read_csv("/home/dariomarino/Thesis/salescompustat.csv")

# Drop specified columns from pair_product (only if they exist)
columns_to_drop <- c("exchg_start", "cik_start", "sich_start", "exchg_recv", "cik_recv", "sich_recv")
existing_columns_to_drop <- intersect(columns_to_drop, names(pair_product))

if(length(existing_columns_to_drop) > 0) {
  cat("Dropping columns:", paste(existing_columns_to_drop, collapse = ", "), "\n")
  pair_product <- pair_product %>%
    select(-all_of(existing_columns_to_drop))
} else {
  cat("None of the specified columns to drop were found in the dataset.\n")
}

# SOLUTION: Standardize gvkey formats
# The issue is that salescompustat has gvkeys like "001000" while pair_product has "1000"
# We need to create a standardized version for matching

# Process salescompustat: extract year and create both gvkey formats
salescompustat_processed <- salescompustat %>%
  mutate(
    fyear = year(datadate),
    gvkey_original = gvkey,
    # Create both formats: keep original and create numeric version
    gvkey_numeric = as.character(as.numeric(gvkey))  # This removes leading zeros
  ) %>%
  select(gvkey_original, gvkey_numeric, fyear, sale, ebitda)

# Create two versions of salescompustat for matching - one for each gvkey format
# Version 1: Keep original format (for gvkeys that already match)
sales_original <- salescompustat_processed %>%
  select(gvkey = gvkey_original, fyear, sale, ebitda)

# Version 2: Use numeric format (for gvkeys that need leading zeros removed)
sales_numeric <- salescompustat_processed %>%
  select(gvkey = gvkey_numeric, fyear, sale, ebitda) %>%
  filter(!is.na(gvkey))  # Remove any NAs created by conversion

# Combine both versions, giving priority to exact matches
salescompustat_combined <- bind_rows(sales_original, sales_numeric) %>%
  # Remove duplicates, keeping the first occurrence (original format has priority)
  distinct(gvkey, fyear, .keep_all = TRUE)

cat("=== DATA PREPARATION SUMMARY ===\n")
cat("Original salescompustat rows:", nrow(salescompustat), "\n")
cat("Combined salescompustat rows:", nrow(salescompustat_combined), "\n")
cat("Unique gvkey-fyear combinations:", nrow(distinct(salescompustat_combined, gvkey, fyear)), "\n\n")

# Check the overlap now
cat("=== OVERLAP CHECK AFTER STANDARDIZATION ===\n")
common_start_new <- intersect(unique(pair_product$gvkey_start), unique(salescompustat_combined$gvkey))
common_recv_new <- intersect(unique(pair_product$gvkey_recv), unique(salescompustat_combined$gvkey))

cat("Common gvkeys with gvkey_start:", length(common_start_new), "\n")
cat("Common gvkeys with gvkey_recv:", length(common_recv_new), "\n")
cat("Sample matches:", paste(head(common_start_new, 10), collapse = ", "), "\n\n")

# Now perform the merges
cat("=== PERFORMING MERGES ===\n")

# First merge: gvkey_start + fyear_start
sales_for_start <- salescompustat_combined %>%
  rename(gvkey_start = gvkey,
         fyear_start = fyear,
         sale_start = sale,
         ebitda_start = ebitda)

pair_product_merged <- pair_product %>%
  left_join(sales_for_start, by = c("gvkey_start", "fyear_start"))

cat("After first merge - rows with sale_start data:", 
    sum(!is.na(pair_product_merged$sale_start)), "out of", nrow(pair_product_merged), "\n")

# Second merge: gvkey_recv + fyear_recv
sales_for_recv <- salescompustat_combined %>%
  rename(gvkey_recv = gvkey,
         fyear_recv = fyear,
         sale_recv = sale,
         ebitda_recv = ebitda)

pair_product_final <- pair_product_merged %>%
  left_join(sales_for_recv, by = c("gvkey_recv", "fyear_recv"))

cat("After second merge - rows with sale_recv data:", 
    sum(!is.na(pair_product_final$sale_recv)), "out of", nrow(pair_product_final), "\n\n")

# FINAL RESULTS
cat("=== FINAL RESULTS ===\n")
cat("Original pair_product dimensions:", dim(pair_product)[1], "rows x", dim(pair_product)[2], "columns\n")
cat("Final merged dataset dimensions:", dim(pair_product_final)[1], "rows x", dim(pair_product_final)[2], "columns\n\n")

cat("Columns dropped: exchg_start, cik_start, sich_start, exchg_recv, cik_recv, sich_recv\n")
cat("New columns added: sale_start, ebitda_start, sale_recv, ebitda_recv\n\n")

# Check for missing values in the new columns
cat("Missing values in new columns:\n")
cat("sale_start:", sum(is.na(pair_product_final$sale_start)), 
    "(", round(sum(is.na(pair_product_final$sale_start))/nrow(pair_product_final)*100, 1), "%)\n")
cat("ebitda_start:", sum(is.na(pair_product_final$ebitda_start)), 
    "(", round(sum(is.na(pair_product_final$ebitda_start))/nrow(pair_product_final)*100, 1), "%)\n")
cat("sale_recv:", sum(is.na(pair_product_final$sale_recv)), 
    "(", round(sum(is.na(pair_product_final$sale_recv))/nrow(pair_product_final)*100, 1), "%)\n")
cat("ebitda_recv:", sum(is.na(pair_product_final$ebitda_recv)), 
    "(", round(sum(is.na(pair_product_final$ebitda_recv))/nrow(pair_product_final)*100, 1), "%)\n\n")

# Check match rates
total_rows <- nrow(pair_product_final)
cat("Match rates:\n")
cat("gvkey_start + fyear_start matches:", 
    round((total_rows - sum(is.na(pair_product_final$sale_start))) / total_rows * 100, 2), "%\n")
cat("gvkey_recv + fyear_recv matches:", 
    round((total_rows - sum(is.na(pair_product_final$sale_recv))) / total_rows * 100, 2), "%\n\n")

# Show some examples of successful matches
cat("Sample of successful matches (first 5 rows with data):\n")
successful_matches <- pair_product_final %>%
  filter(!is.na(sale_start) | !is.na(sale_recv)) %>%
  select(gvkey_start, fyear_start, sale_start, ebitda_start, 
         gvkey_recv, fyear_recv, sale_recv, ebitda_recv) %>%
  head(5)

if(nrow(successful_matches) > 0) {
  print(successful_matches)
} else {
  cat("No successful matches found - there may be a year mismatch issue.\n")
  
  # Additional debugging for year ranges
  cat("\nYear range analysis:\n")
  cat("pair_product fyear_start range:", min(pair_product$fyear_start), "to", max(pair_product$fyear_start), "\n")
  cat("pair_product fyear_recv range:", min(pair_product$fyear_recv), "to", max(pair_product$fyear_recv), "\n")  
  cat("salescompustat fyear range:", min(salescompustat_combined$fyear), "to", max(salescompustat_combined$fyear), "\n")
}

# Save the merged dataset
write_csv(pair_product_final, "pair_product_merged.csv")
cat("\nDataset saved as 'pair_product_merged.csv'\n")

# Final verification: show some examples where we know there should be matches
cat("\n=== VERIFICATION EXAMPLES ===\n")
cat("Checking a few specific gvkey combinations that should match:\n")

# Pick a common gvkey and see if it has data
if(length(common_start_new) > 0) {
  test_gvkey <- common_start_new[1]
  
  # Check if this gvkey exists in salescompustat for relevant years
  test_sales_data <- salescompustat_combined %>%
    filter(gvkey == test_gvkey) %>%
    arrange(fyear)
  
  cat("Test gvkey:", test_gvkey, "\n")
  cat("Available years in salescompustat:", paste(test_sales_data$fyear, collapse = ", "), "\n")
  
  # Check if this gvkey appears in pair_product
  test_pair_data <- pair_product %>%
    filter(gvkey_start == test_gvkey | gvkey_recv == test_gvkey) %>%
    select(gvkey_start, fyear_start, gvkey_recv, fyear_recv) %>%
    head(3)
  
  if(nrow(test_pair_data) > 0) {
    cat("This gvkey appears in pair_product:\n")
    print(test_pair_data)
  }
}



