# Define the directory where your RDS files are saved
rds_dir <- "ICML_real_datasets"

# Get a list of all RDS files (with full paths to read them)
file_paths <- list.files(path = rds_dir, pattern = "\\.rds$", full.names = TRUE)

# Print a clean header for the console
cat(sprintf("%-50s | %-8s | %-8s | %-12s\n", "Dataset Name", "Rows", "Columns", "Missing (%)"))
cat(strrep("-", 88), "\n")

# Initialize an empty list if you want to save these stats later
# summary_stats <- list()

# Loop through each file and compute statistics
for (file_path in file_paths) {
  
  # Extract just the file name for display
  dataset_name <- tools::file_path_sans_ext(basename(file_path))
  
  # Read the dataset
  data <- readRDS(file_path)
  
  # Calculate dimensions
  n_rows <- nrow(data)
  n_cols <- ncol(data)
  
  # Calculate missingness proportion
  total_cells <- n_rows * n_cols
  total_nas <- sum(is.na(data))
  pct_missing <- (total_nas / total_cells) * 100
  
  # Truncate dataset name if it's exceptionally long to keep the table aligned
  short_name <- substr(dataset_name, 1, 50)
  
  # Print the formatted row
  cat(sprintf("%-50s | %8d | %8d | %11.2f%%\n", 
              short_name, n_rows, n_cols, pct_missing))
  
  # Optional: Save to a list to create a data.frame later
  # summary_stats[[dataset_name]] <- data.frame(
  #   Dataset = dataset_name, Rows = n_rows, Columns = n_cols, Missing_Pct = pct_missing
  # )
}
