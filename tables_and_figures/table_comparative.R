library(readr)
library(dplyr)
library(tidyr)
library(tools)

build_presentation_table <- function(combined_scores, methods_map, sim_names, sizes, metric_name) {
  scores_data <- combined_scores %>%
    filter(metric == metric_name)
  
  times_data <- combined_scores %>%
    filter(metric == "running_time_train") %>%
    group_by(method, n_train) %>%
    summarise(
      sd_score = sd(mean_score, na.rm = TRUE),
      mean_score = mean(mean_score, na.rm = TRUE),
      .groups = 'drop'
    ) %>%
    mutate(metric = "Time")
  
  scores_wide <- scores_data %>%
    pivot_wider(id_cols = method, 
                names_from = c(exp_name, n_train), 
                names_sep = "_", 
                values_from = c(mean_score, sd_score))
  
  times_wide <- times_data %>%
    pivot_wider(id_cols = method, 
                names_from = c(metric, n_train), 
                names_sep = "_", 
                values_from = c(mean_score, sd_score))
  
  table_df <- full_join(scores_wide, times_wide, by = "method")
  
  mean_col_order <- c()
  sd_col_order <- c()
  for (n_size in sizes) {
    for (sim_name in sim_names) {
      mean_col_order <- c(mean_col_order, paste("mean_score", sim_name, n_size, sep = "_"))
      sd_col_order <- c(sd_col_order, paste("sd_score", sim_name, n_size, sep = "_"))
    }
    mean_col_order <- c(mean_col_order, paste("mean_score_Time", n_size, sep = "_"))
    sd_col_order <- c(sd_col_order, paste("sd_score_Time", n_size, sep = "_"))
  }
  
  col_order <- c("method", unlist(mapply(c, mean_col_order, sd_col_order, SIMPLIFY = FALSE)))
  
  missing_cols <- setdiff(col_order, names(table_df))
  if (length(missing_cols) > 0) {
    for (col in missing_cols) {
      table_df[[col]] <- NA_real_
    }
  }
  
  # Ensure dataframe only contains columns that are actually needed and in the correct order
  table_df <- table_df[, intersect(col_order, names(table_df))]
  
  table_df <- table_df %>%
    slice(match(names(methods_map), method))
  
  return(table_df)
}

apply_latex_formatting <- function(df, methods_map, digits_vector = NULL) {
  formatted_df <- data.frame(method = paste0("{\\tiny \\texttt{", unlist(methods_map[df$method]), "}}"))
  
  mean_cols <- names(df)[grepl("^mean_score", names(df))]
  
  for (i in seq_along(mean_cols)) {
    mean_col_name <- mean_cols[i]
    sd_col_name <- sub("mean_score", "sd_score", mean_col_name)
    
    if (!sd_col_name %in% names(df)) next
    
    mean_values <- as.numeric(df[[mean_col_name]])
    sd_values <- as.numeric(df[[sd_col_name]])
    formatted_values <- rep("---", nrow(df))
    non_na_indices <- which(!is.na(mean_values))
    
    if (length(non_na_indices) > 0) {
      # USE CUSTOM DIGITS IF PROVIDED, ELSE AUTO-DETECT
      if (!is.null(digits_vector) && i <= length(digits_vector)) {
        num_digits <- digits_vector[i]
      } else {
        min_in_column <- min(mean_values[non_na_indices], na.rm = TRUE)
        if (is.finite(min_in_column) && min_in_column < 0.01) {
          num_digits <- 3
        } else if (is.finite(min_in_column) && min_in_column < 0.1) {
          num_digits <- 2
        } else {
          num_digits <- 1
        }
      }
      
      temp_formatted_values_for_markup <- formatC(mean_values[non_na_indices], format = "f", digits = num_digits, preserve.width = "individual")
      
      min_val_index <- which.min(mean_values[non_na_indices])
      min_val <- mean_values[non_na_indices][min_val_index]
      std_of_min <- sd_values[non_na_indices][min_val_index]
      
      if (!is.na(min_val) && !is.na(std_of_min)) {
        bold_threshold <- min_val + std_of_min
        underline_threshold <- min_val + 2 * std_of_min
        scores_to_format <- mean_values[non_na_indices]
        
        bold_indices <- which(scores_to_format <= bold_threshold)
        underline_indices <- which(scores_to_format > bold_threshold & scores_to_format <= underline_threshold)
        
        if(length(bold_indices) > 0) temp_formatted_values_for_markup[bold_indices] <- paste0("\\textbf{", temp_formatted_values_for_markup[bold_indices], "}")
        if(length(underline_indices) > 0) temp_formatted_values_for_markup[underline_indices] <- paste0("\\underline{", temp_formatted_values_for_markup[underline_indices], "}")
      }
      
      over_1000_indices <- which(mean_values[non_na_indices] > 1000)
      if (length(over_1000_indices) > 0) {
        temp_formatted_values_for_markup[over_1000_indices] <- "+1000"
      }
      
      formatted_values[non_na_indices] <- temp_formatted_values_for_markup
    }
    
    output_col_name <- sub("mean_score_", "", mean_col_name)
    formatted_df[[output_col_name]] <- formatted_values
  }
  return(formatted_df)
}

# ... [generate_latex_from_df remains the same as your previous version] ...

generate_metric_table <- function(metric_name, bayes.diff=TRUE, multiplier=1, digits_vector = NULL) {
  Sims <- c("SimMCAR", "SimMAR", "SimMNAR", "SimNL", "SimNLMNAR")
  Sims.names <- c("MCAR", "MAR", "MNAR", "NL", "NLMNAR") 
  train.sizes <- c(100, 50000)
  
  method_map_csv_to_latex <- list(
    "CC"="CC",
    "SAEM.NoReg" = "SAEM",
    "Mean.IMP" = "Mean.IMP",
    "Mean.IMP.M" = "Mean.IMP.M",
    "MICE.1.IMP" = "MICE.1.IMP",
    "MICE.1.Y.IMP" = "MICE.1.Y.IMP",
    "MICE.1.Y.M.IMP.M" = "MICE.1.Y.M.IMP.M",
    "MICE.100.IMP" = "MICE.100.IMP",
    "MICE.100.Y.IMP" = "MICE.100.Y.IMP",
    "MICE.100.Y.M.IMP.M" = "MICE.100.Y.M.IMP.M",
    "MICE.RF.10.IMP" = "MICE.RF.10.IMP",
    "MICE.RF.10.Y.IMP" = "MICE.RF.10.Y.IMP",
    "MICE.RF.10.Y.M.IMP.M" = "MICE.RF.10.Y.M.IMP.M"
  )
  
  all_scores_data <- list()
  
  for (i in seq_along(Sims)) {
    sim_exp <- Sims[i]
    sim_name <- Sims.names[i]
    file_path <- file.path("data", sim_exp, "score_matrix.csv")
    
    if (file.exists(file_path)) {
      raw_file <- read_csv(file_path, show_col_types = FALSE)
      
      score_matrix_df <- raw_file %>%
        filter(filter == "all",
               method %in% names(method_map_csv_to_latex),
               metric == metric_name,
               bayes_adj == bayes.diff,
               n_train %in% train.sizes
        ) %>%
        mutate(score = multiplier * as.numeric(score)) %>%
        select(method, n_train, metric, score) %>%
        group_by(method, n_train, metric) %>%
        summarise(mean_score = mean(score, na.rm = TRUE), sd_score = sd(score, na.rm = TRUE), .groups = "drop") %>%
        mutate(exp_name = sim_name)
      
      running_time_matrix_df <- raw_file %>%
        filter(filter == "all",
               method %in% names(method_map_csv_to_latex),
               metric == "running_time_train",
               n_train %in% train.sizes
        ) %>%
        mutate(score = as.numeric(score)) %>%
        select(method, n_train, metric, score) %>%
        group_by(method, n_train, metric) %>%
        summarise(mean_score = mean(score, na.rm = TRUE), sd_score = sd(score, na.rm = TRUE), .groups = "drop") %>%
        mutate(exp_name = sim_name)
      
      all_scores_data[[sim_exp]] <- rbind(score_matrix_df, running_time_matrix_df)
    }
  }
  
  combined_scores_df <- bind_rows(all_scores_data)
  numeric_table <- build_presentation_table(combined_scores_df, method_map_csv_to_latex, Sims.names, train.sizes, metric_name)
  
  # PASS THE DIGITS VECTOR HERE
  formatted_table <- apply_latex_formatting(numeric_table, method_map_csv_to_latex, digits_vector = digits_vector)
  
  final_latex_code <- generate_latex_from_df(formatted_table, metric_name)
  
  output_filename <- paste0("tables_and_figures/tables/", metric_name, "_summary.tex")
  dir.create(dirname(output_filename), recursive = TRUE, showWarnings = FALSE)
  writeLines(final_latex_code, output_filename)
  cat(paste("Successfully generated table for metric '", metric_name, "'.\n", sep = ""))
}

# --- EXECUTION EXAMPLE ---
# Your table has 12 data columns: 
# (5 Sims + 1 Time) for N=100 AND (5 Sims + 1 Time) for N=50,000.
# Example: 3 decimals for scores, 1 decimal for time.
my_digits <- c(2, 2, 2, 2, 2, 3, 3, 3, 2, 3, 2, 1)

generate_metric_table("mse_error", bayes.diff=FALSE, multiplier=1, digits_vector = my_digits)