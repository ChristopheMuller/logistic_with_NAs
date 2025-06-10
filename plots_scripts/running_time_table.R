
library(readr)
library(dplyr)
library(tidyr)


exp <- "SimA"
n_train_selected <- c(100,500,1000,5000,10000,50000)
methods_selected <- NULL

df <- read.csv(file.path("data", exp, "simulation.csv")) %>% 
  select(method, n_train, running_time_train, running_time_pred) %>% 
  filter(n_train %in% n_train_selected)

if (!is.null(methods_selected)){
  df <- df %>% filter(method %in% methods_selected)
}
if (!is.null(methods_selected)){
  df <- df %>% filter(method %in% methods_selected)
}

train_time_agg <- df %>%
  group_by(method, n_train) %>%
  summarise(mean_running_time_train = mean(running_time_train, na.rm = TRUE)) %>%
  pivot_wider(names_from = n_train, values_from = mean_running_time_train, names_prefix = "")

pred_time_agg <- df %>%
  group_by(method) %>%
  summarise(mean_running_time_pred = mean(running_time_pred, na.rm = TRUE))

combined_agg_df <- left_join(train_time_agg, pred_time_agg, by = "method")


generate_latex_table <- function(data_frame, caption_text, label_text) {
  column_names <- colnames(data_frame)
  
  time_columns <- column_names[!column_names %in% c("method", "mean_running_time_pred")]
  
  num_train_cols <- length(time_columns)
  
  # Format table header
  header <- paste0(
    "\\begin{table}[htbp]\n",
    "\\centering\n",
    "\\begin{tabular}{|l|", paste(rep("c", num_train_cols), collapse = "|"), "|c|}\n",
    "\\toprule\n",
    "\\textbf{Procedure} & \\multicolumn{", num_train_cols, "}{c|}{\\textbf{Training}} & \\multicolumn{1}{c|}{\\textbf{Prediction}} \\\\\n",
    "\\cmidrule(lr){2-", num_train_cols + 1, "} \\cmidrule(lr){", num_train_cols + 2, "-", num_train_cols + 2, "}\n",
    "& ", paste(time_columns, collapse = " & "), " & 15000 \\\\\n",
    "\\midrule\n"
  )
  
  # Format table rows
  rows <- apply(data_frame, 1, function(row) {
    method <- row["method"]
    train_times <- as.numeric(row[time_columns])
    formatted_train_times <- paste0(sprintf("%.3f s.", train_times), collapse = " & ")
    
    pred_time <- as.numeric(row["mean_running_time_pred"])
    if (!is.na(pred_time)){
      formatted_pred_time <- sprintf("%.3f s.", pred_time)
    } else {
      formatted_pred_time <- "N/A"
    }
    
    # Ensure each row ends with a LaTeX line break (\\\\) followed by a true newline (\n)
    paste0("\\texttt{", method, "} & ", formatted_train_times, " & ", formatted_pred_time, " \\\\\n")
  })
  
  # Format table footer
  footer <- paste0(
    "\\bottomrule\n",
    "\\end{tabular}\n",
    "\\caption{", caption_text, "}\n",
    "\\label{", label_text, "}\n",
    "\\end{table}\n"
  )
  
  # Combine header, rows, and footer
  paste0(header, paste(rows, collapse = ""), footer)
}

# Generate the LaTeX table
latex_output <- generate_latex_table(
  combined_agg_df,
  "Average training and prediction time of the procedures for different training sample sizes, for the experiment described in \ref{sec:methodo_SimA}.",
  "tab:runtimeSimA"
)

# Print the LaTeX output
cat(latex_output)

# Save the R script to a file
# Make sure the directory exists or create it
dir.create("plots_scripts/tables", recursive = TRUE, showWarnings = FALSE)
writeLines(latex_output, "plots_scripts/tables/runtime_table.tex")
