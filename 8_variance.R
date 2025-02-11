# Load required libraries
library(dplyr)
library(ggplot2)
library(gridExtra)
library(effectsize)
library(tidyr)

# Read and prepare data
data <- read.csv("data/ExpA/simulation_set_up.csv")
metric <- "angular_error"
data <- data[c("method", "n_train", "rep", "prop1", "corr", "prcNA", metric)]
methods_used <- c("SAEM", "Mean.IMP", "MICE.IMP", "05.IMP", "Mean.IMP.M")
data <- data[data$method %in% methods_used,]

# Convert to factors
data <- data %>%
  mutate(across(c(method, corr, n_train, rep, prcNA, prop1), factor))

# 1. Enhanced visualization of distributions
plot_distributions <- function(data) {
  ggplot(data, aes(x = angular_error, fill = method)) +
    geom_density(alpha = 0.5) +
    facet_grid(corr ~ prcNA) +
    theme_minimal() +
    labs(title = "Distribution of Angular Error by Method",
         subtitle = "Faceted by Correlation and Missing Data Percentage",
         x = "Angular Error",
         y = "Density") +
    theme(legend.position = "bottom")
}

# 2. Box plots with interaction
plot_interactions <- function(data) {
  ggplot(data, aes(x = interaction(corr, prcNA), y = angular_error, fill = method)) +
    geom_boxplot() +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = "Angular Error by Method and Conditions",
         x = "Correlation x Missing Data %",
         y = "Angular Error")
}

# 3. Heat map of mean angular errors
plot_heatmap <- function(data) {
  data %>%
    group_by(method, corr, prcNA) %>%
    summarize(mean_error = mean(angular_error), .groups = 'drop') %>%
    ggplot(aes(x = corr, y = prcNA, fill = mean_error)) +
    geom_tile() +
    facet_wrap(~method) +
    scale_fill_viridis_c() +
    theme_minimal() +
    labs(title = "Mean Angular Error Across Conditions",
         fill = "Mean Error")
}

# 4. Variance decomposition visualization
variance_decomposition <- function(fit) {
  # Calculate eta squared for all terms
  eta <- eta_squared(fit, partial = FALSE)

  
  # Create plot
  ggplot(data.frame(term = eta$Parameter, 
                    eta_sq = eta$Eta2), 
         aes(x = reorder(term, eta_sq), y = eta_sq)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    theme_minimal() +
    labs(title = "Variance Decomposition",
         x = "Model Terms",
         y = "Eta Squared")
}

# 5. Diagnostic plots using ggplot2
diagnostic_plots <- function(fit) {
  # Residuals vs Fitted
  p1 <- ggplot(data.frame(fitted = fitted(fit),
                          residuals = resid(fit)),
               aes(x = fitted, y = residuals)) +
    geom_point() +
    geom_smooth(method = "loess", se = FALSE) +
    theme_minimal() +
    labs(title = "Residuals vs Fitted")
  
  # Normal Q-Q
  p2 <- ggplot(data.frame(theoretical = qnorm(ppoints(length(resid(fit)))),
                          sample = sort(scale(resid(fit)))),
               aes(x = theoretical, y = sample)) +
    geom_point() +
    geom_abline() +
    theme_minimal() +
    labs(title = "Normal Q-Q Plot")
  
  grid.arrange(p1, p2, ncol = 2)
}

# Fit model
options(contrasts = c("contr.sum", "contr.poly"))
fit <- aov(angular_error ~ corr * prop1 * n_train * method * prcNA, data = data)

# Generate all plots
p1 <- plot_distributions(data)
p2 <- plot_interactions(data)
p3 <- plot_heatmap(data)
p4 <- variance_decomposition(fit)

# Display plots
print(p1)
print(p2)
print(p3)
print(p4)
diagnostic_plots(fit)

# Summary statistics
summary(fit)

# Effect size analysis
eta_squared(fit, partial = TRUE) %>% 
  arrange(desc(Eta2_partial))
