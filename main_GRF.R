# rm(list = ls(all = T))
source("./private/data_preprocessing.R")
source("GRF_func.R")
library(dplyr)
library(tidyr)
library(grf)
library(policytree)
library(fastpolicytree)
library(glue)
library(ggplot2)
library(lubridate)
library(gridExtra)


# DATA PREPARATION -------------------------------------------------------------
pre_periods = c(1:3)
post_periods = c(9, 10)

get_datasets = function(pre_periods,
                        post_periods,
                        temp_type = 1,
                        preprocess_temp = FALSE,
                        single_pre_usage = FALSE,
                        imputation = FALSE,
                        interaction = FALSE,
                        factor_to_one_hot = TRUE,
                        log_transformation = TRUE,
                        outcome_normalization = TRUE) {
  preprocess_results = preprocess(
    pre_periods = pre_periods,
    post_periods = post_periods,
    temp_type = temp_type,
    preprocess_temp = preprocess_temp,
    single_pre_usage = single_pre_usage,
    imputation = imputation,
    interaction = interaction,
    factor_to_one_hot = factor_to_one_hot,
    log_transformation = log_transformation,
    outcome_normalization = outcome_normalization
  )
  
  post_period_cols = preprocess_results$post_period_cols
  
  datasets = list()

  # T1 treatment dataset (T1 vs. Control)
  datasets$T1 = preprocess_results$data %>%
    filter(Treatment == "C" | Treatment == "T1") %>%
    mutate(
      Treatment = dplyr::recode(Treatment, "C" = 0, "T1" = 1),
      across(everything(), as.numeric)
    )
  
  # LATE takers dataset (T2 vs. Control)
  datasets$T2y = preprocess_results$data %>%
    filter(Treatment == "C" | Treatment == "T2") %>%
    mutate(
      Treatment = dplyr::recode(Treatment, "C" = 0, "T2" = 1),
      across(everything(), as.numeric)
    )
  
  # LATE non-takers dataset (T2 vs. T1)
  datasets$T2n = preprocess_results$data %>%
    filter(Treatment == "T1" | Treatment == "T2") %>%
    mutate(
      Treatment = dplyr::recode(Treatment, "T1" = 0, "T2" = 1),
      across(everything(), as.numeric)
    )
  
  # rm(preprocess_results, create_interaction_terms, predict_similar_households, preprocess)
  return(list(datasets = datasets, post_period_cols = preprocess_results$post_period_cols))
}


preprocess_results = get_datasets(
  pre_periods = pre_periods,
  post_periods = post_periods,
  temp_type = 4,
  preprocess_temp = FALSE,
  single_pre_usage = FALSE,
  imputation = FALSE,
  interaction = FALSE,
  factor_to_one_hot = TRUE,
  log_transformation = TRUE,
  outcome_normalization = TRUE
)
datasets.log = preprocess_results$datasets
post_period_cols = preprocess_results$post_period_cols

rm(preprocess_results)
# END: # DATA PREPARATION


# MAIN GRF ANALYSIS ------------------------------------------------------------
# default model
GRF_results.log = run_GRF(
  datasets.log,
  Y_var = "log_outcome_diff",
  post_periods,
  post_periods_cols,
  seed = 123
)

estimate_results = GRF_results.log$estimates

# plot estimate results
estimate_results$mu = as.numeric(estimate_results$mu)
estimate_results$se = as.numeric(estimate_results$se)
plot_coef(estimate_results)

# focus on period 10 analysis
models.log = GRF_results.log$models$period_10

rm(GRF_results.log)
# END: MAIN GRF ANALYSIS


# FEATURE IMPORTANCE ANALYSIS --------------------------------------------------
# set maximum tree depth for importance analysis
depth = 5

# T1 vs. Control model
importance.T1 = analyze_feature_importance(models.log$T1, depth = depth)
plot_feature_importance(
  importance.T1,
  depth,
  top_n = 12,
  plot_name = "T1 vs. Control ATE"
)
plot_split_pattern(
  importance.T1,
  depth,
  rank_by = "frequency",
  top_n = 12,
  plot_name = "T1 vs. Control ATE"
)

# T2 vs. Control model (LATE)
importance.T2y = analyze_feature_importance(models.log$T2y, depth)
plot_feature_importance(
  importance.T2y,
  depth,
  top_n = 12,
  plot_name = "T2 vs. Control LATE"
)
plot_split_pattern(
  importance.T2y,
  depth,
  rank_by = "frequency",
  top_n = 12,
  plot_name = "T2 vs. Control LATE"
)

# select top 10 most important variables for policy tree construction
important_vars = importance.T1$variable[1:5]
# END: # FEATURE IMPORTANCE ANALYSIS


# POLICY TREE ANALYSIS ---------------------------------------------------------
# learning simple rule-based policies
# ref: https://grf-labs.github.io/policytree/

## CONSTRUCT POLICY TREE -------------------------------------------------------
# find a shallow, but globally optimal decision tree by exhaustive search
# depth=2 ensures interpretable policies with simple decision rules
opt_tree = policy_tree(
  datasets.log$T1[important_vars],
  -double_robust_scores(models.log$T1),  # more negative treatment effect = more electricity reduction
  depth = 2
)

# construct policy tree using fastpolicytree
# It aims to do the same job as the policytree but to do so more quickly.
# ref: https://github.com/jcussens/tailoring
# opt_tree = fastpolicytree(
#   datasets.log$T1[important_vars],
#   -double_robust_scores(models.log$T1),  # more negative treatment effect = more electricity reduction
#   depth = 3
# )

# display and visualize the optimal policy tree
opt_tree.plot = plot(opt_tree, leaf.labels = c("control", "treated"))
opt_tree.plot

# export policy trees as .svg file
cat(DiagrammeRsvg::export_svg(opt_tree.plot), file = "./private/figure/policy_tree.svg")

# compare effectiveness of self-selection (T2y) vs. policy tree targeting
targeting_results = compare_targeting_strategies(datasets.log$T1, datasets.log$T2y, models.log, opt_tree)
targeting_results

# policy tree leaf node analysis
leaf_analysis.T1 = analyze_tree_leaves(datasets.log$T1, models.log$T1, opt_tree)
leaf_analysis.T1

# plot full GRF tree
# export decision trees as .svg files
# this allows detailed inspection of tree structures
# plot_grftree(c("T1", "T2_ITT", "T2y", "T2n"), models)


## TOTAL ELECTRICITY SAVINGS ---------------------------------------------------
# Build original scale datasets for calculating total electricity savings
# outcome = post_usage - pre_usage_avg (in original kWh units)

# Generate datasets without log transformation
preprocess_results = get_datasets(
  pre_periods = pre_periods,
  post_periods = post_periods,
  preprocess_temp = FALSE,
  single_pre_usage = FALSE,
  imputation = FALSE,
  interaction = FALSE,
  factor_to_one_hot = TRUE,
  log_transformation = FALSE,  # Use original scale
  outcome_normalization = TRUE
)
datasets.orig = preprocess_results$datasets
rm(preprocess_results)

# Verify that smart_id ordering matches between datasets.log and datasets.orig
# Check T1 dataset consistency
if (!all(datasets.log$T1$smart_id == datasets.orig$T1$smart_id)) {
  warning("smart_id ordering is inconsistent between log and original datasets!")
  datasets.orig$T1 = datasets.orig$T1 %>%
    arrange(match(smart_id, datasets.log$T1$smart_id))
}

# Check T2y dataset consistency  
if (!all(datasets.log$T2y$smart_id == datasets.orig$T2y$smart_id)) {
  warning("smart_id ordering is inconsistent between log and original datasets!")
  datasets.orig$T2y = datasets.orig$T2y %>%
    arrange(match(smart_id, datasets.log$T2y$smart_id))
}

# Check T2n dataset consistency  
if (!all(datasets.log$T2n$smart_id == datasets.orig$T2n$smart_id)) {
  warning("smart_id ordering is inconsistent between log and original datasets!")
  datasets.orig$T2n = datasets.orig$T2n %>%
    arrange(match(smart_id, datasets.log$T2n$smart_id))
}

# build GRF models using original scale data
GRF_results.orig = run_GRF(
  datasets.orig,
  "outcome_diff",
  post_periods,
  post_periods_cols,
  seed = 123
)

models.orig = GRF_results.orig$models$period_10
rm(GRF_results.orig)

targeting_results.savings = compare_targeting_strategies(
  T1_dat = datasets.log$T1,  # for prediction
  T2_dat = datasets.log$T2y, # for prediction
  models = models.orig,  # for calculating ATE
  tree = opt_tree
)
targeting_results.savings

# Calculate total electricity savings for each targeting strategy
# ref: https://www.moeaea.gov.tw/ecw/populace/content/ContentDesc.aspx?menu_id=26678
# 2024年電力排碳係數 = 0.474 公斤CO2e/度
carbon_emission_factor = 0.474

# ATE × number of targeted households = total electricity savings (kWh)
total_savings.tree_T1 = -targeting_results.savings[1, "ATE"] * targeting_results.savings[1, "n"]
carbon_reduction.tree_T1 = total_savings.tree_T1 * carbon_emission_factor
cat("=== POLICY TREE T1 ANALYSIS ===")
cat(glue("Total electricity savings on T1: {total_savings.tree_T1}(kWh)"))
cat(glue("Total carbon reduction: {round(carbon_reduction.tree_T1, 5)} kg CO2e\n"))

total_savings.tree_T2 = -targeting_results.savings[2, "ATE"] * targeting_results.savings[2, "n"]
carbon_reduction.tree_T2 = total_savings.tree_T2 * carbon_emission_factor
cat("=== POLICY TREE T2 ANALYSIS ===")
cat(glue("Total electricity savings on T2: {total_savings.tree_T2}(kWh)"))
cat(glue("Total carbon reduction: {round(carbon_reduction.tree_T2, 5)} kg CO2e\n"))

# rm(models.orig, targeting_results.savings, total_savings.tree_T1, total_savings.tree_T2)


## POLICY TARGETING USING OPTIMAL POLICY TREE ----------------------------------
# This analysis learns optimal policy tree using T2 data, then evaluates targeting 
# effectiveness using T1 trained model to avoid Winner's Bias
RATE_results.opt_tree = rank_average_treatment_effect(
  forest = models.log$T1,
  priorities = predict(opt_tree, datasets.log$T1[important_vars])
)
plot(RATE_results.opt_tree)

# construct a 95 % CI for the AUTOC
RATE_results.opt_tree$estimate
RATE_results.opt_tree$estimate + 1.96 * c(-1, 1) * RATE_results.opt_tree$std.err
# END: # POLICY TREE ANALYSIS


# Temperature Analysis
plot_ate_scatter(
  datasets.log$T2y$temp_mean_10,
  get_scores(models.log$T2y),
  title = "LATE vs. Temperature",
  x_lab = "Temperature (°C)",
  y_lab = "LATE",
  add_trend = "poly"
)


# DISTRIBUTIONS OF ATE ---------------------------------------------------------
# period 10
# # doubly robust scores
# p1 = plot_ate_dist(
#   get_scores(models.log$T1),
#   x_lab = "T1 ATE",
#   add_to_caption = "period 10"
# )
# 
# p2 = plot_ate_dist(
#   get_scores(models.log$T2_ITT),
#   x_lab = "T2 ITT",
#   add_to_caption = "period 10"
# )
# 
# p3 = plot_ate_dist(
#   get_scores(models.log$T2y),
#   x_lab = "LATE_takers",
#   add_to_caption = "period 10"
# )
# 
# p4 = plot_ate_dist(
#   -get_scores(models.log$T2n),
#   x_lab = "LATE_non-takers",
#   add_to_caption = "period 10"
# )

# grf model predictions
p1 = plot_ate_dist(
  predict(models.log$T1)$predictions,
  x_lab = "T1 ATE",
  add_to_caption = "period 10"
)

p2 = plot_ate_dist(
  predict(models.log$T2_ITT)$predictions,
  x_lab = "T2 ITT",
  add_to_caption = "period 10"
)

p3 = plot_ate_dist(
  predict(models.log$T2y)$predictions,
  x_lab = "LATE_takers",
  add_to_caption = "period 10"
)

p4 = plot_ate_dist(
  -predict(models.log$T2n)$predictions,
  x_lab = "LATE_non-takers",
  add_to_caption = "period 10"
)

grid.arrange(grobs = list(p1, p2, p3, p4), nrow = 2)
rm(p1, p2, p3, p4)
# END: # DISTRIBUTIONS OF ATE


# HETEROGENEITY ANALYSIS -------------------------------------------------------
# load additional variables from the raw data
dat = read_dta("./private/data/renamed_all_data.dta")

mission_vars = c("mission", "photo_1", "photo_2", "photo_3", "photo_4", "photo_5", "finish_time")
photo_mission_send = as.POSIXct("2024-09-24 12:00:00", tz = "Asia/Taipei", format = "%Y-%m-%d %H:%M:%S")
photo_mission_end = photo_mission_send + as.difftime(14, units = "days")

dat = dat %>%
  dplyr::select(all_of(c("smart_id", mission_vars))) %>%
  # filter to include only smart_ids that exist in modeling datasets
  filter(
    smart_id %in% unique(c(datasets.log$T1$smart_id, datasets.log$T2y$smart_id, datasets.log$T2n$smart_id))
  ) %>%
  distinct(smart_id, .keep_all = TRUE) %>%
  # handle missing values
  mutate(
    across(
      all_of(mission_vars[1:6]), \(x) coalesce(x, 0)  # NA -> 0
    ),
    finish_time = force_tz(as.POSIXct(finish_time, format = "%Y-%m-%d %H:%M:%S"), "Asia/Taipei"),
    finish_time = coalesce(finish_time, photo_mission_end),  # NA -> photo_mission_end
    spending_hours = as.numeric(finish_time - photo_mission_send, units = "hours")
  )

dr_scores.T2y = data.frame(
  smart_id = datasets.log$T2y$smart_id[datasets.log$T2y$T2y == 1],
  ATE = get_scores(models.log$T2y, subset = datasets.log$T2y$T2y == 1)
) %>%
  left_join(dat, by = "smart_id")

rm(dat)

# T2y (T2 vs. Control)
# remove mission == 0
plot_ate_scatter(
  dr_scores.T2y$spending_hours[dr_scores.T2y$mission != 0],
  dr_scores.T2y$ATE[dr_scores.T2y$mission != 0],
  title = "LATE_takers vs. Mission Completion Time",
  x_lab = "Hours from Mission Send",
  y_lab = "LATE",
  add_trend = "linear"
)

for (var in mission_vars[1:6]) {
  group_data = dr_scores.T2y[[var]]
  group_levels = unique(group_data)
  
  results = data.frame(
    group = character(0),
    Avg_Effect = numeric(0),
    Std_Err = numeric(0),
    CI_Lower = numeric(0),
    CI_Upper = numeric(0),
    n = integer(0)
  )
  for(g in group_levels) {
    curr_idx = which(group_data == g)
    curr_result = average_treatment_effect(models$T2y, subset = curr_idx)
    
    results = rbind(
      results,
      data.frame(
        group = g,
        Avg_Effect = curr_result[["estimate"]],
        Std_Err = curr_result[["std.err"]],
        CI_Lower = curr_result[["estimate"]] - 1.96 * curr_result[["std.err"]],
        CI_Upper = curr_result[["estimate"]] + 1.96 * curr_result[["std.err"]],
        n = length(curr_idx)
      )
    )
  }
  
  numeric_cols = c("Avg_Effect", "Std_Err", "CI_Lower", "CI_Upper")
  results[numeric_cols] = round(results[numeric_cols], 5)

  cat("\n", glue("LATE by `{var}`"), "\n")
  print(results %>% arrange(group))
}

# rm(mission_vars, photo_mission_send, photo_mission_end, dr_scores.T2y)
# END: # HETEROGENEITY ANALYSIS


# ADD T2y TO X (COVARIATES) ----------------------------------------------------
# causal_forest
X = datasets.log$T2y[get_covariate_cols(datasets.log$T2y, 10, post_period_cols)] %>%
  mutate(
    T2y = datasets.log$T2y$T2y
  )
Y = datasets.log$T2y$log_outcome_diff_10

cf_model = causal_forest(
  X = X,
  Y = Y,
  W = datasets.log$T2y$Treatment,
  mtry = floor(ncol(X) / 3),
  seed = 123
)

# depth = 5
plot_split_pattern(
  analyze_feature_importance(cf_model, depth),
  depth,
  rank_by = "frequency",
  top_n = 20,
  plot_name = "T2 vs. Control ITT (with T2y as Covariate)"
)

# regression_forest
rf_model = regression_forest(
  X = X,
  Y = Y,
  num.trees = 2000,
  mtry = floor(ncol(X) / 3),
  seed = 123
)

plot_split_pattern(
  analyze_feature_importance(rf_model, depth),
  depth,
  rank_by = "frequency",
  top_n = 20,
  plot_name = "log(daily_usage_10) (with T2y as Covariate)"
)

rm(X, Y, cf_model, rf_model)
# END: # ADD T2y TO X (COVARIATES)


# TARGETING ANALYSIS: T2 (Athey et al., 2025) ----------------------------------
# ref: Athey, S., Inoue, K., & Tsugawa, Y. (2025). Targeted Treatment Assignment Using Data from Randomized Experiments with Noncompliance. AEA Papers and Proceedings, 115, 209–214.
# focus on T2 treatment, Period 10 for targeting analysis
# Goal: Understand who benefits most from treatment and design targeting policies

## CONSTRUCT MODELS FOR TARGETING ----------------------------------------------
# prepare data for targeting
Y = datasets.log$T2y$log_outcome_diff_10
X = datasets.log$T2y[get_covariate_cols(datasets.log$T2y, 10, post_period_cols)]
W = datasets.log$T2y$T2y
Z = datasets.log$T2y$Treatment

# predict individual-level treatment effects (both LATE and ATE)
CLATE_hat = predict(models.log$T2y)$predictions

# fit causal forest for compliance
compliance_forest = causal_forest(
  X = X,
  Y = W,
  W = Z,
  num.trees = 2000,
  mtry = floor(ncol(X) / 3),
  seed = 456
)

# predict individual-level compliance probabilities
compliance_hat = predict(compliance_forest)$predictions

# fit causal forest for welfare (net benefit)
# include treatment cost to evaluate cost-effectiveness
# set the cost of delivering the treatment (can try different values)
# when cost = 0, the model estimate CATE
cost = 0.1
net_CATE_forest = causal_forest(
  X = X,
  Y = Y + cost * W,  # usage cost + implementation cost (to be minimized)
  W = Z,
  num.trees = 2000,
  mtry = floor(ncol(X) / 3),
  seed = 456
)

# predict individual-level net benefit
net_CATE_hat = predict(net_CATE_forest)$predictions


## RANK-WEIGHTED AVERAGE TREATMENT EFFECT (RATE) ANALYSIS ----------------------
# estimate RATE for compliance targeting
# goal: increase enrollment/compliance
RATE_results.compliance = rank_average_treatment_effect(
  forest = compliance_forest,
  priorities = compliance_hat
)
plot(RATE_results.compliance)

# construct a 95 % CI for the AUTOC
RATE_results.compliance$estimate + 1.96 * c(-1, 1) * RATE_results.compliance$std.err

# Treatment Delivery Decision Problem
# goal: maximize treatment effects by targeting high-effect individuals
RATE_results.clate = rank_average_treatment_effect(
  forest = models.log$T2y,
  priorities = -CLATE_hat  # priority based on predicted treatment effect: more negative CLATE = more electricity reduction
  # Alternative: priorities = compliance_hat (target based on compliance)
)
plot(RATE_results.clate)

# construct a 95 % CI for the AUTOC
# Negative AUTOC indicates that targeting successfully identifies individuals
# with better energy conservation effects
RATE_results.clate$estimate + 1.96 * c(-1, 1) * RATE_results.clate$std.err


# Eligibility Decision Problem
# goal: maximize social welfare by considering both benefits and costs
RATE_results.cate = rank_average_treatment_effect(
  forest = net_CATE_forest,
  priorities = -net_CATE_hat,  # priority based on predicted treatment effect
)
plot(RATE_results.cate)

# construct a 95 % CI for the AUTOC
RATE_results.cate$estimate + 1.96 * c(-1, 1) * RATE_results.cate$std.err
# END: # TARGETING ANALYSIS: T2 (Athey et al., 2025)


# # ASSESS MODEL STABILITY -------------------------------------------------------
# bootstrap_GRF = function(datasets,
#                          post_periods, 
#                          n_bootstrap = 1000,
#                          num_trees = 2000,
#                          centering_method = NULL) {
#   
#   print(glue("Starting GRF bootstrap analysis with {n_bootstrap} iterations..."))
#   
#   for (b in 1:n_bootstrap) {
#     if (b %% 50 == 0) {
#       print(glue("Bootstrap iteration {b}"))
#     }
#     
#     # Bootstrap sample (stratified by treatment to maintain balance)
#     # This maintains the same logic as your FRA bootstrap
#     bootstrap_datasets = lapply(
#       datasets,
#       function(dataset) {
#         sample = dataset %>%
#           group_by(Treatment) %>%
#           group_modify(~ slice_sample(.x, n = nrow(.x), replace = T)) %>%
#           ungroup()
#         return(sample)
#       }
#     )
#     
#     try({
#       # run GRF analysis on bootstrap sample
#       curr_results = run_GRF(bootstrap_datasets,
#                              post_periods = post_periods,
#                              num_trees = num_trees,
#                              centering_method = centering_method,
#                              return_models = FALSE,
#                              verbose = FALSE)
# 
#       # store results
#       if (b == 1) {
#         bootstrap_estimates = curr_results %>% select(mu)
#         rownames(bootstrap_estimates) = curr_results$treatment_type
#       }
#       else {
#         bootstrap_estimates = cbind(bootstrap_estimates, curr_results$mu)
#       }
#     }, silent = T)
#   }
#   
#   colnames(bootstrap_estimates) = 1:ncol(bootstrap_estimates)
#   return(bootstrap_estimates)
# }
# 
# calculate_ci = function(bootstrap_results,
#                         confidence_level = 0.95,
#                         original_estimates = NULL) {
#   
#   alpha = 1 - confidence_level
#   lower_quantile = alpha / 2
#   upper_quantile = 1 - alpha / 2
#   
#   ci_results = data.frame(
#     mean_estimate = numeric(0),
#     sd_estimate = numeric(0),
#     ci_lower = numeric(0),
#     ci_upper = numeric(0)
#   )
#   
#   for (row in rownames(bootstrap_results)) {
#     curr_row = as.numeric(bootstrap_results[row, ])
#     ci_results = rbind(
#       ci_results,
#       data.frame(mean_estimate = mean(curr_row),
#                  sd_estimate = sd(curr_row),
#                  ci_lower = quantile(curr_row, lower_quantile),
#                  ci_upper = quantile(curr_row, upper_quantile))
#     )
#   }
#   
#   ci_results = ci_results %>%
#     mutate(treatment_type = rownames(bootstrap_results)) %>%
#     relocate(treatment_type, .before = 1)
#   
#   # Add original estimates if provided
#   if (!is.null(original_estimates)) {
#     original_estimates = original_estimates %>%
#       rename(original = mu)
#     ci_results = ci_results %>%
#       left_join(original_estimates %>% select(-se), by = "treatment_type") %>%
#       mutate(
#         bias = mean_estimate - original,
#         coverage = original >= ci_lower & original <= ci_upper
#       )
#   }
#   
#   return(ci_results)
# }
# 
# 
# plot_bootstrap = function(bootstrap_results, 
#                               ci_results_bootstrap,
#                               original_estimates = NULL) {
#   
#   # plot distribution
#   par(mfrow = c(3, 2), mar = c(4, 4, 4, 2))
#   bootstrap_rownames = rownames(bootstrap_results)
# 
#   for (i in 1:min(3, length(bootstrap_rownames))) {
#     curr_treatment_type = bootstrap_rownames[[i]]
#     curr_row_bootstrap = as.numeric(bootstrap_results[curr_treatment_type,])
#     curr_density_bootstrap = density(curr_row_bootstrap)
# 
#     hist(
#       curr_row_bootstrap,
#       breaks = "FD",
#       main = glue("{title_prefix} Bootstrap ({curr_treatment_type})"),
#       col = "lightblue",
#       xaxs = "i",
#       yaxs = "i",
#       xlim = c(min(curr_row_bootstrap), max(curr_row_bootstrap)),
#       ylim = c(0, max(curr_density_bootstrap$y) + 10),
#       xlab = "Estimate", 
#       ylab = "Density", 
#       freq = F
#     )
#     
#     # add mean line
#     abline(
#       v = ci_results_bootstrap[curr_treatment_type, "mean_estimate"],
#       col = "red",
#       lty = 2,
#       lwd = 2
#     )
#     
#     # add normal curve
#     curve(dnorm(x, 
#                 ci_results_bootstrap[curr_treatment_type, "mean_estimate"], 
#                 ci_results_bootstrap[curr_treatment_type, "sd_estimate"]), 
#           from = min(curr_row_bootstrap), 
#           to = max(curr_row_bootstrap), 
#           add = T, col = "darkgreen", lwd = 2)
#   }
#   
#   # Confidence interval plot
#   ci_plot = ci_results_bootstrap %>%
#     mutate(significant = !(ci_lower <= 0 & ci_upper >= 0)) %>%
#     ggplot(aes(x = treatment_type)) +
#     geom_point(
#       aes(y = mean_estimate, color = significant), 
#       size = 3
#     ) +
#     geom_errorbar(
#       aes(ymin = ci_lower, ymax = ci_upper, color = significant),
#       width = .3, linewidth = 1.2
#     ) +
#     geom_hline(yintercept = 0, linetype = "dashed", color = "darkgreen", linewidth = .8) +
#     scale_color_manual(
#       values = c("FALSE" = "gray70", "TRUE" = "red"),
#       name = "Significant"
#     ) +
#     labs(
#       title = "Bootstrap Confidence Intervals for Treatment Effects",
#       x = "Treatment Type",
#       y = "Treatment Effect"
#     ) +
#     ylim(-0.2, 0.2) +
#     theme_minimal() +
#     theme(axis.text.x = element_text(angle = 30, hjust = 1))
#   
#   # add original estimates if provided
#   if (!is.null(original_estimates)) {
#     ci_plot = ci_plot +
#       geom_point(
#         data = original_estimates,
#         aes(x = treatment_type, y = mu),
#         color = "darkgreen", shape = 18, size = 3.5
#       )
#   }
#   
#   return(ci_plot)
# }
# 
# 
# bootstrap_estimates = bootstrap_GRF(
#   datasets.log,
#   post_periods = post_periods, 
#   n_bootstrap = 1000
# )
# bootstrap_ci = calculate_ci(bootstrap_estimates, original_estimates = estimate_results)
# plot_bootstrap(bootstrap_estimates, bootstrap_ci, estimate_results)
# # END: # ASSESS MODEL STABILITY


# # MODEL PERFORMANCE EVALUATION -------------------------------------------------
# # Evaluate prediction performance using regression forest
# X = datasets.log$T2y[get_covariate_cols(datasets.log$T2y, 10, post_period_cols)]
# Y = datasets.log$T2y$log_outcome_diff_10 + datasets.log$T2y$log_daily_usage_avg
# grf_Y = regression_forest(X, Y,
#                           num.trees = 2000,
#                           mtry = floor(ncol(X) / 3))
# # calculate R-squared
# rf.pred = predict(grf_Y)
# print(glue("R-squared: {1 - sum((Y - predict(grf_Y)$predictions)^2) / sum((Y - mean(Y))^2)}"))
# # 0.738