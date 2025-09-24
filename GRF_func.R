library(dplyr)
library(tidyr)
library(gbm)
library(xgboost)
library(grf)
library(policytree)
library(glue)
library(stringr)
library(ggplot2)


# FUNCTIONS --------------------------------------------------------------------
## BASELINE MACHINE LEARNING ---------------------------------------------------
fit_gbm = function(X_train, Y_train, X_test) {
  data_train = X_train %>%
    mutate(y = Y_train)
  
  formula_str = paste("y ~ ", paste(names(X_train), collapse = " + "))
  gbm_model = gbm(
    as.formula(formula_str),
    data_train,
    interaction.depth = 3, 
    n.trees = 500, 
    shrinkage = 0.05,
    distribution = "gaussian",
    bag.fraction = 0.8,
    cv.folds = 5,
    verbose = F
  )
  
  # use the optimal number of trees
  best_iter = gbm.perf(gbm_model, method = "cv", plot.it = F)
  gbm_pred = predict(gbm_model, X_test, n.trees = best_iter, type = "response")
  return(gbm_pred)
}

fit_xgb = function(X_train, Y_train, X_test) {
  xgb_params = list(
    objective = "reg:squarederror",
    eta = 0.05,
    max_depth = 3,
    subsample = 0.9,
    colsample_bynode = 0.33
  )
  
  X_train = as.matrix(X_train)
  X_test = as.matrix(X_test)
  
  # training with early stopping
  val_size = min(50, floor(length(Y_train) * 0.1))
  val_idx = sample(length(Y_train), val_size)
  
  dtrain = xgb.DMatrix(data = X_train[-val_idx, , drop = F],
                       label = Y_train[-val_idx])
  dval = xgb.DMatrix(data = X_train[val_idx, , drop = F],
                     label = Y_train[val_idx])
  dtest = xgb.DMatrix(data = X_test)
  
  xgb_model = xgb.train(
    params = xgb_params,
    data = dtrain,
    nrounds = 600,
    watchlist = list(train = dtrain, eval = dval),
    early_stopping_rounds = 10,
    verbose = 0
  )
  
  xgb_pred = predict(xgb_model, dtest)
  return(xgb_pred)
}

fit_grf = function(X_train, Y_train, X_test) {
  Y_train = as.vector(Y_train)
  grf_model = regression_forest(
    X_train,
    Y=Y_train,
    sample.fraction = 1,
    num.trees = 2000,
    mtry = floor(ncol(X_train) / 3),
    ci.group.size = 1
  )
  
  # Project fitted values based on covariates of current fold
  grf_pred = predict(grf_model, X_test, estimate.variance = F)$predictions
  return(grf_pred)
}

# k-fold cross-validation predictions for centering
compute_kfold_predictions = function(X, Y, k = 10, method = "gbm") {
  n = length(Y)
  predictions = rep(NA, n)
  
  # for balanced folds
  if (length(unique(Y)) == 2) {
    # binary outcome
    y_levels = unique(Y)
    folds = rep(NA, n)
    
    for (level in y_levels) {
      y_idx = which(Y == level)
      y_folds = sample(rep(1:k, length.out = length(y_idx)))
      folds[y_idx] = y_folds
    }
  }
  else {
    # continuous outcome value
    folds = sample(rep(1:k, length.out = n))
  }
  
  print(glue("Computing {k}-fold CV predictions for {n} observations..."))
  for (fold in 1:k) {
    if (fold %% 5 == 0) {
      print(glue("Progress: {fold}/{k} folds"))
    }
    
    train_idx = which(folds != fold)
    test_idx = which(folds == fold)
    
    X_train = X[train_idx, , drop = F]
    Y_train = Y[train_idx]
    X_test = X[test_idx, , drop = F]
    
    if (method == "gbm") {
      curr_pred = fit_gbm(X_train, Y_train, X_test)
    }
    else if (method == "xgb") {
      curr_pred = fit_xgb(X_train, Y_train, X_test)
    }
    else if (method == "grf") {
      curr_pred = fit_grf(X_train, Y_train, X_test)
    }
    else {
      stop("Method must be in c(\"gbm\", \"xgb\", \"grf\")")
    }
    
    predictions[test_idx] = curr_pred
  }
  
  # handle any remaining NA values
  na_idx = is.na(predictions)
  if (any(na_idx)) {
    print("NA(s) exist")
    predictions[na_idx] = mean(Y, na.rm = T)
  }
  
  print(glue("MSE: {round(mean((predictions - Y)^2), 5)}"))
  return(predictions)
}

# main function to run GRF analysis for treatment effect estimation
run_GRF = function(datasets,
                   Y_var = "log_outcome_diff",
                   post_periods,
                   post_periods_cols,
                   num_trees = 2000,
                   centering_method = NULL,
                   return_models = TRUE,
                   verbose = TRUE, 
                   seed = NULL) {
  fitted_models = list()
  estimate_results = data.frame(
    treatment_type = character(0),
    mu = numeric(0),
    se = numeric(0)
  )
  
  # going through each post-treatment period
  for (period in post_periods) {
    if (verbose == T) {
      print(glue("fit period {period} models..."))
    }
    
    curr_models = list(
      T1 = NULL,
      T2_ITT = NULL,
      T2y = NULL,
      T2n = NULL
    )
    
    X.T1 = datasets$T1[get_covariate_cols(datasets$T1, period, post_period_cols)]
    X.T2y = datasets$T2y[get_covariate_cols(datasets$T2y, period, post_period_cols)]
    X.T2n = datasets$T2n[get_covariate_cols(datasets$T2n, period, post_period_cols)]
    
    Y_str = ifelse(glue("{Y_var}_{period}") %in% names(datasets$T1), glue("{Y_var}_{period}"), NA)
    
    if (is.na(Y_str)) {
      warning(glue("outcome (Y) column for period {period} not found"))
    }
    
    seed = ifelse(is.null(seed), runif(1, 0, .Machine$integer.max), seed)
    set.seed(seed)
    
    # use default GRF centering (LOO grf)
    if (is.null(centering_method)) {
      # ref: https://grf-labs.github.io/grf/articles/grf_guide.html
      # set `W.hat = 0.5` under RCT
      curr_models$T1 = causal_forest(
        X = X.T1,
        Y = datasets$T1[[Y_str]],
        W = datasets$T1$Treatment,
        num.trees = num_trees,
        mtry = floor(ncol(X.T1) / 3),
        seed = seed
      )
      curr_models$T2_ITT = causal_forest(
        X = X.T2y,
        Y = datasets$T2y[[Y_str]],
        W = datasets$T2y$Treatment,
        num.trees = num_trees,
        mtry = floor(ncol(X.T2y) / 3),
        seed = seed
      )
      curr_models$T2y = instrumental_forest(
        X = X.T2y,
        Y = datasets$T2y[[Y_str]],
        W = datasets$T2y$T2y,
        Z = datasets$T2y$Treatment,
        num.trees = num_trees,
        mtry = floor(ncol(X.T2y) / 3),
        seed = seed
      )
      curr_models$T2n = instrumental_forest(
        X = X.T2n,
        Y = datasets$T2n[[Y_str]],
        W = datasets$T2n$T2n,
        Z = datasets$T2n$Treatment,
        num.trees = num_trees,
        mtry = floor(ncol(X.T2n) / 3),
        seed = seed
      )
    }
    else{  # use custom centering method
    
      # T1
      Y_hat.T1 = compute_kfold_predictions(
        X.T1,
        datasets$T1[[Y_str]],
        method = centering_method
      )
      Z_hat.T1 = compute_kfold_predictions(
        X.T1,
        datasets$T1$Treatment,
        method = centering_method
      )
      
      # T2 ITT & LATE takers
      Y_hat.T2y = compute_kfold_predictions(
        X.T2y,
        datasets$T2y[[Y_str]],
        method = centering_method
      )
      W_hat.T2y = compute_kfold_predictions(
        X.T2y,
        datasets$T2y$T2y,
        method = centering_method
      )
      Z_hat.T2y = compute_kfold_predictions(
        X.T2y,
        datasets$T2y$Treatment,
        method = centering_method
      )
      
      # T2 ITT & LATE non-takers
      Y_hat.T2n = compute_kfold_predictions(
        X.T2n,
        datasets$T2n[[Y_str]],
        method = centering_method
      )
      W_hat.T2n = compute_kfold_predictions(
        X.T2n,
        datasets$T2n$T2n,
        method = centering_method
      )
      Z_hat.T2n = compute_kfold_predictions(
        X.T2n,
        datasets$T2n$Treatment,
        method = centering_method
      )
      
      # ref: https://grf-labs.github.io/grf/articles/grf_guide.html
      # set `W.hat = 0.5` under RCT
      curr_models$T1 = causal_forest(
        X = X.T1,
        Y = datasets$T1[[Y_str]],
        W = datasets$T1$Treatment,
        Y.hat = Y_hat.T1,
        W.hat = Z_hat.T1,
        num.trees = num_trees,
        mtry = floor(ncol(X.T1) / 3),
        seed = seed
      )
      curr_models$T2_ITT = causal_forest(
        X = X.T2y,
        Y = datasets$T2y[[Y_str]],
        W = datasets$T2y$Treatment,
        Y.hat = Y_hat.T2y,
        W.hat = Z_hat.T2y,
        num.trees = num_trees,
        mtry = floor(ncol(X.T2y) / 3),
        seed = seed
      )
      curr_models$T2y = instrumental_forest(
        X = X.T2y,
        Y = datasets$T2y[[Y_str]],
        W = datasets$T2y$T2y,
        Z = datasets$T2y$Treatment,
        Y.hat = Y_hat.T2y,
        W.hat = W_hat.T2y,
        Z.hat = Z_hat.T2y,
        num.trees = num_trees,
        mtry = floor(ncol(X.T2y) / 3),
        seed = seed
      )
      curr_models$T2n = instrumental_forest(
        X = X.T2n,
        Y = datasets$T2n[[Y_str]],
        W = datasets$T2n$T2n,
        Z = datasets$T2n$Treatment,
        Y.hat = Y_hat.T2n,
        W.hat = W_hat.T2n,
        Z.hat = Z_hat.T2n,
        num.trees = num_trees,
        mtry = floor(ncol(X.T2n) / 3),
        seed = seed
      )
    }
    
    # store ATE and LATE result
    # T1 ATE
    result.T1 = average_treatment_effect(curr_models$T1)
    result.T2_ITT = average_treatment_effect(curr_models$T2_ITT)
    result.T2y = average_treatment_effect(curr_models$T2y)
    result.T2n = average_treatment_effect(curr_models$T2n)
    
    curr_results = data.frame(
      treatment_type = paste(c("T1", "T2_ITT", "T2y", "T2n"), sprintf("%02d", period), sep = "_"),
      mu = c(
        result.T1["estimate"],
        result.T2_ITT["estimate"],
        result.T2y["estimate"],
        -result.T2n["estimate"]
      ),
      se = c(
        result.T1["std.err"],
        result.T2_ITT["std.err"],
        result.T2y["std.err"],
        result.T2n["std.err"]
      )
    )
    
    estimate_results = rbind(
      estimate_results,
      curr_results
    )
    fitted_models[[glue("period_{period}")]] = curr_models
  }
  
  estimate_results = estimate_results %>%
    arrange(treatment_type)
  
  if (return_models) {
    return(list(models = fitted_models, estimates = estimate_results))
  }
  else {
    return(estimate_results)
  }
}

## FEATURE IMPORTANCE ANALYSIS -------------------------------------------------
# analyze feature importance using split frequencies and variable importance
analyze_feature_importance = function(model, depth = 4) {
  split_freq = as.data.frame(
    t(split_frequencies(forest = model, max.depth = depth))
  )
  names(split_freq) = paste("split_freq", paste0("depth", 1:depth), sep = "_")
  
  split_freq = split_freq %>%
    mutate(
      total_freq = rowSums(select(., everything())),
      variable = names(model$X.orig)
    )
  
  importance_df = data.frame(
    variable = split_freq$variable,
    importance = variable_importance(model, max.depth = depth)
  ) %>%
    arrange(desc(importance)) %>%
    left_join(split_freq, by = "variable")
  
  return(importance_df)
}

# find important variables common across different models
find_important_vars = function(importance_df_1,
                               importance_df_2, 
                               top_n = 15,
                               rank_by = "importance",
                               method = "intersection") {
  
  rank_var = case_when(
    rank_by == "importance" ~ "importance",
    rank_by == "frequency" ~ "total_freq",
    .default = "importance"
  )
  top_vars_1 = importance_df_1 %>%
    arrange(desc(!!sym(rank_var))) %>%
    slice_head(n = top_n) %>%
    pull(variable)
  
  top_vars_2 = importance_df_2 %>%
    arrange(desc(!!sym(rank_var))) %>%
    slice_head(n = top_n) %>%
    pull(variable)
  
  if (method == "intersection") {
    common_vars = intersect(top_vars_1, top_vars_2)
  }
  else if (method == "union") {
    common_vars = union(top_vars_1, top_vars_2)
  } 
  
  print(glue("Summary: {length(common_vars)} important variable(s) with method \"{method}\""))
  
  return(common_vars)
}

## POLICY TREE -----------------------------------------------------------------
compare_targeting_strategies = function(T1_dat,
                                        T2_dat, 
                                        models,
                                        tree) {
  features = tree$columns
  
  # T1 Policy Tree
  # Apply policy tree to T1 data
  tree_pred.T1 = predict(tree, T1_dat[features])
  tree_idx.T1 = which(tree_pred.T1 == 2)  # targeted group
  tree_ATE.T1 = average_treatment_effect(models$T1, subset = tree_idx.T1)
  
  # Winner's Bias Corrected
  # Apply policy tree to T2 data for Winner's Bias correction
  tree_pred.T2 = predict(tree, T2_dat[features])
  tree_idx.T2 = which(tree_pred.T2 == 2)  # targeted group
  tree_ITT.T2 = average_treatment_effect(models$T2_ITT, subset = tree_idx.T2)
  
  # T2 Self-Selection
  # calculate treatment effects for endogenous group (T2y == 1)
  endo_idx = which(T2_dat$T2y == 1)
  endo_ITT = average_treatment_effect(models$T2_ITT, subset = endo_idx)
  
  # create result table
  results_table = data.frame(
    Targeting_Method = c(
      "Policy Tree (T1, ATE)",
      "Policy Tree (T2, ITT)",
      "T2 Self-Selection (ITT)"
    ),
    ATE = c(
      tree_ATE.T1["estimate"],
      tree_ITT.T2["estimate"],
      endo_ITT["estimate"]
    ),
    Std_Err = c(
      tree_ATE.T1["std.err"],
      tree_ITT.T2["std.err"],
      endo_ITT["std.err"]
    ),
    CI_Lower = c(
      tree_ATE.T1["estimate"] - 1.96 * tree_ATE.T1["std.err"],
      tree_ITT.T2["estimate"] - 1.96 * tree_ITT.T2["std.err"], 
      endo_ITT["estimate"] - 1.96 * endo_ITT["std.err"]
    ),
    CI_Upper = c(
      tree_ATE.T1["estimate"] + 1.96 * tree_ATE.T1["std.err"],
      tree_ITT.T2["estimate"] + 1.96 * tree_ITT.T2["std.err"], 
      endo_ITT["estimate"] + 1.96 * endo_ITT["std.err"]
    ),
    n = c(
      length(tree_idx.T1),
      length(tree_idx.T2),
      length(endo_idx)
    ),
    Targeting_Rate = c(
      length(tree_idx.T1) / nrow(T1_dat),
      length(tree_idx.T2) / nrow(T2_dat),
      length(endo_idx) / nrow(T2_dat)
    )
  )
  
  float_cols = c("ATE", "Std_Err", "CI_Lower", "CI_Upper", "Targeting_Rate")
  results_table[, float_cols] = round(results_table[, float_cols], 5)
  
  return(results_table)
}

# analyze each leaf node of a policy tree
analyze_tree_leaves = function(data,
                               model,
                               tree) {
  features = tree$columns
  
  # get node assignments for each observation
  node_ids = predict(tree, data[features], type = "node.id")
  
  # get action assignments for each observation  
  action_ids = predict(tree, data[features], type = "action.id")
  
  node_levels = unique(node_ids)
  
  # initialization
  results = data.frame(
    Node_ID = integer(),
    Action = integer(),
    Avg_Effect = numeric(),
    Std_Err = numeric(),
    CI_Lower = numeric(),
    CI_Upper = numeric(),
    n = integer(),
    Targeting_Rate = numeric(),
    stringsAsFactors = F
  )
  
  # calculate statistics for each leaf node
  for (node in node_levels) {
    node_idx = which(node_ids == node)
    curr_action = unique(action_ids[node_idx])
    
    # handle case where there might be multiple actions (shouldn't happen in leaf)
    if (length(curr_action) > 1) {
      warning(glue("Multiple actions found in leaf node {node}. Using most frequent action."))
      curr_action = as.numeric(names(sort(table(action_ids[node_idx]), decreasing = T)[1]))
    }
    
    # calculate ATE for this subset
    if (length(node_idx) > 1) {
      ate_result = average_treatment_effect(model, subset = node_idx)
      
      avg_effect = ate_result["estimate"]
      std_err = ate_result["std.err"]
      ci_lower = avg_effect - 1.96 * std_err
      ci_upper = avg_effect + 1.96 * std_err
    }
    else {
      # case when only one observation
      avg_effect = NA
      std_err = NA
      ci_lower = NA
      ci_upper = NA
    }
    
    results = rbind(results, data.frame(
      Node_ID = node,
      Action = glue("{curr_action} ({tree$action.names[curr_action]})"),
      Avg_Effect = avg_effect,
      Std_Err = std_err,
      CI_Lower = ci_lower,
      CI_Upper = ci_upper,
      n = length(node_idx),
      Targeting_Rate = length(node_idx) / nrow(data),
      stringsAsFactors = F
    ))
  }
  
  float_cols = c("Avg_Effect", "Std_Err", "CI_Lower", "CI_Upper", "Targeting_Rate")
  results[float_cols] = round(results[float_cols], 5)
  
  results = results %>%
    arrange(Node_ID)
  rownames(results) = NULL
  
  return(results)
}

## VISUALIZATION ---------------------------------------------------------------
# create coefficient plot with CIs
plot_coef = function(treatment_results,
                     title = "Treatment Effects with Confidence Intervals (GRF)",
                     phase_label = "Phase 2") {
  
  plot_data = treatment_results %>%
    mutate(
      ci_90_lower = mu - 1.645 * se,
      ci_90_upper = mu + 1.645 * se,
      ci_95_lower = mu - 1.96 * se,
      ci_95_upper = mu + 1.96 * se,
      ci_99_lower = mu - 2.576 * se,
      ci_99_upper = mu + 2.576 * se
    )
  
  plot = ggplot(plot_data, aes(x = treatment_type, y = mu)) +
    geom_errorbar(aes(ymin = ci_99_lower, ymax = ci_99_upper),
                  width = .3, linewidth = 1.5, color = "grey80") +
    geom_errorbar(aes(ymin = ci_95_lower, ymax = ci_95_upper),
                  width = .3, linewidth = 1.5, color = "grey70") +
    geom_errorbar(aes(ymin = ci_90_lower, ymax = ci_90_upper),
                  width = .3, linewidth = 1.5, color = "darkgreen") +
    geom_point(size = 2, color = "darkgreen") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red", linewidth = .8) +
    # ylim(-0.2, 0.2) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1)) +
    labs(title = title,
         x = phase_label,
         y = "Coefficient")
  return(plot)
}

plot_feature_importance = function(importance_df,
                                   depth,
                                   top_n = 10,
                                   plot_name = NULL) {
  plot_name = ifelse(is.null(plot_name), "", plot_name)
  importance_df %>%
    slice_head(n = top_n) %>%
    ggplot(aes(x = reorder(variable, importance), y = importance)) +
    geom_col(fill = "aquamarine3", alpha = 0.8) +
    coord_flip() +
    labs(
      title = glue("Variable Importance - {plot_name}"),
      x = "Variables",
      y = "Importance Score",
      caption = glue("Based on weighted split frequency (decay.exponent = 2, max.depth = {depth})")
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 12, hjust = 0.5),
      axis.text.y = element_text(size = 9),
      plot.caption = element_text(size = 9, hjust = 0.5)
    )
}

plot_split_pattern = function(importance_df,
                              depth,
                              top_n = 10,
                              rank_by = "importance",
                              plot_name = NULL) {
  plot_name = ifelse(is.null(plot_name), "", plot_name)
  
  if (rank_by == "importance") {
    selected_df = importance_df %>%
      arrange(desc(importance))
    slice_head(n = top_n)
  }
  if (rank_by == "frequency") {
    selected_df = importance_df %>%
      arrange(desc(total_freq)) %>%
      slice_head(n = top_n)
  }
  else {
    stop("rank_by must be in c(\"importance\", \"frequency\")")
  }
  
  # plot
  selected_df %>%
    pivot_longer(
      cols = all_of(paste("split_freq", paste0("depth", 1:depth), sep = "_")),
      names_to = "depth_level",
      values_to = "frequency"
    ) %>%
    mutate(
      depth_level = as.numeric(str_extract(depth_level, "\\d+"))
    ) %>%
    ggplot(aes(x = reorder(variable, !!sym(rank_by)), y = frequency, fill = depth_level)) +
    geom_col(position = "stack", alpha = 0.8) +
    coord_flip() +
    scale_fill_viridis_c(option = "mako", name = "Depth") +
    labs(
      title = glue("Split Frequency by Depth - {plot_name}"),
      x = "Variables",
      y = "Split Frequency",
      caption = "Stacked bars show frequency of splits at different depths"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 12, hjust = 0.5),
      legend.title = element_text(size = 9),
      legend.text = element_text(size = 7),
      legend.position = "right",
      plot.caption = element_text(size = 9, hjust = 0.5)
    )
}

# export individual decision trees as .svg files
plot_grftree = function(model_names,
                        models,
                        tree_idx = c(1, 100, 500, 1000, 2000)) {
  # `get_tree()`: retrieve a single tree from a trained forest
  # create output directory
  folder = "./private/figure"
  if (!dir.exists(folder)) {
    dir.create(folder)
  }
  for (model_name in model_names) {
    subfolder = glue("{folder}/{model_name}")
    if (!dir.exists(subfolder)) {
      dir.create(subfolder)
    }
    for(i in tree_idx) {
      tree = get_tree(models[[model_name]], i)
      tree.plot = plot(tree)
      cat(DiagrammeRsvg::export_svg(tree.plot), file = glue("{subfolder}/{model_name}_tree_{i}.svg"))
    }
  }
  print("Saving the GRF trees in .svg (see ./figure)")
}

# create scatter plot showing the relationship between specified variable and ATE
plot_ate_scatter = function(x_var,
                            ate_values,
                            title = NULL,
                            x_lab = NULL,
                            y_lab = "Average Treatment Effect (ATE)",
                            add_trend = NULL) {
  
  # create plotting data
  plot_data = data.frame(
    x = x_var,
    ate = ate_values
  ) %>%
    filter(!is.na(x) & !is.na(ate))
  
  # generate labels if not provided
  if (is.null(x_lab)) {
    x_lab = "x_var"
  }
  if (is.null(title)) {
    title = glue("{y_lab} vs. {x_lab}")
  }
  
  # create base plot
  plot = ggplot(plot_data, aes(x = x, y = ate)) +
    geom_point(size = .6, alpha = .8, color = "aquamarine4")
  
  # add trend line if requested
  if (!is.null(add_trend)) {
    if (add_trend == "linear") {
      plot = plot + 
        geom_smooth(
          method = "lm",
          se = F,
          color = "grey", 
          linewidth = .8,
          alpha = .3
        )
    }
    else if (add_trend == "poly") {
      plot = plot + 
        geom_smooth(
          method = "lm",
          formula = y ~ poly(x, 2),
          se = F,
          color = "grey", 
          linewidth = .8,
          alpha = .3
        )
    }
  }
  
  plot = plot +
    labs(
      title = title,
      x = x_lab,
      y = y_lab,
      caption = glue("N = {nrow(plot_data)} observations")
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
      axis.title = element_text(size = 10),
      axis.text = element_text(size = 10)
    )
  
  return(plot)
}

plot_ate_dist = function(scores,
                         x_lab = "ATE",
                         add_to_caption = NULL) {
  
  scores_df = data.frame(score = scores)
  
  caption = glue("N = {nrow(scores_df)} observations")
  if (!is.null(caption)) {
    caption = glue("{add_to_caption}; {caption}")
  }
  
  binwidth = ifelse(max(scores) - min(scores) > 2, 0.2, 0.005)
  plot = ggplot(scores_df, aes(x = score)) +
    geom_histogram(
      aes(y = after_stat(count / sum(count)), fill = score < 0),
      binwidth = binwidth,
      boundary = 0,
      color = "white",
      alpha = .8
    ) +
    scale_fill_manual(
      values = c("TRUE" = "aquamarine3", "FALSE" = "azure3"),
      guide = "none"
    ) +
    labs(
      title = glue("Distribution of {x_lab}"),
      x = x_lab,
      y = "fraction",
      caption = caption
    ) +
    theme_minimal() +
    ylim(0, 0.3) +
    theme(
      plot.title = element_text(size = 12, hjust = 0.5, face = "bold")
    )
  
  if (binwidth == 0.2) {
    plot = plot + scale_x_continuous(
      breaks = seq(2.5 * floor(min(scores) / 2.5), 2.5 * ceiling(max(scores) / 2.5), by = 2.5)
    )
  }
  else {
    plot = plot + scale_x_continuous(
      breaks = seq(0.05 * floor(min(scores) / 0.05), 0.05 * ceiling(max(scores) / 0.05), by = 0.05)
    )
  }
  
  return(plot)
}
# END: # FUNCTIONS