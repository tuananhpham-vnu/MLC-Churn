install.packages(c("tidyverse", "corrplot", "ggpubr","caret"))
library(tidyverse)
library(ranger)
library(caret)
library(dplyr)
library(stringr)
library(caret)
library(ranger)

set.seed(1234)

# CONFIG
input_csv <- "D:/Folder F/phamtuananh@23020010/UET.iSEML/2026.DAE.MLC Churn/Phase_0_EDA/outputs/preprocessed_improved.csv"
output_dir <- "D:/Folder F/phamtuananh@23020010/UET.iSEML/2026.DAE.MLC Churn/Phase_1_Model/outputs"

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

max_depth_values <- c(3, 5, NA)   # NA tương ứng với None
random_seed <- 1234
test_size <- 0.2

# F1 positive class = yes
f1_score_yes <- function(truth, pred) {
  truth <- factor(truth, levels = c("no", "yes"))
  pred <- factor(pred, levels = c("no", "yes"))
  
  cm <- table(truth, pred)
  
  tp <- cm["yes", "yes"]
  fp <- cm["no", "yes"]
  fn <- cm["yes", "no"]
  
  precision <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
  recall <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
  
  ifelse(precision + recall == 0, 0, 2 * precision * recall / (precision + recall))
}

ci95 <- function(x) {
  n <- length(x)
  mean_x <- mean(x)
  if (n < 2) return(c(mean_x, mean_x))
  se <- sd(x) / sqrt(n)
  margin <- qt(0.975, df = n - 1) * se
  c(mean_x - margin, mean_x + margin)
}
prepare_prun_state <- function(df) {
  # Target
  if ("churn_binary" %in% names(df)) {
    y <- ifelse(df$churn_binary == 1, "yes", "no")
  } else {
    y <- tolower(as.character(df$churn))
  }
  
  X <- df %>%
    select(-any_of(c("churn", "churn_binary")))
  
  # Nếu còn biến raw yes/no thì mã hóa thành *_yes
  if ("international_plan" %in% names(X)) {
    X <- X %>%
      mutate(international_plan_yes = ifelse(tolower(international_plan) == "yes", 1, 0)) %>%
      select(-international_plan)
  }
  
  if ("voice_mail_plan" %in% names(X)) {
    X <- X %>%
      mutate(voice_mail_plan_yes = ifelse(tolower(voice_mail_plan) == "yes", 1, 0)) %>%
      select(-voice_mail_plan)
  }
  
  # Bỏ state raw và state one-hot
  X <- X %>%
    select(-any_of("state")) %>%
    select(-starts_with("state_"))
  
  # KHÔNG bỏ area_code ở model này
  
  # Bỏ các cột *_no nếu đã có *_yes
  no_cols <- names(X)[str_detect(names(X), "_no$")]
  for (col in no_cols) {
    yes_col <- str_replace(col, "_no$", "_yes")
    if (yes_col %in% names(X)) {
      X <- X %>% select(-all_of(col))
    }
  }
  
  # Encode categorical còn lại, ví dụ area_code nếu còn dạng raw
  X <- model.matrix(~ . - 1, data = X) %>% as.data.frame()
  
  list(X = X, y = factor(y, levels = c("no", "yes")))
}
prepare_prun_area_code <- function(df) {
  # Target
  if ("churn_binary" %in% names(df)) {
    y <- ifelse(df$churn_binary == 1, "yes", "no")
  } else {
    y <- tolower(as.character(df$churn))
  }
  
  X <- df %>%
    select(-any_of(c("churn", "churn_binary")))
  
  # Nếu còn biến raw yes/no thì mã hóa thành *_yes
  if ("international_plan" %in% names(X)) {
    X <- X %>%
      mutate(international_plan_yes = ifelse(tolower(international_plan) == "yes", 1, 0)) %>%
      select(-international_plan)
  }
  
  if ("voice_mail_plan" %in% names(X)) {
    X <- X %>%
      mutate(voice_mail_plan_yes = ifelse(tolower(voice_mail_plan) == "yes", 1, 0)) %>%
      select(-voice_mail_plan)
  }
  
  # Bỏ area_code raw và area_code one-hot
  X <- X %>%
    select(-any_of("area_code")) %>%
    select(-starts_with("area_code_"))
  
  # KHÔNG bỏ state ở model này
  
  # Bỏ các cột *_no nếu đã có *_yes
  no_cols <- names(X)[str_detect(names(X), "_no$")]
  for (col in no_cols) {
    yes_col <- str_replace(col, "_no$", "_yes")
    if (yes_col %in% names(X)) {
      X <- X %>% select(-all_of(col))
    }
  }
  
  # Encode categorical còn lại, ví dụ state nếu còn dạng raw
  X <- model.matrix(~ . - 1, data = X) %>% as.data.frame()
  
  list(X = X, y = factor(y, levels = c("no", "yes")))
}
run_rf_experiment <- function(X, y, model_name, output_dir) {
  set.seed(1234)
  
  train_idx <- createDataPartition(y, p = 0.8, list = FALSE)
  
  X_train <- X[train_idx, , drop = FALSE]
  X_test <- X[-train_idx, , drop = FALSE]
  y_train <- y[train_idx]
  y_test <- y[-train_idx]
  
  results <- list()
  
  for (depth in max_depth_values) {
    depth_label <- ifelse(is.na(depth), "None", as.character(depth))
    
    if (is.na(depth)) {
      rf_model <- ranger(
        x = X_train,
        y = y_train,
        num.trees = 500,
        seed = random_seed,
        probability = FALSE,
        importance = "impurity"
      )
    } else {
      rf_model <- ranger(
        x = X_train,
        y = y_train,
        num.trees = 500,
        max.depth = depth,
        seed = random_seed,
        probability = FALSE,
        importance = "impurity"
      )
    }
    
    train_pred <- predict(rf_model, data = X_train)$predictions
    test_pred <- predict(rf_model, data = X_test)$predictions
    
    train_f1 <- f1_score_yes(y_train, train_pred)
    holdout_f1 <- f1_score_yes(y_test, test_pred)
    
    # 5-fold stratified CV trên train
    folds <- createFolds(y_train, k = 5, returnTrain = FALSE)
    cv_scores <- c()
    
    for (i in seq_along(folds)) {
      val_idx <- folds[[i]]
      
      X_cv_train <- X_train[-val_idx, , drop = FALSE]
      y_cv_train <- y_train[-val_idx]
      X_cv_val <- X_train[val_idx, , drop = FALSE]
      y_cv_val <- y_train[val_idx]
      
      if (is.na(depth)) {
        cv_model <- ranger(
          x = X_cv_train,
          y = y_cv_train,
          num.trees = 500,
          seed = random_seed,
          probability = FALSE
        )
      } else {
        cv_model <- ranger(
          x = X_cv_train,
          y = y_cv_train,
          num.trees = 500,
          max.depth = depth,
          seed = random_seed,
          probability = FALSE
        )
      }
      
      cv_pred <- predict(cv_model, data = X_cv_val)$predictions
      cv_scores[i] <- f1_score_yes(y_cv_val, cv_pred)
    }
    
    cv_mean <- mean(cv_scores)
    cv_interval <- ci95(cv_scores)
    
    result_row <- tibble(
      model = model_name,
      max_depth = depth_label,
      train_f1 = train_f1,
      holdout_f1 = holdout_f1,
      cv_f1_mean = cv_mean,
      cv_f1_ci95_lower = cv_interval[1],
      cv_f1_ci95_upper = cv_interval[2]
    )
    
    results[[depth_label]] <- result_row
    
    # Save selected features
    writeLines(
      colnames(X),
      file.path(output_dir, paste0("selected_features_", model_name, "_d", depth_label, ".txt"))
    )
    
    # Save feature importance
    importance_df <- tibble(
      feature = names(rf_model$variable.importance),
      importance = as.numeric(rf_model$variable.importance)
    ) %>%
      arrange(desc(importance))
    
    write.csv(
      importance_df,
      file.path(output_dir, paste0("feature_importance_", model_name, "_d", depth_label, ".csv")),
      row.names = FALSE
    )
    
    # Save CV fold scores
    cv_df <- tibble(
      fold = seq_along(cv_scores),
      f1_score = cv_scores
    )
    
    write.csv(
      cv_df,
      file.path(output_dir, paste0("cv_f1_scores_", model_name, "_d", depth_label, ".csv")),
      row.names = FALSE
    )
    
    cat("--------------------------------------------------\n")
    cat(model_name, "complete, max_depth =", depth_label, "\n")
    cat("Features used:", ncol(X), "\n")
    cat("Train F1:", round(train_f1, 6), "\n")
    cat("Holdout F1:", round(holdout_f1, 6), "\n")
    cat(
      "CV F1 mean (95% CI):",
      round(cv_mean, 6),
      "[",
      round(cv_interval[1], 6),
      ",",
      round(cv_interval[2], 6),
      "]\n"
    )
  }
  
  bind_rows(results)
}
df <- read.csv(input_csv)

# Model v1: prune state, keep area_code
data_prun_state <- prepare_prun_state(df)
results_prun_state <- run_rf_experiment(
  X = data_prun_state$X,
  y = data_prun_state$y,
  model_name = "model_prun_state",
  output_dir = output_dir
)

write.csv(
  results_prun_state,
  file.path(output_dir, "metrics_model_prun_state_by_max_depth.csv"),
  row.names = FALSE
)
