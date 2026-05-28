#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ranger)
  library(ggplot2)
})

POSITIVE_CLASS <- "yes"
DEFAULT_MAX_DEPTHS <- "3,5,None"

get_script_path <- function() {
  raw_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", raw_args, value = TRUE)
  if (length(file_arg) == 0) {
    return(file.path(getwd(), "Phase_1_Model", "01_model_v2.R"))
  }
  normalizePath(sub("^--file=", "", file_arg[[1]]), winslash = "/", mustWork = FALSE)
}

SCRIPT_PATH <- get_script_path()

is_project_root <- function(path) {
  dir.exists(file.path(path, "Phase_0_EDA")) &&
    dir.exists(file.path(path, "Phase_1_Model"))
}

find_project_root <- function(script_path) {
  start_points <- unique(c(
    normalizePath(getwd(), winslash = "/", mustWork = FALSE),
    normalizePath(dirname(script_path), winslash = "/", mustWork = FALSE)
  ))

  for (start in start_points) {
    current <- start
    repeat {
      if (is_project_root(current)) {
        return(current)
      }
      parent <- dirname(current)
      if (identical(parent, current)) {
        break
      }
      current <- parent
    }
  }

  stop("Cannot detect project root. Please run from project folder or pass --input-csv explicitly.")
}

ROOT_DIR <- find_project_root(SCRIPT_PATH)
DEFAULT_INPUT <- file.path(ROOT_DIR, "Phase_0_EDA", "outputs", "preprocessed_improved.csv")
DEFAULT_OUTPUT_DIR <- file.path(ROOT_DIR, "Phase_1_Model", "outputs")

parse_cli_args <- function(raw_args, defaults) {
  parsed <- defaults
  i <- 1L
  while (i <= length(raw_args)) {
    token <- raw_args[[i]]
    if (!startsWith(token, "--")) {
      stop(sprintf("Invalid argument format: %s", token))
    }

    key <- NULL
    value <- NULL
    stripped <- sub("^--", "", token)
    if (grepl("=", stripped, fixed = TRUE)) {
      parts <- strsplit(stripped, "=", fixed = TRUE)[[1]]
      key <- parts[[1]]
      value <- paste(parts[-1], collapse = "=")
      i <- i + 1L
    } else {
      key <- stripped
      if (i == length(raw_args)) {
        stop(sprintf("Missing value for argument --%s", key))
      }
      value <- raw_args[[i + 1L]]
      i <- i + 2L
    }

    key <- gsub("-", "_", key)
    if (!key %in% names(parsed)) {
      stop(sprintf("Unknown argument: --%s", key))
    }
    parsed[[key]] <- value
  }
  parsed
}

parse_max_depths <- function(raw_value) {
  tokens <- trimws(unlist(strsplit(raw_value, ",", fixed = TRUE)))
  tokens <- tokens[nzchar(tokens)]
  if (length(tokens) == 0) {
    stop("No valid max_depth was provided.")
  }

  values <- c()
  for (token in tokens) {
    low <- tolower(token)
    if (low %in% c("none", "null")) {
      values <- c(values, NA_real_)
    } else {
      number <- suppressWarnings(as.integer(token))
      if (is.na(number) || number <= 0) {
        stop("max_depth must be a positive integer or None.")
      }
      values <- c(values, as.numeric(number))
    }
  }

  unique(values)
}

depth_tag <- function(max_depth) {
  if (is.na(max_depth)) "none" else as.character(as.integer(max_depth))
}

as_yes_no_factor <- function(binary_vector) {
  factor(ifelse(binary_vector == 1L, "yes", "no"), levels = c("no", "yes"))
}

ensure_binary_target <- function(df) {
  if ("churn_binary" %in% names(df)) {
    y <- suppressWarnings(as.integer(df$churn_binary))
    if (any(is.na(y))) {
      stop("Column 'churn_binary' contains invalid values.")
    }
    y <- ifelse(y > 0L, 1L, 0L)
    return(as_yes_no_factor(y))
  }
  if ("churn" %in% names(df)) {
    churn_text <- tolower(trimws(as.character(df$churn)))
    y <- ifelse(churn_text == "yes", 1L, 0L)
    return(as_yes_no_factor(y))
  }
  stop("Input data must contain 'churn_binary' or 'churn'.")
}

drop_redundant_no_columns <- function(columns) {
  drop_cols <- c()
  column_set <- unique(columns)
  for (col in columns) {
    if (grepl("_no$", col)) {
      paired_yes <- sub("_no$", "_yes", col)
      if (paired_yes %in% column_set) {
        drop_cols <- c(drop_cols, col)
      }
    }
  }
  unique(drop_cols)
}

find_high_corr_pairs <- function(feature_df, threshold = 0.85) {
  numeric_mask <- vapply(feature_df, is.numeric, logical(1))
  numeric_df <- feature_df[, numeric_mask, drop = FALSE]
  if (ncol(numeric_df) < 2) {
    return(data.frame(feature_1 = character(0), feature_2 = character(0), correlation = numeric(0)))
  }

  cor_mat <- suppressWarnings(cor(numeric_df, use = "pairwise.complete.obs"))
  if (any(is.na(cor_mat))) {
    return(data.frame(feature_1 = character(0), feature_2 = character(0), correlation = numeric(0)))
  }

  idx <- which(abs(cor_mat) >= threshold & upper.tri(cor_mat), arr.ind = TRUE)
  if (nrow(idx) == 0) {
    return(data.frame(feature_1 = character(0), feature_2 = character(0), correlation = numeric(0)))
  }

  data.frame(
    feature_1 = colnames(cor_mat)[idx[, "row"]],
    feature_2 = colnames(cor_mat)[idx[, "col"]],
    correlation = cor_mat[idx],
    stringsAsFactors = FALSE
  )
}

sanitize_binary_column <- function(x) {
  tolower(trimws(as.character(x)))
}

prepare_features <- function(df) {
  y <- ensure_binary_target(df)
  X <- df[, setdiff(names(df), c("churn_binary", "churn")), drop = FALSE]

  selection_notes <- data.frame(
    action = character(0),
    variable = character(0),
    reason = character(0),
    stringsAsFactors = FALSE
  )

  for (base_col in c("international_plan", "voice_mail_plan")) {
    if (base_col %in% names(X)) {
      values <- sanitize_binary_column(X[[base_col]])
      yes_col <- paste0(base_col, "_yes")
      X[[yes_col]] <- ifelse(values == "yes", 1L, 0L)
      X[[base_col]] <- NULL
      selection_notes <- rbind(
        selection_notes,
        data.frame(
          action = "transform",
          variable = base_col,
          reason = sprintf("Chuyen ve chi bao %s de dong nhat voi positive class yes.", yes_col),
          stringsAsFactors = FALSE
        )
      )
    }
  }

  if ("state" %in% names(X)) {
    X$state <- NULL
    selection_notes <- rbind(
      selection_notes,
      data.frame(
        action = "drop",
        variable = "state",
        reason = "Loai bo bien dia ly co tinh dinh danh, kho tong quat va kho hanh dong trong nghiep vu.",
        stringsAsFactors = FALSE
      )
    )
  }
  state_cols <- grep("^state_", names(X), value = TRUE)
  if (length(state_cols) > 0) {
    X <- X[, setdiff(names(X), state_cols), drop = FALSE]
    selection_notes <- rbind(
      selection_notes,
      data.frame(
        action = "drop",
        variable = paste(state_cols, collapse = ","),
        reason = "Loai bo one-hot state theo policy model_v2 (y nghia thuc tien thap cho quyet dinh giu chan).",
        stringsAsFactors = FALSE
      )
    )
  }

  if ("area_code" %in% names(X)) {
    X$area_code <- NULL
    selection_notes <- rbind(
      selection_notes,
      data.frame(
        action = "drop",
        variable = "area_code",
        reason = "Loai bo ma vung do tinh thong tin han che va khong phan anh hanh vi su dung.",
        stringsAsFactors = FALSE
      )
    )
  }
  area_cols <- grep("^area_code_", names(X), value = TRUE)
  if (length(area_cols) > 0) {
    X <- X[, setdiff(names(X), area_cols), drop = FALSE]
    selection_notes <- rbind(
      selection_notes,
      data.frame(
        action = "drop",
        variable = paste(area_cols, collapse = ","),
        reason = "Loai bo one-hot area_code theo policy model_v2.",
        stringsAsFactors = FALSE
      )
    )
  }

  redundant_no_cols <- drop_redundant_no_columns(names(X))
  if (length(redundant_no_cols) > 0) {
    X <- X[, setdiff(names(X), redundant_no_cols), drop = FALSE]
    selection_notes <- rbind(
      selection_notes,
      data.frame(
        action = "drop",
        variable = paste(redundant_no_cols, collapse = ","),
        reason = "Loai bo cot *_no vi trung lap thong tin voi cot *_yes.",
        stringsAsFactors = FALSE
      )
    )
  }

  object_cols <- names(X)[vapply(X, function(col) is.character(col) || is.factor(col), logical(1))]
  if (length(object_cols) > 0) {
    for (col in object_cols) {
      X[[col]] <- factor(tolower(trimws(as.character(X[[col]]))))
    }
    dummy_df <- as.data.frame(model.matrix(~ . - 1, data = X[, object_cols, drop = FALSE]))
    X <- cbind(X[, setdiff(names(X), object_cols), drop = FALSE], dummy_df)
  }

  logical_cols <- names(X)[vapply(X, is.logical, logical(1))]
  for (col in logical_cols) {
    X[[col]] <- as.integer(X[[col]])
  }

  numeric_cols <- names(X)[vapply(X, is.numeric, logical(1))]
  for (col in numeric_cols) {
    X[[col]][is.infinite(X[[col]])] <- NA_real_
  }
  if (anyNA(X)) {
    stop("Found missing or invalid values in feature matrix after preprocessing.")
  }

  corr_pairs <- find_high_corr_pairs(X, threshold = 0.85)

  list(
    X = X,
    y = y,
    selected_features = names(X),
    selection_notes = selection_notes,
    corr_pairs = corr_pairs
  )
}

stratified_train_test_split <- function(y, test_size = 0.2, seed = 1234L) {
  set.seed(seed)
  n <- length(y)
  all_idx <- seq_len(n)
  test_idx <- c()

  for (level_name in levels(y)) {
    class_idx <- all_idx[y == level_name]
    n_class <- length(class_idx)
    n_test <- max(1L, round(n_class * test_size))
    n_test <- min(n_test, n_class - 1L)
    chosen <- sample(class_idx, size = n_test, replace = FALSE)
    test_idx <- c(test_idx, chosen)
  }

  test_idx <- sort(unique(test_idx))
  train_idx <- setdiff(all_idx, test_idx)
  list(train = train_idx, test = test_idx)
}

make_stratified_folds <- function(y, n_splits = 5L, seed = 1234L) {
  set.seed(seed)
  all_idx <- seq_along(y)
  folds <- vector("list", n_splits)
  for (i in seq_len(n_splits)) {
    folds[[i]] <- integer(0)
  }

  for (level_name in levels(y)) {
    class_idx <- sample(all_idx[y == level_name], length(all_idx[y == level_name]))
    fold_id <- rep(seq_len(n_splits), length.out = length(class_idx))
    for (k in seq_len(n_splits)) {
      folds[[k]] <- c(folds[[k]], class_idx[fold_id == k])
    }
  }

  lapply(folds, sort)
}

metric_counts <- function(y_true, y_pred, positive = POSITIVE_CLASS) {
  tp <- sum(y_pred == positive & y_true == positive)
  fp <- sum(y_pred == positive & y_true != positive)
  fn <- sum(y_pred != positive & y_true == positive)
  tn <- sum(y_pred != positive & y_true != positive)
  list(tp = tp, fp = fp, fn = fn, tn = tn)
}

metric_precision <- function(y_true, y_pred, positive = POSITIVE_CLASS) {
  cnt <- metric_counts(y_true, y_pred, positive)
  denom <- cnt$tp + cnt$fp
  if (denom == 0) return(0)
  cnt$tp / denom
}

metric_recall <- function(y_true, y_pred, positive = POSITIVE_CLASS) {
  cnt <- metric_counts(y_true, y_pred, positive)
  denom <- cnt$tp + cnt$fn
  if (denom == 0) return(0)
  cnt$tp / denom
}

metric_f1 <- function(y_true, y_pred, positive = POSITIVE_CLASS) {
  p <- metric_precision(y_true, y_pred, positive)
  r <- metric_recall(y_true, y_pred, positive)
  if (p + r == 0) return(0)
  2 * p * r / (p + r)
}

metric_accuracy <- function(y_true, y_pred) {
  mean(y_true == y_pred)
}

cv_confidence_interval <- function(scores, confidence = 0.95) {
  n <- length(scores)
  m <- mean(scores)
  if (n < 2) {
    return(c(lower = m, upper = m))
  }
  sem <- sd(scores) / sqrt(n)
  margin <- sem * qt((1 + confidence) / 2, df = n - 1)
  c(lower = m - margin, upper = m + margin)
}

fit_rf_model <- function(X_train, y_train, max_depth, random_seed) {
  train_df <- X_train
  train_df$churn_label <- y_train

  if (is.na(max_depth)) {
    ranger(
      churn_label ~ .,
      data = train_df,
      seed = random_seed,
      importance = "impurity"
    )
  } else {
    ranger(
      churn_label ~ .,
      data = train_df,
      seed = random_seed,
      max.depth = as.integer(max_depth),
      importance = "impurity"
    )
  }
}

predict_labels <- function(model, X_data) {
  pred <- predict(model, data = X_data)$predictions
  as.character(pred)
}

save_feature_importance_plot <- function(importance_df, path, max_depth) {
  top_df <- head(importance_df, 25)
  top_df <- top_df[order(top_df$importance, decreasing = FALSE), , drop = FALSE]
  p <- ggplot(top_df, aes(x = reorder(feature, importance), y = importance)) +
    geom_col(fill = "#B279A2") +
    coord_flip() +
    labs(
      title = sprintf("Model v2 RF Feature Importance (max_depth=%s)", ifelse(is.na(max_depth), "None", max_depth)),
      x = "Feature",
      y = "Importance"
    ) +
    theme_minimal(base_size = 11)
  ggsave(path, p, width = 9, height = max(5, nrow(top_df) * 0.28), dpi = 160)
}

save_confusion_matrix_plot <- function(y_true, y_pred, path, max_depth) {
  cm <- table(
    actual = factor(y_true, levels = c("no", "yes")),
    predicted = factor(y_pred, levels = c("no", "yes"))
  )
  cm_df <- as.data.frame(cm)
  p <- ggplot(cm_df, aes(x = predicted, y = actual, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), color = "black", size = 5) +
    scale_fill_gradient(low = "#f2e8f1", high = "#8f3f86") +
    labs(
      title = sprintf("Model v2 Holdout Confusion Matrix (max_depth=%s)", ifelse(is.na(max_depth), "None", max_depth)),
      x = "Predicted",
      y = "Actual"
    ) +
    theme_minimal(base_size = 11)
  ggsave(path, p, width = 5, height = 4.5, dpi = 160)
}

write_metrics_txt <- function(metrics_row, path) {
  lines <- c("Phase 1 Model v2 Metrics (R)", "================================", "")
  for (nm in names(metrics_row)) {
    value <- metrics_row[[nm]]
    if (is.numeric(value) && !is.na(value)) {
      lines <- c(lines, sprintf("%s: %.6f", nm, value))
    } else {
      lines <- c(lines, sprintf("%s: %s", nm, as.character(value)))
    }
  }
  writeLines(lines, con = path, useBytes = TRUE)
}

write_selection_notes <- function(selection_notes, corr_pairs, path) {
  lines <- c(
    "Feature Selection Notes (Model v2 - R)",
    "=======================================",
    "",
    "Nguyen tac:",
    "- Loai bo state*/area_code* vi y nghia thuc tien thap cho hanh dong giu chan va de giam do phuc tap dac trung.",
    "- Loai bo *_no neu da co *_yes de tranh trung lap thong tin.",
    "- Su dung tuong quan de ra soat cap bien co |r| cao trong tap bien sau tien xu ly.",
    ""
  )

  if (nrow(selection_notes) > 0) {
    lines <- c(lines, "Cac thao tac da ap dung:")
    for (i in seq_len(nrow(selection_notes))) {
      row <- selection_notes[i, ]
      lines <- c(lines, sprintf("- [%s] %s: %s", row$action, row$variable, row$reason))
    }
    lines <- c(lines, "")
  }

  if (nrow(corr_pairs) == 0) {
    lines <- c(lines, "Khong phat hien cap bien nao co |r| >= 0.85 sau khi loc feature.")
  } else {
    lines <- c(lines, "Cac cap bien co |r| >= 0.85 sau khi loc feature:")
    for (i in seq_len(nrow(corr_pairs))) {
      row <- corr_pairs[i, ]
      lines <- c(lines, sprintf("- %s ~ %s: r = %.4f", row$feature_1, row$feature_2, row$correlation))
    }
  }

  writeLines(lines, con = path, useBytes = TRUE)
}

to_json_like_text <- function(named_list) {
  parts <- c("{")
  keys <- names(named_list)
  for (i in seq_along(keys)) {
    key <- keys[[i]]
    value <- named_list[[key]]
    value_text <- if (is.na(value)) {
      "null"
    } else if (is.numeric(value)) {
      format(value, scientific = FALSE, trim = TRUE)
    } else {
      sprintf("\"%s\"", gsub("\"", "\\\\\"", as.character(value)))
    }
    suffix <- if (i < length(keys)) "," else ""
    parts <- c(parts, sprintf("  \"%s\": %s%s", key, value_text, suffix))
  }
  parts <- c(parts, "}")
  paste(parts, collapse = "\n")
}

main <- function() {
  raw_args <- commandArgs(trailingOnly = TRUE)
  defaults <- list(
    input_csv = DEFAULT_INPUT,
    output_dir = DEFAULT_OUTPUT_DIR,
    random_seed = "1234",
    test_size = "0.2",
    max_depths = DEFAULT_MAX_DEPTHS
  )
  args <- parse_cli_args(raw_args, defaults)

  input_csv <- normalizePath(args$input_csv, winslash = "/", mustWork = FALSE)
  output_dir <- normalizePath(args$output_dir, winslash = "/", mustWork = FALSE)
  random_seed <- as.integer(args$random_seed)
  test_size <- as.numeric(args$test_size)
  max_depth_values <- parse_max_depths(args$max_depths)

  if (!file.exists(input_csv)) {
    stop(sprintf("Input file not found: %s", input_csv))
  }
  if (is.na(random_seed)) {
    stop("random_seed must be an integer.")
  }
  if (is.na(test_size) || test_size <= 0 || test_size >= 1) {
    stop("test_size must be in (0, 1).")
  }

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  set.seed(random_seed)

  df <- read.csv(input_csv, stringsAsFactors = FALSE, check.names = FALSE)
  prepared <- prepare_features(df)
  X <- prepared$X
  y <- prepared$y
  selected_features <- prepared$selected_features

  corr_pairs_path <- file.path(output_dir, "correlation_pairs_v2_r.csv")
  write.csv(prepared$corr_pairs, corr_pairs_path, row.names = FALSE)
  write_selection_notes(
    selection_notes = prepared$selection_notes,
    corr_pairs = prepared$corr_pairs,
    path = file.path(output_dir, "feature_selection_notes_v2_r.txt")
  )

  split_idx <- stratified_train_test_split(y, test_size = test_size, seed = random_seed)
  train_idx <- split_idx$train
  test_idx <- split_idx$test

  X_train <- X[train_idx, , drop = FALSE]
  X_test <- X[test_idx, , drop = FALSE]
  y_train <- y[train_idx]
  y_test <- y[test_idx]

  folds <- make_stratified_folds(y_train, n_splits = 5L, seed = random_seed)
  summary_rows <- list()

  for (max_depth in max_depth_values) {
    model <- fit_rf_model(X_train, y_train, max_depth, random_seed)

    train_pred <- predict_labels(model, X_train)
    test_pred <- predict_labels(model, X_test)

    cv_scores <- c()
    for (k in seq_along(folds)) {
      val_idx <- folds[[k]]
      tr_idx <- setdiff(seq_len(nrow(X_train)), val_idx)
      cv_model <- fit_rf_model(
        X_train = X_train[tr_idx, , drop = FALSE],
        y_train = y_train[tr_idx],
        max_depth = max_depth,
        random_seed = random_seed
      )
      val_pred <- predict_labels(cv_model, X_train[val_idx, , drop = FALSE])
      cv_scores <- c(cv_scores, metric_f1(as.character(y_train[val_idx]), val_pred, positive = POSITIVE_CLASS))
    }

    ci <- cv_confidence_interval(cv_scores, confidence = 0.95)
    tag <- depth_tag(max_depth)

    metrics <- data.frame(
      n_samples_total = nrow(X),
      n_features_used = ncol(X),
      test_size = test_size,
      random_seed = random_seed,
      state_policy = "drop_all_state_features",
      max_depth = ifelse(is.na(max_depth), NA, as.integer(max_depth)),
      max_depth_label = tag,
      train_f1 = metric_f1(as.character(y_train), train_pred, positive = POSITIVE_CLASS),
      train_precision = metric_precision(as.character(y_train), train_pred, positive = POSITIVE_CLASS),
      train_recall = metric_recall(as.character(y_train), train_pred, positive = POSITIVE_CLASS),
      holdout_f1 = metric_f1(as.character(y_test), test_pred, positive = POSITIVE_CLASS),
      holdout_precision = metric_precision(as.character(y_test), test_pred, positive = POSITIVE_CLASS),
      holdout_recall = metric_recall(as.character(y_test), test_pred, positive = POSITIVE_CLASS),
      holdout_accuracy = metric_accuracy(as.character(y_test), test_pred),
      cv_f1_mean = mean(cv_scores),
      cv_f1_std = ifelse(length(cv_scores) > 1, sd(cv_scores), 0),
      cv_f1_ci95_lower = ci[["lower"]],
      cv_f1_ci95_upper = ci[["upper"]],
      stringsAsFactors = FALSE
    )

    cv_df <- data.frame(
      fold = seq_along(cv_scores),
      f1_score = cv_scores,
      max_depth_label = tag,
      stringsAsFactors = FALSE
    )
    write.csv(cv_df, file.path(output_dir, sprintf("cv_f1_scores_v2_d%s.csv", tag)), row.names = FALSE)

    importance_df <- data.frame(
      feature = names(model$variable.importance),
      importance = as.numeric(model$variable.importance),
      stringsAsFactors = FALSE
    )
    importance_df <- importance_df[order(importance_df$importance, decreasing = TRUE), , drop = FALSE]
    write.csv(importance_df, file.path(output_dir, sprintf("feature_importances_v2_d%s.csv", tag)), row.names = FALSE)

    save_feature_importance_plot(
      importance_df = importance_df,
      path = file.path(output_dir, sprintf("feature_importance_v2_d%s.png", tag)),
      max_depth = max_depth
    )

    save_confusion_matrix_plot(
      y_true = as.character(y_test),
      y_pred = test_pred,
      path = file.path(output_dir, sprintf("confusion_matrix_v2_d%s.png", tag)),
      max_depth = max_depth
    )

    writeLines(selected_features, con = file.path(output_dir, sprintf("selected_features_v2_d%s.txt", tag)))
    saveRDS(model, file = file.path(output_dir, sprintf("model_v2_d%s.rds", tag)))

    metric_row <- metrics[1, , drop = FALSE]
    write.csv(metric_row, file.path(output_dir, sprintf("metrics_v2_d%s.csv", tag)), row.names = FALSE)
    writeLines(
      to_json_like_text(as.list(metric_row[1, ])),
      con = file.path(output_dir, sprintf("metrics_v2_d%s.json", tag)),
      useBytes = TRUE
    )
    write_metrics_txt(as.list(metric_row[1, ]), file.path(output_dir, sprintf("metrics_v2_d%s.txt", tag)))

    summary_rows[[length(summary_rows) + 1L]] <- metric_row

    message(strrep("-", 50))
    message(sprintf("Model v2 (R) complete (max_depth=%s).", ifelse(is.na(max_depth), "None", as.integer(max_depth))))
    message(sprintf("Features used: %d", length(selected_features)))
    message(sprintf("Train F1: %.6f", metrics$train_f1))
    message(sprintf("Holdout F1: %.6f", metrics$holdout_f1))
    message(sprintf("CV F1 mean (95%% CI): %.6f [%.6f, %.6f]", metrics$cv_f1_mean, ci[["lower"]], ci[["upper"]]))
  }

  summary_df <- do.call(rbind, summary_rows)
  write.csv(summary_df, file.path(output_dir, "metrics_v2_by_max_depth.csv"), row.names = FALSE)
  message(sprintf("Saved summary: %s", file.path(output_dir, "metrics_v2_by_max_depth.csv")))
}

main()
