# Phase 1 analysis with lm() and optional brm()
# Usage examples:
# Rscript Phase_1_Model/01_model_analysis_lm.R
# Rscript Phase_1_Model/01_model_analysis_lm.R --output_dir Phase_1_Model/outputs --use_brm TRUE

parse_args <- function(args) {
  cfg <- list(
    output_dir = "Phase_1_Model/outputs",
    baseline_cv = "cv_f1_scores_baseline.csv",
    improved_cv = "cv_f1_scores_improved.csv",
    seed = 1234L,
    use_brm = FALSE
  )

  if (length(args) == 0) return(cfg)
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (i == length(args)) break
    value <- args[[i + 1]]
    if (key == "--output_dir") cfg$output_dir <- value
    if (key == "--baseline_cv") cfg$baseline_cv <- value
    if (key == "--improved_cv") cfg$improved_cv <- value
    if (key == "--seed") cfg$seed <- as.integer(value)
    if (key == "--use_brm") cfg$use_brm <- tolower(value) %in% c("true", "1", "yes")
    i <- i + 2
  }
  cfg
}

resolve_rel_path <- function(root_dir, path_value) {
  if (grepl("^([A-Za-z]:|/)", path_value)) return(path_value)

  candidates <- c(file.path(root_dir, path_value))
  if (basename(root_dir) == "Phase_1_Model" && startsWith(path_value, "Phase_1_Model/")) {
    candidates <- c(file.path(root_dir, sub("^Phase_1_Model/", "", path_value)), candidates)
  }

  for (candidate in candidates) {
    if (dir.exists(candidate) || file.exists(candidate)) return(candidate)
  }
  candidates[[1]]
}

read_cv_file <- function(path, model_name) {
  if (!file.exists(path)) return(NULL)
  df <- read.csv(path, stringsAsFactors = FALSE)
  required <- c("fold", "f1")
  if (!all(required %in% names(df))) {
    stop(sprintf("Missing required columns in %s. Expected: fold, f1", path))
  }
  df$model <- model_name
  df
}

safe_ci95 <- function(x) {
  n <- length(x)
  m <- mean(x)
  if (n < 2) return(c(mean = m, ci95_low = m, ci95_high = m))
  se <- sd(x) / sqrt(n)
  tcrit <- qt(0.975, df = n - 1)
  c(mean = m, ci95_low = m - tcrit * se, ci95_high = m + tcrit * se)
}

format_lm_summary <- function(model_fit, model_type) {
  lines <- c(
    sprintf("LM summary (%s)", model_type),
    paste(rep("=", 40), collapse = ""),
    capture.output(summary(model_fit)),
    "",
    "95% CI for coefficients",
    paste(rep("-", 40), collapse = ""),
    capture.output(confint(model_fit))
  )
  lines
}

run_optional_brm <- function(cv_df, output_dir, seed) {
  if (!requireNamespace("brms", quietly = TRUE)) {
    message("brms package not installed, skip brm() comparison.")
    return(invisible(NULL))
  }

  if (!("model" %in% names(cv_df)) || length(unique(cv_df$model)) < 2) {
    message("Need at least 2 model groups for brm(f1 ~ model). Skip brm().")
    return(invisible(NULL))
  }

  cv_df$model <- factor(cv_df$model)
  fit_brm <- brms::brm(
    formula = f1 ~ model,
    data = cv_df,
    family = gaussian(),
    seed = seed,
    chains = 2,
    iter = 2000,
    cores = 2,
    refresh = 0
  )

  summary_lines <- capture.output(summary(fit_brm))
  writeLines(summary_lines, con = file.path(output_dir, "brm_phase1_summary.txt"))

  draws <- as.data.frame(brms::posterior_summary(fit_brm))
  draws$parameter <- rownames(draws)
  rownames(draws) <- NULL
  write.csv(draws, file.path(output_dir, "brm_phase1_posterior_summary.csv"), row.names = FALSE)
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  cfg <- parse_args(args)
  root_dir <- normalizePath(".", winslash = "/", mustWork = TRUE)
  output_dir <- resolve_rel_path(root_dir, cfg$output_dir)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  baseline_path <- file.path(output_dir, cfg$baseline_cv)
  improved_path <- file.path(output_dir, cfg$improved_cv)

  baseline_df <- read_cv_file(baseline_path, "baseline")
  improved_df <- read_cv_file(improved_path, "improved")

  if (is.null(baseline_df) && is.null(improved_df)) {
    stop("No CV files found. Run Python scripts first to generate cv_f1_scores_*.csv")
  }

  cv_df <- rbind(baseline_df, improved_df)
  cv_df$model <- factor(cv_df$model, levels = unique(cv_df$model))
  write.csv(cv_df, file.path(output_dir, "phase1_cv_f1_combined.csv"), row.names = FALSE)

  # Descriptive statistics with 95% CI by model
  split_list <- split(cv_df$f1, cv_df$model)
  stats_df <- do.call(
    rbind,
    lapply(names(split_list), function(name) {
      vals <- split_list[[name]]
      ci <- safe_ci95(vals)
      data.frame(
        model = name,
        n = length(vals),
        mean_f1 = unname(ci[["mean"]]),
        sd_f1 = ifelse(length(vals) > 1, sd(vals), 0),
        ci95_low = unname(ci[["ci95_low"]]),
        ci95_high = unname(ci[["ci95_high"]])
      )
    })
  )
  write.csv(stats_df, file.path(output_dir, "phase1_cv_f1_summary.csv"), row.names = FALSE)

  # lm(): compare groups if both exist; otherwise intercept-only model
  if (length(unique(cv_df$model)) >= 2) {
    lm_fit <- lm(f1 ~ model, data = cv_df)
    anova_df <- anova(lm_fit)
    coef_df <- as.data.frame(summary(lm_fit)$coefficients)
    coef_df$term <- rownames(coef_df)
    rownames(coef_df) <- NULL
    ci_df <- as.data.frame(confint(lm_fit))
    ci_df$term <- rownames(ci_df)
    rownames(ci_df) <- NULL
    names(ci_df)[1:2] <- c("ci95_low", "ci95_high")
    coef_ci <- merge(coef_df, ci_df, by = "term")

    write.csv(coef_ci, file.path(output_dir, "lm_phase1_coefficients.csv"), row.names = FALSE)
    write.csv(anova_df, file.path(output_dir, "lm_phase1_anova.csv"), row.names = TRUE)
    writeLines(format_lm_summary(lm_fit, "f1 ~ model"), con = file.path(output_dir, "lm_phase1_summary.txt"))
  } else {
    lm_fit <- lm(f1 ~ 1, data = cv_df)
    coef_df <- as.data.frame(summary(lm_fit)$coefficients)
    coef_df$term <- rownames(coef_df)
    rownames(coef_df) <- NULL
    ci_df <- as.data.frame(confint(lm_fit))
    ci_df$term <- rownames(ci_df)
    rownames(ci_df) <- NULL
    names(ci_df)[1:2] <- c("ci95_low", "ci95_high")
    coef_ci <- merge(coef_df, ci_df, by = "term")
    write.csv(coef_ci, file.path(output_dir, "lm_phase1_coefficients.csv"), row.names = FALSE)
    writeLines(format_lm_summary(lm_fit, "f1 ~ 1"), con = file.path(output_dir, "lm_phase1_summary.txt"))
  }

  # Save a simple plot from base R for compatibility
  png(file.path(output_dir, "phase1_cv_f1_boxplot.png"), width = 900, height = 600)
  boxplot(f1 ~ model, data = cv_df, col = c("#4C78A8", "#F58518"), main = "Phase 1 CV F1 by model", ylab = "F1")
  stripchart(f1 ~ model, data = cv_df, vertical = TRUE, method = "jitter", pch = 16, cex = 0.8, add = TRUE)
  dev.off()

  if (isTRUE(cfg$use_brm)) {
    run_optional_brm(cv_df, output_dir, cfg$seed)
  }

  cat("Phase 1 lm() analysis completed.\n")
  cat("Output directory:", output_dir, "\n")
}

main()
