# ============================================================
# BAI 3 - THI NGHIEM CRFD
# Completely Randomized Factorial Design
# Factors: k = 3, 5, 10 and max_depth = 3, 5, None
# Metric: F1, positive class = yes
# Seed: 1234
# Output: results/crfd_results.csv and related analysis files
# ============================================================

# install.packages(c("tidyverse", "caret", "ranger", "car", "multcompView"), dependencies = TRUE)
# install.packages("brms") # optional

library(tidyverse)
library(caret)
library(ranger)
library(car)
library(multcompView)

set.seed(1234)

# ============================================================
# 1. CONFIG
# ============================================================
input_csv <- "Phase_0_EDA/outputs/preprocessed_improved.csv"
if (!file.exists(input_csv)) {
  input_csv <- "mlc_churn.csv"
}

output_dir <- "results"
figure_dir <- file.path(output_dir, "figures")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

random_seed <- 1234
k_values <- c(3, 5, 10)
n_repeats <- 10
max_depth_values <- c(3, 5, NA)  # NA = None
num_trees_fixed <- 500
num_threads_fixed <- 1           # giup ket qua on dinh hon

use_brms <- TRUE                 # neu khong cai brms, script se tu bo qua va ghi crfd_brm.txt
brms_iter <- 4000
brms_warmup <- 1000
brms_chains <- 4
brms_cores <- min(4, parallel::detectCores())

depth_label <- function(max_depth) {
  ifelse(is.na(max_depth), "None", as.character(max_depth))
}

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================
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
  m <- mean(x)
  if (n < 2) return(c(lower = m, upper = m))
  se <- sd(x) / sqrt(n)
  margin <- qt(0.975, df = n - 1) * se
  c(lower = m - margin, upper = m + margin)
}

prepare_model_M <- function(df) {
  if ("churn_binary" %in% names(df)) {
    y <- ifelse(df$churn_binary == 1, "yes", "no")
  } else {
    y <- tolower(as.character(df$churn))
  }

  X <- df %>% select(-any_of(c("churn", "churn_binary")))

  if ("international_plan" %in% names(X)) {
    X <- X %>%
      mutate(international_plan_yes = ifelse(tolower(as.character(international_plan)) == "yes", 1, 0)) %>%
      select(-international_plan)
  }

  if ("voice_mail_plan" %in% names(X)) {
    X <- X %>%
      mutate(voice_mail_plan_yes = ifelse(tolower(as.character(voice_mail_plan)) == "yes", 1, 0)) %>%
      select(-voice_mail_plan)
  }

  # Model M: bo state va cac bien cuoc phi *_charge, giu area_code
  X <- X %>%
    select(-any_of("state")) %>%
    select(-starts_with("state_")) %>%
    select(-matches("_charge$"))

  no_cols <- names(X)[stringr::str_detect(names(X), "_no$")]
  for (col in no_cols) {
    yes_col <- stringr::str_replace(col, "_no$", "_yes")
    if (yes_col %in% names(X)) {
      X <- X %>% select(-all_of(col))
    }
  }

  if ("area_code" %in% names(X)) {
    X <- X %>%
      mutate(
        area_code = as.character(area_code),
        area_code = stringr::str_replace(area_code, "^area_code", ""),
        area_code = as.factor(area_code)
      )
  }

  X <- model.matrix(~ . - 1, data = X) %>% as.data.frame()
  list(X = X, y = factor(y, levels = c("no", "yes")))
}

fit_rf <- function(X_train, y_train, max_depth_current) {
  if (is.na(max_depth_current)) {
    ranger(
      x = X_train,
      y = y_train,
      num.trees = num_trees_fixed,
      seed = random_seed,
      probability = FALSE,
      num.threads = num_threads_fixed
    )
  } else {
    ranger(
      x = X_train,
      y = y_train,
      num.trees = num_trees_fixed,
      max.depth = max_depth_current,
      seed = random_seed,
      probability = FALSE,
      num.threads = num_threads_fixed
    )
  }
}

run_crfd_for_depth <- function(X, y, max_depth_current) {
  results <- list()
  row_id <- 1

  for (k in k_values) {
    for (r in 1:n_repeats) {
      set.seed(random_seed + 1000 * r + k)
      folds <- createFolds(y, k = k, returnTrain = FALSE)

      for (fold_id in seq_along(folds)) {
        val_idx <- folds[[fold_id]]
        train_idx <- setdiff(seq_along(y), val_idx)

        rf_model <- fit_rf(X[train_idx, , drop = FALSE], y[train_idx], max_depth_current)
        pred <- predict(rf_model, data = X[val_idx, , drop = FALSE])$predictions
        f1 <- f1_score_yes(y[val_idx], pred)

        results[[row_id]] <- tibble(
          treatment = paste0("k=", k, "_depth=", depth_label(max_depth_current)),
          k = factor(paste0("k=", k), levels = paste0("k=", k_values)),
          k_numeric = k,
          max_depth = factor(depth_label(max_depth_current), levels = c("3", "5", "None")),
          repeat_id = r,
          fold_id = fold_id,
          f1 = as.numeric(f1),
          n_train = length(train_idx),
          n_valid = length(val_idx),
          positive_rate_valid = mean(y[val_idx] == "yes")
        )
        row_id <- row_id + 1
      }
    }
  }

  bind_rows(results)
}

# ============================================================
# 3. MAIN EXPERIMENT
# ============================================================
cat("Input file:", input_csv, "\n")
df <- read.csv(input_csv)
data_M <- prepare_model_M(df)
X <- data_M$X
y <- data_M$y

cat("Dataset size:", nrow(X), "rows,", ncol(X), "features\n")
cat("Class distribution:\n")
print(table(y))

all_results <- list()
for (max_depth_current in max_depth_values) {
  current_depth_label <- depth_label(max_depth_current)
  cat("\nRunning CRFD cells with max_depth =", current_depth_label, "\n")
  all_results[[current_depth_label]] <- run_crfd_for_depth(X, y, max_depth_current)
}

crfd_results <- bind_rows(all_results)
write.csv(crfd_results, file.path(output_dir, "crfd_results.csv"), row.names = FALSE)

summary_crfd <- crfd_results %>%
  group_by(k, max_depth) %>%
  summarise(
    n = n(),
    mean_f1 = mean(f1),
    sd_f1 = sd(f1),
    ci_lower = ci95(f1)["lower"],
    ci_upper = ci95(f1)["upper"],
    .groups = "drop"
  )

write.csv(summary_crfd, file.path(output_dir, "crfd_summary_by_k_max_depth.csv"), row.names = FALSE)
print(summary_crfd)

# ============================================================
# 4. LEVENE TEST
# ============================================================
crfd_results <- crfd_results %>%
  mutate(cell = interaction(k, max_depth, sep = ":"))

levene_9_cells <- leveneTest(f1 ~ cell, data = crfd_results, center = median)
levene_by_depth <- crfd_results %>%
  group_by(max_depth) %>%
  group_map(~ {
    res <- leveneTest(f1 ~ k, data = .x, center = median)
    tibble(
      max_depth = as.character(.y$max_depth),
      df = res$Df[1],
      f_value = res$`F value`[1],
      p_value = res$`Pr(>F)`[1]
    )
  }) %>%
  bind_rows()

print(levene_9_cells)
print(levene_by_depth)

capture.output(
  cat("Levene's Test for CRFD\n"),
  cat("1) Compare variance across 9 treatment cells k:max_depth\n\n"),
  levene_9_cells,
  cat("\n2) Compare variance across k within each max_depth\n\n"),
  levene_by_depth,
  file = file.path(output_dir, "crfd_levene.txt")
)

write.csv(levene_by_depth, file.path(output_dir, "crfd_levene_by_depth.csv"), row.names = FALSE)

# ============================================================
# 5. LM / ANOVA FOR CRFD
# ============================================================
lm_crfd <- lm(f1 ~ k * max_depth, data = crfd_results)
aov_crfd <- aov(f1 ~ k * max_depth, data = crfd_results)
lm_crfd_summary <- summary(lm_crfd)
aov_crfd_summary <- summary(aov_crfd)
coef_crfd_ci <- confint(lm_crfd, level = 0.95)

print(lm_crfd_summary)
print(aov_crfd_summary)
print(coef_crfd_ci)

capture.output(
  cat("Linear model and ANOVA for CRFD\n"),
  cat("Formula: lm(f1 ~ k * max_depth)\n\n"),
  lm_crfd_summary,
  cat("\nANOVA table:\n"),
  aov_crfd_summary,
  cat("\n95% confidence intervals for coefficients:\n"),
  coef_crfd_ci,
  file = file.path(output_dir, "crfd_lm.txt")
)

# Save coefficient table as CSV for convenience
coef_table <- as.data.frame(coef_crfd_ci) %>%
  rownames_to_column("term") %>%
  rename(ci_lower = `2.5 %`, ci_upper = `97.5 %`)
coef_table$estimate <- coef(lm_crfd)
coef_table <- coef_table %>% select(term, estimate, ci_lower, ci_upper)
write.csv(coef_table, file.path(output_dir, "crfd_lm_coefficients_ci.csv"), row.names = FALSE)

# ============================================================
# 6. OPTIONAL BRM
# ============================================================
if (use_brms) {
  if (!requireNamespace("brms", quietly = TRUE)) {
    writeLines(
      c(
        "Package brms chua duoc cai nen bo qua phan brm().",
        "Neu muon chay Bayesian: install.packages('brms')",
        "Mo hinh du kien: brm(f1 ~ k * max_depth)."
      ),
      con = file.path(output_dir, "crfd_brm.txt")
    )
  } else {
    library(brms)
    set.seed(random_seed)

    brm_crfd <- brm(
      formula = f1 ~ k * max_depth,
      data = crfd_results,
      family = gaussian(),
      chains = brms_chains,
      iter = brms_iter,
      warmup = brms_warmup,
      cores = brms_cores,
      seed = random_seed,
      refresh = 0
    )

    brm_crfd_summary <- summary(brm_crfd)
    print(brm_crfd_summary)

    capture.output(
      cat("Bayesian model for CRFD\n"),
      cat("Formula: brm(f1 ~ k * max_depth)\n\n"),
      brm_crfd_summary,
      file = file.path(output_dir, "crfd_brm.txt")
    )

    saveRDS(brm_crfd, file.path(output_dir, "crfd_brm_model.rds"))

    brm_fixed_effects <- fixef(brm_crfd) %>%
      as.data.frame() %>%
      rownames_to_column("term")
    write.csv(brm_fixed_effects, file.path(output_dir, "crfd_brm_fixed_effects.csv"), row.names = FALSE)
  }
} else {
  writeLines("use_brms <- FALSE, bo qua phan brm(f1 ~ k * max_depth).", file.path(output_dir, "crfd_brm.txt"))
}

# ============================================================
# 7. INTERACTION PLOT
# ============================================================
p_interaction <- ggplot(summary_crfd, aes(x = k, y = mean_f1, group = max_depth, color = max_depth)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), width = 0.12) +
  labs(
    title = "CRFD: Interaction plot giua k va max_depth",
    subtitle = "Cac duong khong song song cho thay co tuong tac giua hai yeu to",
    x = "So fold k",
    y = "Mean F1 score",
    color = "max_depth"
  ) +
  theme_minimal(base_size = 13)

ggsave(file.path(figure_dir, "crfd_interaction_plot_k_max_depth.png"), p_interaction, width = 8, height = 5, dpi = 300)

cat("\nDone. Main CRFD files saved to:", output_dir, "\n")
cat("- crfd_results.csv\n")
cat("- crfd_levene.txt\n")
cat("- crfd_lm.txt\n")
cat("- crfd_brm.txt\n")
