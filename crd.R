# ============================================================
# BAI 2 - THI NGHIEM CRD
# Repeated stratified k-fold Random Forest for mlc_churn
# Factor: k = 3, 5, 10; repeat = 10
# Fixed max_depth = None
# Metric: F1, positive class = yes
# Seed: 1234
# Output: results/crd_results.csv and related analysis files
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
max_depth_fixed <- NA          # NA = None
num_trees_fixed <- 500
num_threads_fixed <- 1         # giup ket qua on dinh hon

use_brms <- TRUE               # neu khong cai brms, script se tu bo qua va ghi crd_brm.txt
brms_iter <- 4000
brms_warmup <- 1000
brms_chains <- 4
brms_cores <- min(4, parallel::detectCores())

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

  # Neu co cap *_no va *_yes thi bo *_no de tranh du thua
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

fit_rf <- function(X_train, y_train) {
  if (is.na(max_depth_fixed)) {
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
      max.depth = max_depth_fixed,
      seed = random_seed,
      probability = FALSE,
      num.threads = num_threads_fixed
    )
  }
}

run_crd_experiment <- function(X, y) {
  results <- list()
  row_id <- 1

  for (k in k_values) {
    for (r in 1:n_repeats) {
      set.seed(random_seed + 1000 * r + k)
      folds <- createFolds(y, k = k, returnTrain = FALSE)

      for (fold_id in seq_along(folds)) {
        val_idx <- folds[[fold_id]]
        train_idx <- setdiff(seq_along(y), val_idx)

        rf_model <- fit_rf(X[train_idx, , drop = FALSE], y[train_idx])
        pred <- predict(rf_model, data = X[val_idx, , drop = FALSE])$predictions
        f1 <- f1_score_yes(y[val_idx], pred)

        results[[row_id]] <- tibble(
          treatment = paste0("k=", k),
          k = factor(paste0("k=", k), levels = paste0("k=", k_values)),
          k_numeric = k,
          max_depth = "None",
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

crd_results <- run_crd_experiment(X, y)

write.csv(
  crd_results,
  file.path(output_dir, "crd_results.csv"),
  row.names = FALSE
)

summary_by_k <- crd_results %>%
  group_by(k) %>%
  summarise(
    n = n(),
    mean_f1 = mean(f1),
    sd_f1 = sd(f1),
    ci_lower = ci95(f1)["lower"],
    ci_upper = ci95(f1)["upper"],
    .groups = "drop"
  )

write.csv(summary_by_k, file.path(output_dir, "crd_summary_by_k.csv"), row.names = FALSE)
print(summary_by_k)

# ============================================================
# 4. LEVENE TEST
# ============================================================
levene_res <- leveneTest(f1 ~ k, data = crd_results, center = median)
print(levene_res)

capture.output(
  cat("Levene's Test for CRD: f1 ~ k\n\n"),
  levene_res,
  file = file.path(output_dir, "crd_levene.txt")
)

# ============================================================
# 5. LM / ANOVA
# ============================================================
lm_crd <- lm(f1 ~ k, data = crd_results)
aov_crd <- aov(f1 ~ k, data = crd_results)
lm_summary <- summary(lm_crd)
aov_summary <- summary(aov_crd)
coef_ci <- confint(lm_crd, level = 0.95)

print(lm_summary)
print(aov_summary)
print(coef_ci)

capture.output(
  cat("Linear model and ANOVA for CRD\n"),
  cat("Formula: lm(f1 ~ k)\n\n"),
  lm_summary,
  cat("\nANOVA table:\n"),
  aov_summary,
  cat("\n95% confidence intervals for coefficients:\n"),
  coef_ci,
  file = file.path(output_dir, "crd_lm.txt")
)

# ============================================================
# 6. TUKEY HSD
# ============================================================
tukey_res <- TukeyHSD(aov_crd)
print(tukey_res)

capture.output(
  cat("TukeyHSD pairwise comparison for CRD\n\n"),
  tukey_res,
  file = file.path(output_dir, "crd_tukey.txt")
)

tukey_df <- as.data.frame(tukey_res$k) %>%
  rownames_to_column("comparison") %>%
  rename(diff = diff, lower = lwr, upper = upr, p_adj = `p adj`)

write.csv(tukey_df, file.path(output_dir, "crd_tukey.csv"), row.names = FALSE)

letters_list <- multcompLetters4(aov_crd, tukey_res)
tukey_letters <- as.data.frame.list(letters_list$k)
tukey_letters$k <- rownames(tukey_letters)
colnames(tukey_letters)[1] <- "letters"

summary_plot <- summary_by_k %>% left_join(tukey_letters, by = "k")
write.csv(summary_plot, file.path(output_dir, "crd_tukey_letters_by_k.csv"), row.names = FALSE)

# ============================================================
# 7. OPTIONAL BRM
# ============================================================
if (use_brms) {
  if (!requireNamespace("brms", quietly = TRUE)) {
    writeLines(
      c(
        "Package brms chua duoc cai nen bo qua phan brm().",
        "Neu muon chay Bayesian: install.packages('brms')",
        "Mo hinh du kien: brm(f1 ~ k)."
      ),
      con = file.path(output_dir, "crd_brm.txt")
    )
  } else {
    library(brms)
    set.seed(random_seed)

    brm_crd <- brm(
      formula = f1 ~ k,
      data = crd_results,
      family = gaussian(),
      prior = c(
        prior(normal(0.7, 0.3), class = "Intercept"),
        prior(normal(0, 0.2), class = "b"),
        prior(exponential(10), class = "sigma")
      ),
      chains = brms_chains,
      iter = brms_iter,
      warmup = brms_warmup,
      cores = brms_cores,
      seed = random_seed,
      refresh = 0
    )

    brm_summary <- summary(brm_crd)
    print(brm_summary)

    capture.output(
      cat("Bayesian model for CRD\n"),
      cat("Formula: brm(f1 ~ k)\n\n"),
      brm_summary,
      file = file.path(output_dir, "crd_brm.txt")
    )

    saveRDS(brm_crd, file.path(output_dir, "crd_brm_model.rds"))
  }
} else {
  writeLines("use_brms <- FALSE, bo qua phan brm(f1 ~ k).", file.path(output_dir, "crd_brm.txt"))
}

# ============================================================
# 8. PLOTS
# ============================================================
p1 <- ggplot(crd_results, aes(x = k, y = f1)) +
  geom_boxplot(width = 0.45, outlier.alpha = 0.35) +
  geom_jitter(width = 0.08, alpha = 0.35, size = 1.4) +
  geom_point(data = summary_plot, aes(x = k, y = mean_f1), inherit.aes = FALSE, size = 3) +
  geom_errorbar(data = summary_plot, aes(x = k, ymin = ci_lower, ymax = ci_upper), inherit.aes = FALSE, width = 0.12, linewidth = 0.8) +
  geom_text(data = summary_plot, aes(x = k, y = ci_upper + 0.02, label = letters), inherit.aes = FALSE, size = 5) +
  labs(
    title = "CRD: So sanh F1 giua cac gia tri k bang TukeyHSD",
    subtitle = "Diem la mean F1, thanh loi la 95% CI; chu cai khac nhau nghia la khac biet co y nghia",
    x = "So fold k",
    y = "F1 score, positive class = yes"
  ) +
  theme_minimal(base_size = 13)

ggsave(file.path(figure_dir, "crd_group_comparison_tukey_letters.png"), p1, width = 8, height = 5, dpi = 300)

p2 <- ggplot(tukey_df, aes(x = comparison, y = diff)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.15, linewidth = 0.8) +
  coord_flip() +
  labs(
    title = "TukeyHSD pairwise comparison for k",
    subtitle = "Khoang tin cay khong cat 0 => khac biet co y nghia thong ke",
    x = "Cap so sanh",
    y = "Chenh lech trung binh F1"
  ) +
  theme_minimal(base_size = 13)

ggsave(file.path(figure_dir, "crd_tukey_pairwise_ci.png"), p2, width = 8, height = 4.8, dpi = 300)

cat("\nDone. Main CRD files saved to:", output_dir, "\n")
cat("- crd_results.csv\n")
cat("- crd_levene.txt\n")
cat("- crd_lm.txt\n")
cat("- crd_tukey.txt\n")
cat("- crd_brm.txt\n")
