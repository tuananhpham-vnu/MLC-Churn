# ============================================================
# BAI 3 - THI NGHIEM CRFD
# Completely Randomized Factorial Design
# Factors:
#   1. k = 3, 5, 10
#   2. max_depth = 3, 5, None
# Metric: F1, positive class = yes
# Seed: 1234
# ============================================================
# 
# Cai goi neu chua co
#install.packages(c("fs","sass","tidyverse", "caret", "ranger", "car", "multcompView"),dependencies = TRUE)

library(tidyverse)
library(caret)
library(ranger)
library(car)
library(multcompView)
library(brms)

set.seed(1234)
getwd()
setwd('/media/dainn98/New Volume/Folder F/phamtuananh@23020010/UET.iSEML/2026.DAE.MLC-Churn/MLC-Churn')
# ============================================================
# 1. CONFIG
# ============================================================
output_dir <- "Phase_3_CRFD/outputs"
input_csv <- "mlc_churn.csv"


dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

random_seed <- 1234
k_values <- c(3, 5, 10)
n_repeats <- 10

# Bai 2 chi xet anh huong cua k, nen giu co dinh mo hinh M.
# Theo ket qua bai 1, neu max_depth = None tot nhat thi de NA.
# Neu mo hinh M cua ban chon max_depth = 10 hoac 5, doi thanh 10 hoac 5.
max_depth_values <- c(3, 5, NA)
num_trees_fixed <- 500

depth_label <- function(max_depth) {
  ifelse(is.na(max_depth), "None", as.character(max_depth))
}
num_trees_fixed <- 500

# Bayesian comparison bang brms.
# Luu y: brms can cai Stan backend, nen lan dau chay co the mat thoi gian.
# Neu may chua cai brms/cmdstanr/rstan, co the de FALSE de bo qua phan Bayesian.
use_brms <- TRUE
brms_iter <- 4000
brms_warmup <- 1000
brms_chains <- 4
brms_cores <- min(4, parallel::detectCores())

# ============================================================
# 2. HAM TINH F1 VOI POSITIVE CLASS = yes
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

# ============================================================
# 3. CHUAN BI DU LIEU CHO MO HINH M
#    Mo hinh M dang dung: bo state va cac bien *_charge, giu area_code.
#    Neu file preprocess da bo san cac cot nay thi any_of() khong gay loi.
# ============================================================
prepare_model_M <- function(df) {
  # Target
  if ("churn_binary" %in% names(df)) {
    y <- ifelse(df$churn_binary == 1, "yes", "no")
  } else {
    y <- tolower(as.character(df$churn))
  }

  X <- df %>%
    select(-any_of(c("churn", "churn_binary")))

  # Ma hoa yes/no neu con dang raw
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

  # Mo hinh M: bo state va cac bien cuoc phi *_charge
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
  
  # Dam bao area_code duoc xu ly nhu bien phan loai
  if ("area_code" %in% names(X)) {
    X <- X %>%
      mutate(
        area_code = as.character(area_code),
        area_code = stringr::str_replace(area_code, "^area_code", ""),
        area_code = as.factor(area_code)
      )
  }

  # One-hot encode categorical con lai, vi du area_code neu la factor/character
  X <- model.matrix(~ . - 1, data = X) %>% as.data.frame()

  list(X = X, y = factor(y, levels = c("no", "yes")))
}


# ============================================================
# 4. CHAY REPEATED STRATIFIED K-FOLD
# ============================================================
fit_rf <- function(X_train, y_train, max_depth_current) {
  if (is.na(max_depth_current)) {
    ranger(
      x = X_train,
      y = y_train,
      num.trees = num_trees_fixed,
      seed = random_seed,
      probability = FALSE
    )
  } else {
    ranger(
      x = X_train,
      y = y_train,
      num.trees = num_trees_fixed,
      max.depth = max_depth_current,
      seed = random_seed,
      probability = FALSE
    )
  }
}

run_crd_experiment <- function(X, y, max_depth_current) {
  results <- list()
  row_id <- 1

  for (k in k_values) {
    for (r in 1:n_repeats) {
      # createFolds cua caret la stratified khi y la factor.
      # Doi seed theo repeat/k de cac lan lap co fold khac nhau nhung van tai lap duoc.
      set.seed(random_seed + 1000 * r + k)
      folds <- createFolds(y, k = k, returnTrain = FALSE)

      for (fold_id in seq_along(folds)) {
        val_idx <- folds[[fold_id]]
        train_idx <- setdiff(seq_along(y), val_idx)

        X_train <- X[train_idx, , drop = FALSE]
        y_train <- y[train_idx]
        X_val <- X[val_idx, , drop = FALSE]
        y_val <- y[val_idx]

        rf_model <- fit_rf(X_train, y_train, max_depth_current)
        pred <- predict(rf_model, data = X_val)$predictions
        f1 <- f1_score_yes(y_val, pred)

        results[[row_id]] <- tibble(
          treatment = paste0("k=", k),
          k = factor(paste0("k=", k), levels = paste0("k=", k_values)),
          k_numeric = k,
          max_depth = factor(depth_label(max_depth_current), levels = c("3", "5", "None")),
          repeat_id = r,
          fold_id = fold_id,
          f1 = as.numeric(f1),
          n_train = length(train_idx),
          n_valid = length(val_idx),
          positive_rate_valid = mean(y_val == "yes")
        )
        row_id <- row_id + 1
      }
    }
  }

  bind_rows(results)
}

# ============================================================
# 5. MAIN
# ============================================================
df <- read.csv(input_csv)
data_M <- prepare_model_M(df)
ncol(data_M$X)
colnames(data_M$X)
table(data_M$y)
X <- data_M$X
y <- data_M$y

cat("Dataset size:", nrow(X), "rows,", ncol(X), "features\n")
cat("Class distribution:\n")
print(table(y))
cat("Positive class rate:", mean(y == "yes"), "\n")

all_crd_results <- list()

for (max_depth_current in max_depth_values) {
  
  current_depth_label <- depth_label(max_depth_current)
  current_output_dir <- file.path(output_dir, paste0("max_depth_", current_depth_label))
  
  dir.create(current_output_dir, recursive = TRUE, showWarnings = FALSE)
  
  cat("\n========================================\n")
  cat("Running CRD with max_depth =", current_depth_label, "\n")
  cat("Saving outputs to:", current_output_dir, "\n")
  cat("========================================\n")
  
  crd_results <- run_crd_experiment(X, y, max_depth_current)
  
  all_crd_results[[current_depth_label]] <- crd_results
  
  write.csv(
    crd_results,
    file.path(current_output_dir, "crd_repeated_kfold_f1_raw.csv"),
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
  
  write.csv(
    summary_by_k,
    file.path(current_output_dir, "crd_summary_by_k.csv"),
    row.names = FALSE
  )
  
  print(summary_by_k)
  
  levene_res <- leveneTest(f1 ~ k, data = crd_results, center = median)
  print(levene_res)
  
  capture.output(
    levene_res,
    file = file.path(current_output_dir, "crd_levene_test.txt")
  )
  
  lm_crd <- lm(f1 ~ k, data = crd_results)
  aov_crd <- aov(f1 ~ k, data = crd_results)
  
  lm_summary <- summary(lm_crd)
  aov_summary <- summary(aov_crd)
  coef_ci <- confint(lm_crd, level = 0.95)
  
  print(lm_summary)
  print(aov_summary)
  print(coef_ci)
  
  capture.output(
    lm_summary,
    aov_summary,
    coef_ci,
    file = file.path(current_output_dir, "crd_lm_anova_results.txt")
  )
  
  tukey_res <- TukeyHSD(aov_crd)
  print(tukey_res)
  
  capture.output(
    tukey_res,
    file = file.path(current_output_dir, "crd_tukey_hsd.txt")
  )
  
  tukey_df <- as.data.frame(tukey_res$k) %>%
    rownames_to_column("comparison") %>%
    rename(
      diff = diff,
      lower = lwr,
      upper = upr,
      p_adj = `p adj`
    )
  
  write.csv(
    tukey_df,
    file.path(current_output_dir, "crd_tukey_hsd.csv"),
    row.names = FALSE
  )
  
  letters_list <- multcompLetters4(aov_crd, tukey_res)
  tukey_letters <- as.data.frame.list(letters_list$k)
  tukey_letters$k <- rownames(tukey_letters)
  colnames(tukey_letters)[1] <- "letters"
  
  summary_plot <- summary_by_k %>%
    left_join(tukey_letters, by = "k")
  
  write.csv(
    summary_plot,
    file.path(current_output_dir, "crd_tukey_letters_by_k.csv"),
    row.names = FALSE
  )
  
  p1 <- ggplot(crd_results, aes(x = k, y = f1)) +
    geom_boxplot(width = 0.45, outlier.alpha = 0.35) +
    geom_jitter(width = 0.08, alpha = 0.35, size = 1.4) +
    geom_point(
      data = summary_plot,
      aes(x = k, y = mean_f1),
      inherit.aes = FALSE,
      size = 3
    ) +
    geom_errorbar(
      data = summary_plot,
      aes(x = k, ymin = ci_lower, ymax = ci_upper),
      inherit.aes = FALSE,
      width = 0.12,
      linewidth = 0.8
    ) +
    geom_text(
      data = summary_plot,
      aes(x = k, y = ci_upper + 0.02, label = letters),
      inherit.aes = FALSE,
      size = 5
    ) +
    labs(
      title = paste0("CRD: So sanh F1 giua cac gia tri k, max_depth = ", current_depth_label),
      subtitle = "Diem la mean F1, thanh loi la 95% CI; chu cai khac nhau nghia la khac biet co y nghia theo TukeyHSD",
      x = "So fold k",
      y = "F1 score, positive class = yes"
    ) +
    theme_minimal(base_size = 13)
  
  ggsave(
    filename = file.path(current_output_dir, "crd_group_comparison_tukey_letters.png"),
    plot = p1,
    width = 8,
    height = 5,
    dpi = 300
  )
  
  p2 <- ggplot(tukey_df, aes(x = comparison, y = diff)) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    geom_point(size = 3) +
    geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.15, linewidth = 0.8) +
    coord_flip() +
    labs(
      title = paste0("TukeyHSD pairwise comparison for k, max_depth = ", current_depth_label),
      subtitle = "Khoang tin cay khong cat 0 => khac biet co y nghia thong ke",
      x = "Cap so sanh",
      y = "Chenh lech trung binh F1"
    ) +
    theme_minimal(base_size = 13)
  
  ggsave(
    filename = file.path(current_output_dir, "crd_tukey_pairwise_ci.png"),
    plot = p2,
    width = 8,
    height = 4.8,
    dpi = 300
  )
}

crfd_results <- bind_rows(all_crd_results)

write.csv(
  crfd_results,
  file.path(output_dir, "crfd_all_max_depth_results.csv"),
  row.names = FALSE
)

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

write.csv(
  summary_crfd,
  file.path(output_dir, "crfd_summary_by_k_max_depth.csv"),
  row.names = FALSE
)

print(summary_crfd)

lm_crfd <- lm(f1 ~ k * max_depth, data = crfd_results)
aov_crfd <- aov(f1 ~ k * max_depth, data = crfd_results)

lm_crfd_summary <- summary(lm_crfd)
aov_crfd_summary <- summary(aov_crfd)
coef_crfd_ci <- confint(lm_crfd, level = 0.95)

print(lm_crfd_summary)
print(aov_crfd_summary)
print(coef_crfd_ci)

capture.output(
  lm_crfd_summary,
  aov_crfd_summary,
  coef_crfd_ci,
  file = file.path(output_dir, "crfd_lm_anova_interaction_results.txt")
)



p_interaction <- ggplot(
  summary_crfd,
  aes(x = k, y = mean_f1, group = max_depth, color = max_depth)
) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), width = 0.12) +
  labs(
    title = "CRFD: Interaction plot giua k va max_depth",
    subtitle = "Neu cac duong khong song song, co dau hieu tuong tac giua hai yeu to",
    x = "So fold k",
    y = "Mean F1 score",
    color = "max_depth"
  ) +
  theme_minimal(base_size = 13)

ggsave(
  filename = file.path(output_dir, "crfd_interaction_plot_k_max_depth.png"),
  plot = p_interaction,
  width = 8,
  height = 5,
  dpi = 300
)


# ============================================================
# 7. CRFD BAYESIAN ANALYSIS BANG brm()
#    Kiem tra anh huong cua k, max_depth va tuong tac k:max_depth
# ============================================================

if (use_brms) {
  
  cat("\n========================================\n")
  cat("Running Bayesian CRFD model with brm()\n")
  cat("Formula: f1 ~ k * max_depth\n")
  cat("========================================\n")
  
  brm_crfd <- brm(
    formula = f1 ~ k * max_depth,
    data = crfd_results,
    family = gaussian(),
    iter = brms_iter,
    warmup = brms_warmup,
    chains = brms_chains,
    cores = brms_cores,
    seed = random_seed
  )
  
  brm_crfd_summary <- summary(brm_crfd)
  print(brm_crfd_summary)
  
  capture.output(
    brm_crfd_summary,
    file = file.path(output_dir, "crfd_brm_summary.txt")
  )
  
  # Luu model de lan sau doc lai, khong can fit lai
  saveRDS(
    brm_crfd,
    file = file.path(output_dir, "crfd_brm_model.rds")
  )
  
  # Lay bang he so Bayesian
  brm_fixed_effects <- fixef(brm_crfd) %>%
    as.data.frame() %>%
    rownames_to_column("term")
  
  write.csv(
    brm_fixed_effects,
    file.path(output_dir, "crfd_brm_fixed_effects.csv"),
    row.names = FALSE
  )
  
  print(brm_fixed_effects)
  
  # Ve conditional effects
  brm_conditional_effects <- conditional_effects(brm_crfd)
  
  png(
    filename = file.path(output_dir, "crfd_brm_conditional_effects.png"),
    width = 1200,
    height = 800
  )
  plot(brm_conditional_effects, ask = FALSE)
  dev.off()
}
