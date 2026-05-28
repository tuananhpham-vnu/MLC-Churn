# ============================================================
# BAI 2 - THI NGHIEM CRD
# Repeated stratified k-fold Random Forest for mlc_churn
# Factor: k = 3, 5, 10; repeat = 10
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
setwd('D:/Folder F/phamtuananh@23020010/UET.iSEML/2026.DAE.MLC-Churn/MLC-Churn')
# ============================================================
# 1. CONFIG
# ============================================================
# Sua duong dan neu can. Neu chay trong project cua ban, nen dung file preprocess cua Phase_0_EDA.
input_csv <- "Phase_0_EDA/outputs/preprocessed_improved.csv"
output_dir <- "Phase_2_CRD/outputs"

# Neu file tren khong ton tai, dung file raw cung thu muc voi script.
if (!file.exists(input_csv)) {
  input_csv <- "mlc_churn.csv"
}
input_csv

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

random_seed <- 1234
k_values <- c(3, 5, 10)
n_repeats <- 10

# Bai 2 chi xet anh huong cua k, nen giu co dinh mo hinh M.
# Theo ket qua bai 1, neu max_depth = None tot nhat thi de NA.
# Neu mo hinh M cua ban chon max_depth = 10 hoac 5, doi thanh 10 hoac 5.
max_depth_fixed <- 3
num_trees_fixed <- 500

# Bayesian comparison bang brms.
# Luu y: brms can cai Stan backend, nen lan dau chay co the mat thoi gian.
# Neu may chua cai brms/cmdstanr/rstan, co the de FALSE de bo qua phan Bayesian.
use_brms <- FALSE
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
fit_rf <- function(X_train, y_train) {
  if (is.na(max_depth_fixed)) {
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
      max.depth = max_depth_fixed,
      seed = random_seed,
      probability = FALSE
    )
  }
}

run_crd_experiment <- function(X, y) {
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

        rf_model <- fit_rf(X_train, y_train)
        pred <- predict(rf_model, data = X_val)$predictions
        f1 <- f1_score_yes(y_val, pred)

        results[[row_id]] <- tibble(
          treatment = paste0("k=", k),
          k = factor(paste0("k=", k), levels = paste0("k=", k_values)),
          k_numeric = k,
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

crd_results <- run_crd_experiment(X, y)

write.csv(
  crd_results,
  file.path(output_dir, "crd_repeated_kfold_f1_raw.csv"),
  row.names = FALSE
)

# ============================================================
# 6. THONG KE MO TA: MEAN + 95% CI
# ============================================================
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
  file.path(output_dir, "crd_summary_by_k.csv"),
  row.names = FALSE
)

print(summary_by_k)

# ============================================================
# 7. LEVENE TEST: SO SANH PHUONG SAI F1 GIUA CAC MUC k
# ============================================================
levene_res <- leveneTest(f1 ~ k, data = crd_results, center = median)
print(levene_res)

capture.output(
  levene_res,
  file = file.path(output_dir, "crd_levene_test.txt")
)

# ============================================================
# 8. LM / ANOVA: DANH GIA ANH HUONG CUA k DEN F1
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
  lm_summary,
  aov_summary,
  coef_ci,
  file = file.path(output_dir, "crd_lm_anova_results.txt")
)

# ============================================================
# 9. TUKEY HSD: SO SANH TUNG CAP k
# ============================================================
tukey_res <- TukeyHSD(aov_crd)
print(tukey_res)

capture.output(
  tukey_res,
  file = file.path(output_dir, "crd_tukey_hsd.txt")
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
  file.path(output_dir, "crd_tukey_hsd.csv"),
  row.names = FALSE
)

# Lay compact letter display tu TukeyHSD de ve do thi 3 nhom
letters_list <- multcompLetters4(aov_crd, tukey_res)
tukey_letters <- as.data.frame.list(letters_list$k)
tukey_letters$k <- rownames(tukey_letters)
colnames(tukey_letters)[1] <- "letters"

summary_plot <- summary_by_k %>%
  left_join(tukey_letters, by = "k")

write.csv(
  summary_plot,
  file.path(output_dir, "crd_tukey_letters_by_k.csv"),
  row.names = FALSE
)

# ============================================================
# 10. BRM BAYESIAN MODEL: SO SANH VOI LM (KHONG BAT BUOC)
# ============================================================
# Mo hinh Bayesian tuong ung voi lm(f1 ~ k):
# f1_i ~ Normal(mu_i, sigma), mu_i = beta0 + beta_k
# Ket qua bao gom posterior mean va 95% credible interval.
if (use_brms) {
  if (!requireNamespace("brms", quietly = TRUE)) {
    cat("\nPackage brms chua duoc cai. Bo qua phan brm().\n")
    cat("Neu muon chay Bayesian, cai thu cong: install.packages('brms')\n")
    writeLines(
      c(
        "Package brms chua duoc cai nen phan brm() bi bo qua.",
        "Cai thu cong bang: install.packages('brms')",
        "Sau do chay lai script voi use_brms <- TRUE."
      ),
      con = file.path(output_dir, "crd_brm_skipped.txt")
    )
  } else {
    library(brms)

    set.seed(random_seed)

    # Dung prior yeu/thong tin it vi F1 nam trong [0, 1].
    # Gaussian la lua chon tuong ung truc tiep voi lm() de so sanh.
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
      brm_summary,
      file = file.path(output_dir, "crd_brm_summary.txt")
    )

    # Posterior mean va 95% credible interval cho tung muc k
    brm_epred <- fitted(
      brm_crd,
      newdata = tibble(k = factor(paste0("k=", k_values), levels = paste0("k=", k_values))),
      probs = c(0.025, 0.975),
      summary = TRUE
    )

    brm_group_summary <- tibble(
      k = factor(paste0("k=", k_values), levels = paste0("k=", k_values)),
      brm_mean_f1 = brm_epred[, "Estimate"],
      brm_ci_lower = brm_epred[, "Q2.5"],
      brm_ci_upper = brm_epred[, "Q97.5"]
    )

    write.csv(
      brm_group_summary,
      file.path(output_dir, "crd_brm_group_summary_by_k.csv"),
      row.names = FALSE
    )

    print(brm_group_summary)

    # Posterior pairwise differences giua cac muc k.
    post <- as_draws_df(brm_crd)
    # Trong brms voi treatment coding: Intercept = k=3,
    # b_kk.5 = k=5 - k=3, b_kk.10 = k=10 - k=3.
    diff_k5_k3 <- post$b_kk.5
    diff_k10_k3 <- post$b_kk.10
    diff_k10_k5 <- post$b_kk.10 - post$b_kk.5

    summarize_draws_diff <- function(draws, name) {
      tibble(
        comparison = name,
        posterior_mean_diff = mean(draws),
        ci_lower = quantile(draws, 0.025),
        ci_upper = quantile(draws, 0.975),
        prob_diff_gt_0 = mean(draws > 0),
        prob_diff_lt_0 = mean(draws < 0)
      )
    }

    brm_pairwise <- bind_rows(
      summarize_draws_diff(diff_k5_k3, "k=5 - k=3"),
      summarize_draws_diff(diff_k10_k3, "k=10 - k=3"),
      summarize_draws_diff(diff_k10_k5, "k=10 - k=5")
    )

    write.csv(
      brm_pairwise,
      file.path(output_dir, "crd_brm_pairwise_differences.csv"),
      row.names = FALSE
    )

    print(brm_pairwise)

    # Do thi Bayesian: posterior mean + 95% credible interval cho tung k
    p_brm_group <- ggplot(brm_group_summary, aes(x = k, y = brm_mean_f1)) +
      geom_point(size = 3) +
      geom_errorbar(aes(ymin = brm_ci_lower, ymax = brm_ci_upper), width = 0.12, linewidth = 0.8) +
      labs(
        title = "Bayesian brm(): Posterior mean F1 theo k",
        subtitle = "Diem la posterior mean; thanh loi la 95% credible interval",
        x = "So fold k",
        y = "Posterior mean F1, positive class = yes"
      ) +
      theme_minimal(base_size = 13)

    ggsave(
      filename = file.path(output_dir, "crd_brm_group_posterior_ci.png"),
      plot = p_brm_group,
      width = 8,
      height = 5,
      dpi = 300
    )

    p_brm_pairwise <- ggplot(brm_pairwise, aes(x = comparison, y = posterior_mean_diff)) +
      geom_hline(yintercept = 0, linetype = "dashed") +
      geom_point(size = 3) +
      geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), width = 0.15, linewidth = 0.8) +
      coord_flip() +
      labs(
        title = "Bayesian brm(): Chenh lech F1 giua cac cap k",
        subtitle = "Khoang credible interval khong cat 0 => bang chung khac biet ro hon",
        x = "Cap so sanh",
        y = "Posterior mean difference in F1"
      ) +
      theme_minimal(base_size = 13)

    ggsave(
      filename = file.path(output_dir, "crd_brm_pairwise_posterior_ci.png"),
      plot = p_brm_pairwise,
      width = 8,
      height = 4.8,
      dpi = 300
    )
  }
}

# ============================================================
# 11. VE DO THI
# ============================================================
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
    title = "CRD: So sanh F1 giua cac gia tri k bang TukeyHSD",
    subtitle = "Diem la mean F1, thanh loi la 95% CI; chu cai khac nhau nghia la khac biet co y nghia theo TukeyHSD",
    x = "So fold k",
    y = "F1 score, positive class = yes"
  ) +
  theme_minimal(base_size = 13)

ggsave(
  filename = file.path(output_dir, "crd_group_comparison_tukey_letters.png"),
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
    title = "TukeyHSD pairwise comparison for k",
    subtitle = "Khoang tin cay khong cat 0 => khac biet co y nghia thong ke",
    x = "Cap so sanh",
    y = "Chenhlech trung binh F1"
  ) +
  theme_minimal(base_size = 13)

ggsave(
  filename = file.path(output_dir, "crd_tukey_pairwise_ci.png"),
  plot = p2,
  width = 8,
  height = 4.8,
  dpi = 300
)

cat("\nDone. Files saved to:\n", output_dir, "\n")
cat("Main output files:\n")
cat("- crd_repeated_kfold_f1_raw.csv\n")
cat("- crd_summary_by_k.csv\n")
cat("- crd_levene_test.txt\n")
cat("- crd_lm_anova_results.txt\n")
cat("- crd_tukey_hsd.csv / .txt\n")
cat("- crd_group_comparison_tukey_letters.png\n")
cat("- crd_tukey_pairwise_ci.png\n")
cat("- crd_brm_summary.txt, crd_brm_group_summary_by_k.csv, crd_brm_pairwise_differences.csv neu use_brms = TRUE va brms da cai thanh cong\n")
