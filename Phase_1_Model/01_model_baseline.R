getwd()
setwd('D:\\Folder F\\phamtuananh@23020010\\UET.iSEML\\2026.DAE.MLC Churn')
# 1. Load packages
install.packages(c("tidyverse", "corrplot", "ggpubr"))
library(tidyverse)
library(corrplot)
library(ggpubr)

# 2. Load data
df <- read.csv("mlc_churn.csv")
df
# 3. Chọn các cặp minutes và charge
corr_pairs <- list(
  day   = c("total_day_minutes", "total_day_charge"),
  eve   = c("total_eve_minutes", "total_eve_charge"),
  night = c("total_night_minutes", "total_night_charge"),
  intl  = c("total_intl_minutes", "total_intl_charge")
)

# 4. Kiểm định tương quan Pearson
# H0: rho = 0
# H1: rho != 0
corr_results <- map_dfr(names(corr_pairs), function(name) {
  x <- corr_pairs[[name]][1]
  y <- corr_pairs[[name]][2]
  
  test <- cor.test(df[[x]], df[[y]], method = "pearson")
  
  tibble(
    pair = name,
    variable_1 = x,
    variable_2 = y,
    correlation_r = as.numeric(test$estimate),
    p_value = test$p.value,
    conclusion = ifelse(
      abs(test$estimate) > 0.95 & test$p.value < 0.05,
      "Gần như trùng thông tin, nên giữ minutes và bỏ charge",
      "Chưa đủ cơ sở để loại bỏ"
    )
  )
})

print(corr_results)

# 5. Vẽ heatmap tương quan
selected_vars <- c(
  "total_day_minutes", "total_day_charge",
  "total_eve_minutes", "total_eve_charge",
  "total_night_minutes", "total_night_charge",
  "total_intl_minutes", "total_intl_charge"
)

corr_matrix <- cor(df[selected_vars], method = "pearson")

corrplot(
  corr_matrix,
  method = "color",
  type = "upper",
  addCoef.col = "black",
  tl.col = "black",
  tl.srt = 45,
  number.cex = 0.7,
  title = "Pearson correlation between minutes and charge variables",
  mar = c(0, 0, 2, 0)
)

# 6. Scatter plot từng cặp minutes-charge
plot_day <- ggplot(df, aes(x = total_day_minutes, y = total_day_charge)) +
  geom_point(alpha = 0.4) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(
    title = "Day minutes vs Day charge",
    x = "Total day minutes",
    y = "Total day charge"
  ) +
  theme_minimal()

plot_eve <- ggplot(df, aes(x = total_eve_minutes, y = total_eve_charge)) +
  geom_point(alpha = 0.4) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(
    title = "Evening minutes vs Evening charge",
    x = "Total evening minutes",
    y = "Total evening charge"
  ) +
  theme_minimal()

plot_night <- ggplot(df, aes(x = total_night_minutes, y = total_night_charge)) +
  geom_point(alpha = 0.4) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(
    title = "Night minutes vs Night charge",
    x = "Total night minutes",
    y = "Total night charge"
  ) +
  theme_minimal()

plot_intl <- ggplot(df, aes(x = total_intl_minutes, y = total_intl_charge)) +
  geom_point(alpha = 0.4) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(
    title = "International minutes vs International charge",
    x = "Total international minutes",
    y = "Total international charge"
  ) +
  theme_minimal()

ggarrange(
  plot_day, plot_eve,
  plot_night, plot_intl,
  ncol = 2,
  nrow = 2
)
