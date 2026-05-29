library(tidyverse)

# ============================================================
# Tukey HSD Confidence Interval Plot
# CRD: k thay doi, max_depth = None
# ============================================================

# Lay ket qua CRD voi max_depth = None
crd_none <- crfd_results %>%
  filter(as.character(max_depth) == "None")

# ANOVA va TukeyHSD
aov_none <- aov(f1 ~ k, data = crd_none)
tukey_none <- TukeyHSD(aov_none)

# Chuyen ket qua TukeyHSD thanh data frame
tukey_df <- as.data.frame(tukey_none$k) %>%
  rownames_to_column("comparison") %>%
  rename(
    diff = diff,
    lower = lwr,
    upper = upr,
    p_adj = `p adj`
  ) %>%
  mutate(
    significant = ifelse(p_adj < 0.05, "Significant", "Not significant")
  )

print(tukey_df)

# Ve Tukey HSD Confidence Interval Plot
p_tukey_ci <- ggplot(tukey_df, aes(x = comparison, y = diff)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_point(size = 3) +
  geom_errorbar(
    aes(ymin = lower, ymax = upper),
    width = 0.15,
    linewidth = 0.8
  ) +
  coord_flip() +
  labs(
    x = "Pairwise comparison",
    y = "Mean F1 difference"
  ) +
  theme_minimal(base_size = 13)

# Luu anh dpi = 300
output_dir <- "Phase_3_CRFD/outputs"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

plot_path <- file.path(output_dir, "tukey_hsd_ci_plot_k_none.png")

ggsave(
  filename = plot_path,
  plot = p_tukey_ci,
  width = 7,
  height = 4.5,
  dpi = 300
)

# Mo anh sau khi luu
plot_path_abs <- normalizePath(plot_path, mustWork = FALSE)
print(plot_path_abs)
print(file.exists(plot_path_abs))

if (file.exists(plot_path_abs)) {
  shell.exec(plot_path_abs)
}