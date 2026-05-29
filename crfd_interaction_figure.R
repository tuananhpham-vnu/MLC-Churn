library(tidyverse)

# ============================================================
# Interaction plot: k x max_depth
# ============================================================

# Neu da co summary_crfd thi co the bo qua doan nay
summary_crfd <- crfd_results %>%
  group_by(k, max_depth) %>%
  summarise(
    n = n(),
    mean_f1 = mean(f1),
    sd_f1 = sd(f1),
    se = sd_f1 / sqrt(n),
    ci_lower = mean_f1 - qt(0.975, df = n - 1) * se,
    ci_upper = mean_f1 + qt(0.975, df = n - 1) * se,
    .groups = "drop"
  )

# Dam bao thu tu hien thi
summary_crfd <- summary_crfd %>%
  mutate(
    k = factor(k, levels = c("k=3", "k=5", "k=10")),
    max_depth = factor(max_depth, levels = c("3", "5", "None"))
  )

# Ve do thi tuong tac
p_interaction <- ggplot(
  summary_crfd,
  aes(x = k, y = mean_f1, group = max_depth, color = max_depth)
) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2.8) +
  geom_errorbar(
    aes(ymin = ci_lower, ymax = ci_upper),
    width = 0.10,
    linewidth = 0.6
  ) +
  labs(
    x = "Số fold k",
    y = "Mean F1-score",
    color = "max_depth"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_blank(),
    plot.subtitle = element_blank()
  )

# Luu anh
output_dir <- "Phase_3_CRFD/outputs"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

plot_path <- file.path(output_dir, "crfd_interaction_plot_k_max_depth.png")

ggsave(
  filename = plot_path,
  plot = p_interaction,
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
