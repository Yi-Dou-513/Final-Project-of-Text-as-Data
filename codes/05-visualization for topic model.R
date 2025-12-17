library(tidyverse)
library(ggplot2)

#========================================================
# 0. Inspect the range of the three z-scores (mean ~ 0, sd ~ 1)
#========================================================
final_data %>%
  summarise(
    min_fed   = min(federal_score_z, na.rm = TRUE),
    max_fed   = max(federal_score_z, na.rm = TRUE),
    min_state = min(state_score_z, na.rm = TRUE),
    max_state = max(state_score_z, na.rm = TRUE),
    min_local = min(local_score_z, na.rm = TRUE),
    max_local = max(local_score_z, na.rm = TRUE)
  )

# Automatically choose a "contrast-enhancing" range: 5th–95th percentiles
qs <- final_data %>%
  summarise(
    fed_lo   = quantile(federal_score_z, 0.05, na.rm = TRUE),
    fed_hi   = quantile(federal_score_z, 0.95, na.rm = TRUE),
    state_lo = quantile(state_score_z,   0.05, na.rm = TRUE),
    state_hi = quantile(state_score_z,   0.95, na.rm = TRUE),
    local_lo = quantile(local_score_z,   0.05, na.rm = TRUE),
    local_hi = quantile(local_score_z,   0.95, na.rm = TRUE)
  )

qs

y_min <- min(qs$fed_lo, qs$state_lo, qs$local_lo)
y_max <- max(qs$fed_hi, qs$state_hi, qs$local_hi)

#========================================================
# 1. Pivot to long format and construct party_clean
#========================================================
final_data <- final_data %>%
  mutate(party_clean = case_when(
    party %in% c("D", "DEM", "Dem", "Democratic") ~ "Democrat",
    party %in% c("R", "GOP", "Rep", "Republican") ~ "Republican",
    TRUE ~ "Other"
  ))

scores_long <- final_data %>%
  select(
    party_clean, office_level,
    federal_score_z, state_score_z, local_score_z
  ) %>%
  pivot_longer(
    cols      = c(federal_score_z, state_score_z, local_score_z),
    names_to  = "dimension",
    values_to = "score"
  ) %>%
  mutate(
    dimension = recode(
      dimension,
      federal_score_z = "Federal (z)",
      state_score_z   = "State (z)",
      local_score_z   = "Local (z)"
    )
  )

#========================================================
# 2. Party × three z-score dimensions
#========================================================
party_means <- scores_long %>%
  group_by(party_clean, dimension) %>%
  summarise(mean_score = mean(score, na.rm = TRUE), .groups = "drop")

p_party <- ggplot(
  party_means,
  aes(x = party_clean, y = mean_score, fill = dimension)
) +
  geom_col(position = "dodge") +
  # coord_cartesian(ylim = c(y_min, y_max)) +
  labs(
    x = "Party",
    y = "Average standardized score (z)",
    fill = "Dimension",
    title = "Standardized Federal / State / Local (z-score) by party"
  ) +
  theme_minimal()

ggsave("plot_party_scores_z.png", p_party, width = 7, height = 5, dpi = 300)
p_party

#========================================================
# 3. Office level × three z-score dimensions
#========================================================
level_means <- scores_long %>%
  group_by(office_level, dimension) %>%
  summarise(mean_score = mean(score, na.rm = TRUE), .groups = "drop")

p_level <- ggplot(
  level_means,
  aes(x = office_level, y = mean_score, fill = dimension)
) +
  geom_col(position = "dodge") +
  # coord_cartesian(ylim = c(y_min, y_max)) +
  labs(
    x = "Office level",
    y = "Average standardized score (z)",
    fill = "Dimension",
    title = "Standardized Federal / State / Local (z-score) by office level"
  ) +
  theme_minimal()

ggsave("plot_officelevel_scores_z.png", p_level, width = 7, height = 5, dpi = 300)
p_level

#========================================================
# 4. 3D scatter proxy: randomly sample 50 observations (using z-scores)
#========================================================
set.seed(2025)

sample_df <- final_data %>%
  filter(
    !is.na(federal_score_z),
    !is.na(state_score_z),
    !is.na(local_score_z)
  ) %>%
  slice_sample(n = 50)

nrow(sample_df)  # confirm sample size is 50

p_3d <- ggplot(
  sample_df,
  aes(
    x = federal_score_z,
    y = state_score_z,
    colour = local_score_z
  )
) +
  geom_point(size = 3, alpha = 0.8) +
  scale_colour_viridis_c(option = "plasma") +
  labs(
    x = "Federal (z-score)",
    y = "State (z-score)",
    colour = "Local (z-score)",
    title = "Sample of 50 observations in standardized 3D space"
  ) +
  theme_minimal()

ggsave("scatter_3d_sample50_z.png", p_3d, width = 6, height = 5, dpi = 300)
p_3d
