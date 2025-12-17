library(tidyverse)
library(ggplot2)

# Assign predicted level based on the highest standardized score
final_data <- final_data %>%
  rowwise() %>%
  mutate(
    level_pred = c("federal", "state", "local")[
      which.max(c(federal_score_z, state_score_z, local_score_z))
    ]
  ) %>%
  ungroup()

table(final_data$level_pred)

# Compute distribution of predicted levels
level_dist <- final_data %>%
  count(level_pred) %>%
  mutate(
    prop  = n / sum(n),
    label = scales::percent(prop, accuracy = 0.1)
  )

level_dist

# Pie chart of predicted level distribution
p_pie <- ggplot(
  level_dist,
  aes(x = "", y = prop, fill = level_pred)
) +
  geom_col(width = 1, color = "white") +
  coord_polar(theta = "y") +
  geom_text(
    aes(label = label),
    position = position_stack(vjust = 0.5),
    size = 4
  ) +
  labs(
    fill = "Predicted level",
    title = "Distribution of predicted political levels"
  ) +
  theme_void()

p_pie

# Save figure
ggsave(
  "pie_predicted_levels.png",
  p_pie,
  width = 5,
  height = 5,
  dpi = 300
)
