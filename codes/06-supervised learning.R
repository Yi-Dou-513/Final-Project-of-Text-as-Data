library(quanteda)
library(glmnet)
library(tidyverse)

set.seed(2025)

# Assume `final_data` contains `text` and `level`
# Convert the label to a factor
final_data <- final_data %>%
  mutate(level = factor(level_pred))  # local / state / federal

# Build corpus and dfm
corp <- corpus(final_data, text_field = "text")

dfm_all <- corp %>%
  tokens(
    remove_punct   = TRUE,
    remove_numbers = TRUE,
    remove_symbols = TRUE
  ) %>%
  tokens_tolower() %>%
  tokens_remove(stopwords("en")) %>%
  dfm() %>%
  dfm_trim(min_termfreq = 5, min_docfreq = 5)  # trim extremely rare terms

# Train/test split (e.g., 80% / 20%)
id_train <- sample(seq_len(ndoc(dfm_all)), size = 0.8 * ndoc(dfm_all))
id_test  <- setdiff(seq_len(ndoc(dfm_all)), id_train)

dfm_train <- dfm_all[id_train, ]
dfm_test  <- dfm_all[id_test,  ]

y_train <- docvars(dfm_train, "level")
y_test  <- docvars(dfm_test,  "level")

# VERY IMPORTANT: align test features to the training feature set
dfm_test <- dfm_match(dfm_test, features = featnames(dfm_train))

# Convert to sparse matrices required by glmnet
x_train <- as(dfm_train, "dgCMatrix")
x_test  <- as(dfm_test,  "dgCMatrix")

library(glmnet)

# family = "multinomial": multiclass classification
# alpha = 0: ridge (L2)
cv_fit <- cv.glmnet(
  x = x_train,
  y = y_train,
  family = "multinomial",
  alpha = 0,                 # ridge
  type.measure = "class",    # CV metric: misclassification error
  nfolds = 5                 # you can also set this to 10
)

# Optimal lambda
cv_fit$lambda.min

# Predict classes
pred_test <- predict(
  cv_fit,
  newx = x_test,
  s = "lambda.min",
  type = "class"
)

pred_test <- factor(pred_test[, 1], levels = levels(y_test))

# Confusion matrix
table(Predicted = pred_test, Actual = y_test)

library(caret)

cm <- confusionMatrix(pred_test, y_test)

# Accuracy
accuracy <- cm$overall["Accuracy"]

# Precision / Recall / F1 (by class)
class_results <- as.data.frame(cm$byClass)

# Format into a cleaner tibble
results_table <- tibble(
  Metric = c(
    "Accuracy",
    "Precision (Macro Avg)",
    "Recall (Macro Avg)",
    "F1 (Macro Avg)"
  ),
  Value = c(
    accuracy,
    mean(class_results$Precision, na.rm = TRUE),
    mean(class_results$Recall,    na.rm = TRUE),
    mean(class_results$F1,        na.rm = TRUE)
  )
)

results_table

# Per-class precision, recall, and F1
per_class_table <- class_results %>%
  rownames_to_column("Class") %>%
  select(Class, Precision, Recall, F1)

per_class_table

library(gt)

results_table_gt <- results_table %>%
  gt() %>%
  tab_header(title = "Overall Performance Metrics") %>%
  fmt_number(columns = "Value", decimals = 3)

# Save as PNG
gtsave(results_table_gt, filename = "overall_metrics.png")

per_class_table_gt <- per_class_table %>%
  gt() %>%
  tab_header(title = "Per-Class Performance Metrics") %>%
  fmt_number(columns = c("Precision", "Recall", "F1"), decimals = 3)

# Save as PNG
gtsave(per_class_table_gt, filename = "per_class_metrics.png")

library(readr)

final_data %>%
  select(text, label = level) %>%   # rename `level` to `label`
  write.csv("final_data_for_bert.csv")

df <- tibble::tibble(
  metric = c(
    "Accuracy",
    "Federal Precision", "Federal Recall",
    "Local Precision",   "Local Recall",
    "State Precision",   "State Recall"
  ),
  DistilBERT = c(
    0.650,
    0.638, 0.667,
    0.660, 0.674,
    0.653, 0.610
  ),
  Ridge = c(
    0.670,
    0.684, 0.662,
    0.762, 0.595,
    0.605, 0.744
  ),
  Baseline = c(
    0.348,  # always predict federal -> accuracy equals the federal share
    0.348,  # federal precision equals the federal share
    1.000,  # federal recall = 1
    0.000,  # local precision
    0.000,  # local recall
    0.000,  # state precision
    0.000   # state recall
  )
)

# ------------------------------------------------------------
# 2) Reshape to long format + plot (grouped bars + legend)
# ------------------------------------------------------------
df_long <- df %>%
  pivot_longer(
    cols = c(DistilBERT, Ridge, Baseline),
    names_to = "model",
    values_to = "score"
  ) %>%
  mutate(
    metric = factor(metric, levels = df$metric),
    model  = factor(model, levels = c("DistilBERT", "Ridge", "Baseline"))
  )

p <- ggplot(df_long, aes(x = metric, y = score, fill = model)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.7) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    x = NULL,
    y = "Score",
    fill = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 35, hjust = 1),
    legend.position = "top"
  )

print(p)

# Export as PNG (if needed)
ggsave("model_comparison_bar.png", p, width = 10, height = 4.8, dpi = 300)
