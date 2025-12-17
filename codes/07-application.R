library(quanteda)
library(Matrix)
library(dplyr)

# 1. Load the new dataset
tweets_new <- read.csv("tweets_congress.csv")

# Assume the text column is named `text`. If not, rename it accordingly:
# tweets_new <- tweets_new %>% rename(text = <your_text_column_name>)

# 2. Build corpus + tokens + dfm (match training preprocessing as closely as possible)
corp_new <- corpus(tweets_new, text_field = "text")

dfm_new_raw <- corp_new %>%
  tokens(
    remove_punct   = TRUE,
    remove_numbers = TRUE,
    remove_symbols = TRUE
  ) %>%
  tokens_tolower() %>%
  tokens_remove(stopwords("en")) %>%
  dfm()

# Align features with the training feature set
dfm_new <- dfm_match(dfm_new_raw, features = featnames(dfm_train))

# Convert to the sparse matrix format required by glmnet
x_new <- as(dfm_new, "dgCMatrix")

# Predict classes using the optimal lambda
pred_new_class <- predict(
  cv_fit,
  newx = x_new,
  s = "lambda.min",
  type = "class"
)

# Convert to factor with the same levels as in training
pred_new_class <- factor(
  pred_new_class[, 1],
  levels = levels(y_train)
)

# If you also want class probabilities:
pred_new_prob <- predict(
  cv_fit,
  newx = x_new,
  s = "lambda.min",
  type = "response"
)  # returns a list (one matrix per class)

tweets_new_pred <- tweets_new %>%
  mutate(level_pred = pred_new_class)

# Preview
head(tweets_new_pred)

# Export predictions
write_csv(tweets_new_pred, "tweets_congress_with_level_pred.csv")

# Republicans
data_R <- tweets_new_pred %>%
  filter(Party == "R") %>%
  count(level_pred) %>%
  mutate(
    prop  = n / sum(n),
    label = scales::percent(prop, accuracy = 1)
  )

ggplot(data_R, aes(x = "", y = prop, fill = level_pred)) +
  geom_col(width = 1) +
  coord_polar(theta = "y") +
  geom_label(
    aes(label = label),
    position = position_stack(vjust = 0.5),
    show.legend = FALSE
  ) +
  labs(
    title = "Republican Tweets by Level",
    fill = "Level"
  ) +
  theme_void()

ggsave("pie_R.png", width = 5, height = 5)

# Democrats
data_D <- tweets_new_pred %>%
  filter(Party == "D") %>%
  count(level_pred) %>%
  mutate(
    prop  = n / sum(n),
    label = scales::percent(prop, accuracy = 1)
  )

ggplot(data_D, aes(x = "", y = prop, fill = level_pred)) +
  geom_col(width = 1) +
  coord_polar(theta = "y") +
  geom_label(
    aes(label = label),
    position = position_stack(vjust = 0.5),
    show.legend = FALSE
  ) +
  labs(
    title = "Democratic Tweets by Level",
    fill = "Level"
  ) +
  theme_void()

ggsave("pie_D.png", width = 5, height = 5)

# House
data_House <- tweets_new_pred %>%
  filter(congress == "House") %>%
  count(level_pred) %>%
  mutate(
    prop  = n / sum(n),
    label = scales::percent(prop, accuracy = 1)
  )

ggplot(data_House, aes(x = "", y = prop, fill = level_pred)) +
  geom_col(width = 1) +
  coord_polar(theta = "y") +
  geom_label(
    aes(label = label),
    position = position_stack(vjust = 0.5),
    show.legend = FALSE
  ) +
  labs(
    title = "House Tweets by Level",
    fill = "Level"
  ) +
  theme_void()

ggsave("pie_House.png", width = 5, height = 5)

# Senate
data_Senate <- tweets_new_pred %>%
  filter(congress == "senate") %>%
  count(level_pred) %>%
  mutate(
    prop  = n / sum(n),
    label = scales::percent(prop, accuracy = 1)
  )

ggplot(data_Senate, aes(x = "", y = prop, fill = level_pred)) +
  geom_col(width = 1) +
  coord_polar(theta = "y") +
  geom_label(
    aes(label = label),
    position = position_stack(vjust = 0.5),
    show.legend = FALSE
  ) +
  labs(
    title = "Senate Tweets by Level",
    fill = "Level"
  ) +
  theme_void()

ggsave("pie_Senate.png", width = 5, height = 5)
