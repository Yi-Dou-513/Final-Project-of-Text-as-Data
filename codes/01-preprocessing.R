library(tidyverse)

df <- read.csv("E:/final/2024_tweets_raw.csv")

df %>%
  summarize(
    total        = n(),
    missing      = sum(is.na(text) | text == ""),
    nonmissing   = sum(text != ""),
    missing_rate = missing / total
  )

# Aggregate tweets by official and calendar week
agg <- df %>%
  group_by(official_id, calendar_week) %>%
  summarise(
    text     = paste(na.omit(text), collapse = " "),
    n_tweets = n(),
    .groups  = "drop"
  ) %>%
  filter(!is.na(text), str_squish(text) != "")

# Left join with official-level data
official <- read.csv("E:/final/official_data.csv")
agg2 <- agg %>%
  left_join(official, by = "official_id")

# Text preprocessing
library(quanteda)

corp <- corpus(agg2, text_field = "text")

toks <- corp %>%
  tokens(
    remove_punct   = TRUE,  # Remove punctuation
    remove_numbers = TRUE,  # Remove numbers
    remove_url     = TRUE   # Remove URLs
  ) %>%
  tokens_tolower() %>%       # Convert all tokens to lowercase
  tokens_keep(min_nchar = 2) # Drop one-character tokens (e.g., "a", "I")

dfm_agg <- toks %>%
  dfm() %>%
  dfm_remove(stopwords("en")) # Remove English stopwords
