library(topicmodels)

#========================================================
# 1. Extract LDA term distribution beta (topic × term)
#========================================================
beta <- posterior(lda_officialweek)$terms   # topic × term matrix
dim(beta)
head(colnames(beta))

library(data.table)
library(tidyverse)
library(purrr)

#========================================================
# 2. Load GloVe embeddings
#========================================================
glove_wts <- data.table::fread(
  "glove.6B.300d.txt",
  quote = "",
  data.table = FALSE
) %>%
  as_tibble()

glove_matrix <- as.matrix(glove_wts %>% select(-V1))
rownames(glove_matrix) <- glove_wts$V1

# Convert everything to lowercase for alignment
colnames(beta)         <- tolower(colnames(beta))
rownames(glove_matrix) <- tolower(rownames(glove_matrix))

#========================================================
# 3. Keep only terms that appear in both the LDA vocabulary and GloVe
#========================================================
common_terms <- intersect(colnames(beta), rownames(glove_matrix))
length(common_terms)   # number of matched terms

beta_sub <- beta[, common_terms, drop = FALSE]             # topic × term
E_sub    <- glove_matrix[common_terms, , drop = FALSE]     # term × 300

#========================================================
# 4. L2-normalize embeddings for each term (row vectors)
#========================================================
row_norms <- sqrt(rowSums(E_sub^2))
row_norms[row_norms == 0] <- 1
E_norm <- E_sub / row_norms  # term × 300, rows are unit length

#========================================================
# 5. Define and compute orthogonalized federal/state/local dimension vectors  ### NEW
#========================================================
# Ensure these three words exist in the GloVe vocabulary
if (!all(c("federal", "state", "local") %in% rownames(glove_matrix))) {
  stop("At least one of federal/state/local is not in the GloVe vocabulary.")
}

fed_raw   <- glove_matrix["federal", ]
state_raw <- glove_matrix["state", ]
local_raw <- glove_matrix["local", ]

# Project target out of the subspace spanned by others (take residual)
residualize <- function(target, others) {
  v <- target
  for (o in others) {
    v <- v - (sum(v * o) / sum(o * o)) * o
  }
  return(v)
}

# Residualize the three dimensions against each other (remove shared semantics)
fed_new   <- residualize(fed_raw,   list(state_raw, local_raw))
state_new <- residualize(state_raw, list(fed_raw,   local_raw))
local_new <- residualize(local_raw, list(fed_raw,   state_raw))

# Normalize each vector to unit length
norm_vec <- function(v) v / sqrt(sum(v^2))

fed_new   <- norm_vec(fed_new)
state_new <- norm_vec(state_new)
local_new <- norm_vec(local_new)

# Store in a list for convenient access by name
dim_vecs <- list(
  federal = fed_new,
  state   = state_new,
  local   = local_new
)

#========================================================
# 6. Score each topic using the orthogonalized dimension vectors  ### updated function
#========================================================
score_topics_for_dimension <- function(dim_word,
                                       beta_sub,
                                       E_norm,
                                       dim_vecs) {
  w <- tolower(dim_word)
  if (!w %in% names(dim_vecs)) {
    stop("Dimension word '", dim_word, "' does not have a corresponding orthogonalized vector.")
  }
  
  # Use the residualized + normalized dimension vector
  e_dim <- dim_vecs[[w]]   # length 300
  
  # Cosine similarity between all common_terms and this dimension vector
  # E_norm: term × 300, e_dim: 300
  # sim_vec: term × 1
  sim_vec <- as.numeric(E_norm %*% e_dim)
  names(sim_vec) <- rownames(E_norm)   # helpful for inspecting top-similarity terms
  
  # Topic-level weighted average similarity: (topic × term) × (term × 1)
  scores <- as.numeric(beta_sub %*% sim_vec)
  
  tibble(
    topic     = 1:nrow(beta_sub),
    dimension = w,
    score     = scores
  )
}

# Score topics using the orthogonalized three dimensions
dim_words <- c("federal", "state", "local")

topic_scores <- map_dfr(
  dim_words,
  ~ score_topics_for_dimension(.x, beta_sub, E_norm, dim_vecs)
)

topic_scores

#========================================================
# 7. Standardize (z-score) for comparability
#========================================================
topic_scores_std <- topic_scores %>%
  group_by(dimension) %>%
  mutate(score_z = (score - mean(score)) / sd(score)) %>%
  ungroup()

topic_scores_std %>%
  group_by(dimension) %>%
  slice_max(order_by = score_z, n = 5) %>%
  arrange(dimension, desc(score_z))

#========================================================
# 8. Extract topic distribution gamma (document × topic)
#========================================================
gamma <- posterior(lda_officialweek)$topics  # docs × topics
dim(gamma)
head(gamma[, 1:5])

#========================================================
# 9. Convert to wide format: one row per topic, one column per dimension score
#========================================================
topic_dim_wide <- topic_scores %>%
  select(topic, dimension, score) %>%
  tidyr::pivot_wider(
    names_from  = dimension,
    values_from = score
  ) %>%
  arrange(topic)

topic_dim_wide

topic_dim_wide <- topic_dim_wide %>%
  select(topic, federal, state, local)

# Convert to matrix: topics × 3
topic_dim_mat <- as.matrix(topic_dim_wide[, c("federal", "state", "local")])
rownames(topic_dim_mat) <- topic_dim_wide$topic   # use topic IDs as row names

topic_dim_mat[1:5, ]

#========================================================
# 10. Compute scores for each observation (document × 3 dimensions)
#========================================================
doc_dim_mat <- gamma %*% topic_dim_mat   # docs × 3
colnames(doc_dim_mat) <- c("federal_score", "state_score", "local_score")

#========================================================
# 11. Merge scores back into the original data
#========================================================
library(tibble)

valid_docs <- dtm_agg$dimnames$Docs

agg2 <- agg2 %>%
  mutate(doc_id = docnames(dfm_agg))

agg2_valid <- agg2 %>%
  filter(doc_id %in% valid_docs)

doc_scores <- as_tibble(doc_dim_mat, .name_repair = "minimal") %>%
  mutate(doc_id = valid_docs, .before = 1)

colnames(doc_scores)[-(1)] <- c("federal_score", "state_score", "local_score")

head(doc_scores)

final_data <- agg2_valid %>%
  left_join(doc_scores, by = "doc_id")

final_data %>%
  select(
    official_id, calendar_week,
    federal_score, state_score, local_score
  ) %>%
  head()

#========================================================
# 12. Standardize the three dimension scores (z-score)  ### NEW
#========================================================
final_data <- final_data %>%
  mutate(
    federal_score_z = (federal_score - mean(federal_score, na.rm = TRUE)) /
      sd(federal_score, na.rm = TRUE),
    state_score_z   = (state_score   - mean(state_score,   na.rm = TRUE)) /
      sd(state_score,  na.rm = TRUE),
    local_score_z   = (local_score   - mean(local_score,   na.rm = TRUE)) /
      sd(local_score,  na.rm = TRUE)
  )

# Inspect the results
final_data %>%
  select(
    official_id, calendar_week,
    federal_score, state_score, local_score,
    federal_score_z, state_score_z, local_score_z
  ) %>%
  head()
