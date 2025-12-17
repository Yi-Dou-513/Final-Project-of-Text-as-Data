library(topicmodels)
library(cvTools)
library(reshape)
library(ggplot2)

# quanteda::dfm -> dgCMatrix
dtm_agg <- convert(dfm_agg, to = "topicmodels")
dtm_agg

cvLDA <- function(Ntopics, dtm, K_folds = 5, iter = 500, seed = 1234) {
  set.seed(seed)
  folds   <- cvFolds(nrow(dtm), K_folds, 1)
  perplex <- rep(NA, K_folds)
  llk     <- rep(NA, K_folds)
  
  for (i in unique(folds$which)) {
    cat("Fold", i, "of", K_folds, "for", Ntopics, "topics\n")
    
    which.test  <- folds$subsets[folds$which == i]
    which.train <- {1:nrow(dtm)}[-which.test]
    
    dtm.train <- dtm[which.train, ]
    dtm.test  <- dtm[which.test,  ]
    
    lda.fit <- LDA(
      dtm.train,
      k      = Ntopics,
      method = "Gibbs",
      control = list(
        verbose = 0L,
        iter    = iter,
        seed    = seed + i  # slightly vary the seed across folds
      )
    )
    
    perplex[i] <- perplexity(lda.fit, dtm.test)
    llk[i]     <- logLik(lda.fit)
  }
  
  return(list(K = Ntopics, perplexity = perplex, logLik = llk))
}

# Run CV over a grid of K and automatically select an "optimal K"
selectK_LDA <- function(dtm,
                        K_grid,
                        K_folds   = 5,
                        iter      = 500,
                        seed      = 1234,
                        save_file = NULL) {
  
  results <- vector("list", length(K_grid))
  names(results) <- K_grid
  
  i <- 1
  for (k in K_grid) {
    cat("\n\n##########\n", k, "topics\n##########\n")
    results[[i]] <- cvLDA(
      Ntopics = k,
      dtm     = dtm,
      K_folds = K_folds,
      iter    = iter,
      seed    = seed + i
    )
    if (!is.null(save_file)) {
      save(results, file = save_file)
    }
    i <- i + 1
  }
  
  # Combine into a data.frame
  df <- data.frame(
    k     = rep(K_grid, each = K_folds),
    perp  = unlist(lapply(results, `[[`, "perplexity")),
    loglk = unlist(lapply(results, `[[`, "logLik")),
    stringsAsFactors = FALSE
  )
  
  # Normalize (mirroring the original code)
  df$ratio_perp <- df$perp  / max(df$perp,  na.rm = TRUE)
  df$ratio_lk   <- df$loglk / min(df$loglk, na.rm = TRUE)
  
  # Aggregate mean and sd by K
  df_sum <- data.frame(
    cbind(
      aggregate(df$ratio_perp, by = list(df$k), FUN = mean, na.rm = TRUE),
      aggregate(df$ratio_perp, by = list(df$k), FUN = sd,   na.rm = TRUE)$x,
      aggregate(df$ratio_lk,   by = list(df$k), FUN = mean, na.rm = TRUE)$x,
      aggregate(df$ratio_lk,   by = list(df$k), FUN = sd,   na.rm = TRUE)$x
    ),
    stringsAsFactors = FALSE
  )
  names(df_sum) <- c("k", "ratio_perp", "sd_perp", "ratio_lk", "sd_lk")
  
  # Reshape to long format for plotting
  pd  <- melt(df_sum[, c("k", "ratio_perp", "ratio_lk")], id.vars = "k")
  pd2 <- melt(df_sum[, c("k", "sd_perp", "sd_lk")],       id.vars = "k")
  pd$sd <- pd2$value
  levels(pd$variable) <- c("Perplexity", "LogLikelihood")
  
  p <- ggplot(pd, aes(x = k, y = value, linetype = variable)) +
    geom_line() +
    geom_point(
      aes(shape = variable),
      fill  = "white",
      shape = 21,
      size  = 1.40
    ) +
    geom_errorbar(
      aes(ymax = value + sd, ymin = value - sd),
      width = diff(range(K_grid)) / (length(K_grid) * 2)
    ) +
    scale_y_continuous("Ratio wrt worst value") +
    scale_x_continuous(
      "Number of topics",
      breaks = K_grid
    ) +
    theme_bw() +
    scale_shape_discrete(guide = "none") +
    scale_linetype_discrete(guide = "none") +
    theme(
      axis.line        = element_line(colour = "black"),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border     = element_blank(),
      panel.background = element_blank(),
      legend.key.size  = unit(0.5, "cm"),
      legend.position  = c(0.70, .90),
      legend.box.just  = "left",
      legend.direction = "horizontal",
      legend.title     = element_blank()
    ) +
    annotate("text", x = max(K_grid), y = 0.86,  label = "Perplexity",    size = 3) +
    annotate("text", x = max(K_grid), y = 0.945, label = "logLikelihood", size = 3)
  
  # Select K by minimizing mean ratio_perp
  best_k <- df_sum$k[which.min(df_sum$ratio_perp)]
  
  return(list(
    best_k  = best_k,
    results = results,
    summary = df_sum,
    plot    = p
  ))
}

# Set the grid of K values to evaluate
K_grid <- c(10, 20, 30, 40, 50, 60, 70, 80)

sel_officialweek <- selectK_LDA(
  dtm       = dtm_agg,
  K_grid    = K_grid,
  K_folds   = 5,       # can increase to 10, but it will be slower
  iter      = 500,     # iterations per fold
  seed      = 2024,
  save_file = "k_topics_results_cv_officialweek.Rdata"
)

# Selected "best" K
sel_officialweek$best_k

# Summary table
sel_officialweek$summary

# Plot
sel_officialweek$plot

ggsave(
  "appendix-choosing-k-officialweek.pdf",
  sel_officialweek$plot,
  height = 4,
  width = 6
)

best_k <- sel_officialweek$best_k  # e.g., 40

lda_officialweek <- LDA(
  dtm_agg,
  k      = best_k,
  method = "Gibbs",
  control = list(
    verbose = 50L,
    iter    = 2000,
    seed    = 2025
  )
)

save(lda_officialweek, file = "lda_results_officialweek.Rdata")

top_terms <- terms(lda_officialweek, 15)
top_terms
