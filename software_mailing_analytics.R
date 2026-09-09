# Load libraries
library(readr)       
library(dplyr)       
library(ggplot2)
library(caret)
library(pROC)
library(rpart)

theme_set(theme_minimal())

# File Path
data_path <- 'Software_Mailing_List.csv'

# Load data
df <- read_csv(data_path, show_col_types = FALSE)

# Fix column names
names(df) <- make.names(names(df), unique = TRUE)
cat("Column names after cleaning:\n"); print(names(df)); cat("\n")

# Dataset dimensions
cat("Dataset dimensions (rows x cols):\n"); print(dim(df)); cat("\n")

# Drop ID column
if ("sequence_number" %in% names(df)) {
  df <- df %>% dplyr::select(-sequence_number)
  cat("Dropped 'sequence_number' column (not predictive).\n")
}
cat("Column names after cleaning:\n"); print(names(df)); cat("\n")

# Summary Statistics
cat("Summary statistics for dataset:\n"); print(summary(df)); cat("\n")

# Missing Values
cat("Missing values per column:\n"); print(colSums(is.na(df))); cat("\n")

# Create a SpendingClass (Low / High) for purchasers (median split)
if (all(c("Spending", "Purchase") %in% names(df))) {
  purch_idx <- if (is.factor(df$Purchase)) df$Purchase %in% c("Purchase", 1) else df$Purchase == 1
  if (sum(purch_idx, na.rm = TRUE) > 0) {
    cutoff <- median(df$Spending[purch_idx], na.rm = TRUE)
    df$SpendingClass <- NA_character_
    df$SpendingClass[purch_idx] <- ifelse(df$Spending[purch_idx] <= cutoff, "Low", "High")
    df$SpendingClass <- factor(df$SpendingClass, levels = c("Low","High"))
    
    cat("\nSpendingClass counts (purchasers only, median split):\n")
    print(table(df$SpendingClass, useNA = "ifany")); cat("\n")
    cat(sprintf("Binary SpendingClass cutoff (median among purchasers): %.2f\n\n", cutoff))
  } else {
    cat("\nNo purchasers found; SpendingClass not created.\n")
  }
}

# convert binaries as factors (for plotting + modeling)
binary_cols <- c("US","Web.order","Gender.male","Address_is_res","Purchase")
source_cols <- grep("^source_", names(df), value = TRUE)
binary_cols <- intersect(c(binary_cols, source_cols), names(df))

for (col in binary_cols) {
  if (col == "Purchase") {
    df[[col]] <- factor(df[[col]], levels = c(0,1), labels = c("NoPurchase","Purchase"))
  } else if (is.numeric(df[[col]])) {
    df[[col]] <- factor(df[[col]], levels = c(0,1), labels = c("No","Yes"))
  }
}

# Ensure positive classes for caret's twoClassSummary (positive must be the first level)
if (is.factor(df$Purchase)) df$Purchase <- relevel(df$Purchase, ref = "Purchase")
if ("SpendingClass" %in% names(df) && is.factor(df$SpendingClass)) {
  df$SpendingClass <- relevel(df$SpendingClass, ref = "High")
}

# Recompute types
numeric_cols     <- names(df)[sapply(df, is.numeric)]
categorical_cols <- names(df)[sapply(df, is.factor)]


# Histograms for numeric variables
for (cn in numeric_cols) {
  p <- ggplot(df, aes_string(x = cn)) +
    geom_histogram(bins = 30, color = "white") +
    labs(title = paste("Histogram:", cn), x = cn, y = "Count")
  print(p)
}

# Bar plots for non-source categorical variables
other_cats <- setdiff(categorical_cols, source_cols)
for (cn in other_cats) {
  p <- ggplot(df, aes_string(x = cn)) +
    geom_bar() +
    labs(title = paste("Bar Plot:", cn), x = cn, y = "Count")
  print(p)
}

# Plots of source columns
if (length(source_cols) > 0) {
  # Counts Yes/No per source
  src_counts <- dplyr::bind_rows(lapply(source_cols, function(sc) {
    vals <- df[[sc]]
    if (!is.factor(vals) && is.numeric(vals)) vals <- factor(vals, levels = c("No","Yes"))
    data.frame(source = sc, level = vals)
  })) %>%
    dplyr::group_by(source, level) %>%
    dplyr::summarise(n = dplyr::n(), .groups = "drop")
  
  p_sources_counts <- ggplot(src_counts, aes(x = level, y = n)) +
    geom_col() +
    facet_wrap(~ source, ncol = 4, scales = "free_y") +
    labs(title = "Source Flags - Counts by Level", x = "", y = "Count")
  print(p_sources_counts)
  
  # Purchase rate by source
  if ("Purchase" %in% names(df)) {
    src_pr <- dplyr::bind_rows(lapply(source_cols, function(sc) {
      vals <- df[[sc]]
      if (!is.factor(vals) && is.numeric(vals)) vals <- factor(vals, levels = c("No","Yes"))
      data.frame(source = sc, level = vals, Purchase = df$Purchase)
    })) %>%
      dplyr::group_by(source, level) %>%
      dplyr::summarise(purchase_rate = mean(Purchase == "Purchase"), .groups = "drop")
    
    p_sources_pr <- ggplot(src_pr, aes(x = level, y = purchase_rate)) +
      geom_col() +
      scale_y_continuous(labels = scales::percent_format()) +
      facet_wrap(~ source, ncol = 4, scales = "fixed") +
      labs(title = "Purchase Rate by Source (Yes vs No)", x = "", y = "Purchase rate")
    print(p_sources_pr)
  }
  
  # SpendingClass share by source (purchasers only; Low/High)
  if ("SpendingClass" %in% names(df)) {
    purch_idx <- if (is.factor(df$Purchase)) df$Purchase == "Purchase" else df$Purchase == 1
    purch <- df[purch_idx, ]
    src_sc <- dplyr::bind_rows(lapply(source_cols, function(sc) {
      lv <- purch[[sc]]
      if (!is.factor(lv) && is.numeric(lv)) lv <- factor(lv, levels = c("No","Yes"))
      data.frame(source = sc, level = lv, SpendingClass = purch$SpendingClass)
    }))
    p_sources_sc <- ggplot(src_sc, aes(x = level, fill = SpendingClass)) +
      geom_bar(position = "fill") +
      scale_y_continuous(labels = scales::percent_format()) +
      facet_wrap(~ source, ncol = 4) +
      labs(title = "SpendingClass (Low/High) Share by Source (Purchasers)", x = "", y = "Share")
    print(p_sources_sc)
  }
}

# Predictor vs Outcome visuals
pal_sc <- c(Low = "#88B0F0", High = "#F36E6E")

scale_y_percent <- function(...) scale_y_continuous(labels = scales::percent, ...)

box_by_class <- function(data, num_var, class_var, title_suffix="") {
  ggplot(data, aes(x = .data[[class_var]], y = .data[[num_var]])) +
    geom_boxplot(fill = "white", color = "black") +
    labs(title = paste(num_var, "by", class_var, title_suffix),
         x = class_var, y = num_var) +
    theme_classic(base_size = 12)
}

share_bar <- function(data, cat_var, class_var, title_suffix = "", fill_vals = NULL) {
  g <- ggplot(data, aes(x = .data[[cat_var]], fill = .data[[class_var]])) +
    geom_bar(position = "fill", color = "black", linewidth = 0.2) +
    scale_y_percent(name = "Share") +
    labs(title = paste(class_var, "Share by", cat_var, title_suffix),
         x = cat_var, fill = class_var) +
    theme_classic(base_size = 12)
  if (!is.null(fill_vals)) g <- g + scale_fill_manual(values = fill_vals)
  g
}

# Predcitor VS PURCHASE plots
num_pred <- setdiff(names(df)[sapply(df, is.numeric)], c("Spending"))
for (v in num_pred) {
  print(box_by_class(df, num_var = v, class_var = "Purchase"))
}
cat_pred <- setdiff(names(df)[sapply(df, is.factor)], c("Purchase","SpendingClass", source_cols))
for (v in cat_pred) {
  print(share_bar(df, cat_var = v, class_var = "Purchase"))
}

# Predictor VS SPENDINGCLASS (purchasers only) plots
if ("SpendingClass" %in% names(df)) {
  purch_only_plot <- df %>% dplyr::filter(Purchase == "Purchase")
  for (v in num_pred) {
    print(box_by_class(purch_only_plot, num_var = v, class_var = "SpendingClass",
                                  title_suffix = "(Purchasers)"))
  }
  for (v in cat_pred) {
    print(share_bar(purch_only_plot, cat_var = v, class_var = "SpendingClass",
                               title_suffix = "(Purchasers)", fill_vals = pal_sc))
  }
}

# Functions for binary evaluation/plots
eval_bin_models <- function(model, test_df, outcome = "Purchase", pos_class) {
  pr <- predict(model, newdata = test_df, type = "prob")[, pos_class]
  y_test <- test_df[[outcome]]
  neg_class <- setdiff(levels(y_test), pos_class)
  pred <- factor(ifelse(pr >= 0.5, pos_class, neg_class), levels = levels(y_test))
  roc_obj <- pROC::roc(y_test, pr, quiet = TRUE)
  auc <- as.numeric(pROC::auc(roc_obj))
  cm  <- caret::confusionMatrix(pred, y_test, positive = pos_class)
  list(Probs = pr, Pred = pred, ROC = roc_obj, AUC = auc, CM = cm)
}

cm_heat_bin <- function(cm_obj, title) {
  m <- as.data.frame(cm_obj$table)
  ggplot(m, aes(Prediction, Reference, fill = Freq)) +
    geom_tile() + geom_text(aes(label = Freq)) +
    scale_fill_gradient(low = "white", high = "grey40") +
    labs(title = title, x = "Predicted", y = "Actual")
}

perf_from_cm_bin <- function(cm_obj) {
  byc <- cm_obj$byClass
  acc <- cm_obj$overall["Accuracy"]
  tibble::tibble(
    Accuracy     = as.numeric(acc),
    Sensitivity  = as.numeric(byc["Sensitivity"]),
    Specificity  = as.numeric(byc["Specificity"]),
    Precision    = as.numeric(byc["Pos Pred Value"]),
    F1           = ifelse((2*byc["Pos Pred Value"]*byc["Sensitivity"]) > 0,
                          2 * (byc["Pos Pred Value"] * byc["Sensitivity"]) /
                            (byc["Pos Pred Value"] + byc["Sensitivity"]), NA_real_)
  )
}

vi_plot <- function(fit, title, top_k = 10) {
  # Try varImp safely
  vi_try <- try(caret::varImp(fit), silent = TRUE)
  if (inherits(vi_try, "try-error") || is.null(vi_try$importance)) {
    message("No varImp available for model: ", title)
    return(invisible(NULL))
  }
  
  # Convert to plain data frame
  vi <- as.data.frame(vi_try$importance)
  vi$Variable <- rownames(vi)
  
  # Identify score columns (can be 1 or multiple)
  score_cols <- setdiff(names(vi), "Variable")
  
  # Compute a single score
  vi$Score <- if (length(score_cols) == 1) {
    vi[[score_cols]]
  } else {
    rowMeans(vi[, score_cols, drop = FALSE], na.rm = TRUE)
  }
  
  # Sort by importance
  vi <- vi[order(-vi$Score), , drop = FALSE]
  
  # Keep only top_k
  top_k <- min(top_k, nrow(vi))
  vi <- vi[seq_len(top_k), , drop = FALSE]
  
  # Plot
  ggplot(vi, aes(x = reorder(Variable, Score), y = Score)) +
    geom_col(fill = "steelblue") +
    coord_flip() +
    labs(
      title = title,
      x = NULL,
      y = "Importance"
    ) +
    theme_minimal(base_size = 12)
}


# PURCHASE MODELS
# Logistic + CART + RF + GBM

set.seed(42)

# Train/Test Split (70/30, stratified)
split_idx <- caret::createDataPartition(df$Purchase, p = 0.70, list = FALSE)
train_pur <- df[split_idx, ] %>% dplyr::select(-Spending, -SpendingClass)
test_pur  <- df[-split_idx, ] %>% dplyr::select(-Spending, -SpendingClass)

# caret control (AUC metric)
ctrl_bin <- trainControl(
  method = "cv", number = 5,
  classProbs = TRUE, summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# Train models
set.seed(42)
fit_glm <- caret::train(Purchase ~ ., data = train_pur,
                        method = "glm", family = binomial(),
                        trControl = ctrl_bin, metric = "ROC")

set.seed(42)
fit_cart <- caret::train(Purchase ~ ., data = train_pur,
                         method = "rpart", tuneLength = 15,
                         trControl = ctrl_bin, metric = "ROC")

set.seed(42)
fit_rf <- caret::train(Purchase ~ ., data = train_pur,
                       method = "rf", tuneLength = 5,
                       trControl = ctrl_bin, metric = "ROC",
                       importance = TRUE)

set.seed(42)
fit_gbm <- caret::train(Purchase ~ ., data = train_pur,
                        method = "gbm", distribution = "bernoulli",
                        trControl = ctrl_bin, metric = "ROC",
                        verbose = FALSE, tuneLength = 10)

# Evaluate all four
res_glm  <- eval_bin_models(fit_glm,  test_pur, outcome = "Purchase", pos_class = "Purchase")
res_cart <- eval_bin_models(fit_cart, test_pur, outcome = "Purchase", pos_class = "Purchase")
res_rf   <- eval_bin_models(fit_rf,   test_pur, outcome = "Purchase", pos_class = "Purchase")
res_gbm  <- eval_bin_models(fit_gbm,  test_pur, outcome = "Purchase", pos_class = "Purchase")

# Summaries of all models
cat("\n PURCHASE: MODEL SUMMARIES \n")
cat("\n Logistic GLM (summary of final model) \n"); print(summary(fit_glm$finalModel))
cat("\n CART: Best Tune \n"); print(fit_cart$bestTune); cat("\n"); printcp(fit_cart$finalModel)
cat("\n Random Forest \n"); print(fit_rf$finalModel)
cat("\n GBM: Best Tune \n"); print(fit_gbm$bestTune); cat("\n"); print(fit_gbm$finalModel)

# Metrics table
purchase_perf <- dplyr::bind_rows(
  tibble::tibble(Model = "Logistic",     AUC = res_glm$AUC)  %>% dplyr::bind_cols(perf_from_cm_bin(res_glm$CM)),
  tibble::tibble(Model = "CART",         AUC = res_cart$AUC) %>% dplyr::bind_cols(perf_from_cm_bin(res_cart$CM)),
  tibble::tibble(Model = "RandomForest", AUC = res_rf$AUC)   %>% dplyr::bind_cols(perf_from_cm_bin(res_rf$CM)),
  tibble::tibble(Model = "GBM",          AUC = res_gbm$AUC)  %>% dplyr::bind_cols(perf_from_cm_bin(res_gbm$CM))
)
cat("\n PURCHASE: Metrics (Test set, th=0.5) \n"); print(purchase_perf)

cat("\n PURCHASE: Confusion Matrices (th=0.5) \n")
print(res_glm$CM$table); print(res_cart$CM$table); print(res_rf$CM$table); print(res_gbm$CM$table)

# Plots: ROC overlay + CM heatmaps
plot(res_glm$ROC, col='red', lwd = 2, main = "ROC - Purchase (Logistic/CART/RF/GBM)")
plot(res_cart$ROC, col='orange', lwd = 2, add = TRUE)
plot(res_rf$ROC,  col='skyblue', lwd = 2, add = TRUE)
plot(res_gbm$ROC, col='green', lwd = 2, add = TRUE)
legend("bottomright",
       legend = c(sprintf("Logistic (AUC=%.3f)", res_glm$AUC),
                  sprintf("CART (AUC=%.3f)",     res_cart$AUC),
                  sprintf("RF (AUC=%.3f)",       res_rf$AUC),
                  sprintf("GBM (AUC=%.3f)",      res_gbm$AUC)),
       lwd = 2, col = c("red","orange","skyblue","green"))

print(cm_heat_bin(res_glm$CM,  "Confusion Matrix - Purchase (Logistic)"))
print(cm_heat_bin(res_cart$CM, "Confusion Matrix - Purchase (CART)"))
print(cm_heat_bin(res_rf$CM,   "Confusion Matrix - Purchase (RF)"))
print(cm_heat_bin(res_gbm$CM,  "Confusion Matrix - Purchase (GBM)"))

# Variable-importance 
print(vi_plot(fit_glm,  "Variable Importance - Purchase (Logistic)"))
print(vi_plot(fit_cart, "Variable Importance - Purchase (CART)"))
print(vi_plot(fit_rf,   "Variable Importance - Purchase (Random Forest)"))
print(vi_plot(fit_gbm,  "Variable Importance - Purchase (GBM)"))

# SPENDINGCLASS (Binary: Low vs High)
# Logistic + CART + RF + GBM (purchasers only)
set.seed(43)

# Filter purchasers & split
purch_only <- df %>% dplyr::filter(Purchase == "Purchase") %>% dplyr::select(-Spending)

# Ensure factor and positive class "High"
if (is.factor(purch_only$SpendingClass)) {
  purch_only$SpendingClass <- relevel(purch_only$SpendingClass, ref = "High")
}

split_sc <- caret::createDataPartition(purch_only$SpendingClass, p = 0.70, list = FALSE)
train_sc <- purch_only[split_sc, ]
test_sc  <- purch_only[-split_sc, ]

ctrl_sc <- trainControl(method = "cv", number = 5,
                        classProbs = TRUE, summaryFunction = twoClassSummary,
                        savePredictions = "final")

# Train models
set.seed(43)
fit_glm_sc <- caret::train(SpendingClass ~ ., data = train_sc,
                           method = "glm", family = binomial(),
                           trControl = ctrl_sc, metric = "ROC")

set.seed(43)
fit_cart_sc <- caret::train(SpendingClass ~ ., data = train_sc,
                            method = "rpart", tuneLength = 15,
                            trControl = ctrl_sc, metric = "ROC")

set.seed(43)
fit_rf_sc <- caret::train(SpendingClass ~ ., data = train_sc,
                          method = "rf", tuneLength = 5,
                          trControl = ctrl_sc, metric = "ROC",
                          importance = TRUE)

set.seed(43)
fit_gbm_sc <- caret::train(SpendingClass ~ ., data = train_sc,
                           method = "gbm", distribution = "bernoulli",
                           trControl = ctrl_sc, metric = "ROC",
                           verbose = FALSE, tuneLength = 10)

# Evaluate model performance
res_glm_sc  <- eval_bin_models(fit_glm_sc,  test_sc, outcome = "SpendingClass", pos_class = "High")
res_cart_sc <- eval_bin_models(fit_cart_sc, test_sc, outcome = "SpendingClass", pos_class = "High")
res_rf_sc   <- eval_bin_models(fit_rf_sc,   test_sc, outcome = "SpendingClass", pos_class = "High")
res_gbm_sc  <- eval_bin_models(fit_gbm_sc,  test_sc, outcome = "SpendingClass", pos_class = "High")

# Summaries of models
cat("\n SPENDINGCLASS (Low/High): MODEL SUMMARIES \n")
cat("\n Logistic GLM (summary of final model) \n"); print(summary(fit_glm_sc$finalModel))
cat("\n CART: Best Tune \n"); print(fit_cart_sc$bestTune); cat("\n"); printcp(fit_cart_sc$finalModel)
cat("\n Random Forest \n"); print(fit_rf_sc$finalModel)
cat("\n GBM: Best Tune \n"); print(fit_gbm_sc$bestTune); cat("\n"); print(fit_gbm_sc$finalModel)

# Metrics table
sc_perf <- dplyr::bind_rows(
  tibble::tibble(Model = "Logistic",     AUC = res_glm_sc$AUC)  %>% dplyr::bind_cols(perf_from_cm_bin(res_glm_sc$CM)),
  tibble::tibble(Model = "CART",         AUC = res_cart_sc$AUC) %>% dplyr::bind_cols(perf_from_cm_bin(res_cart_sc$CM)),
  tibble::tibble(Model = "RandomForest", AUC = res_rf_sc$AUC)   %>% dplyr::bind_cols(perf_from_cm_bin(res_rf_sc$CM)),
  tibble::tibble(Model = "GBM",          AUC = res_gbm_sc$AUC)  %>% dplyr::bind_cols(perf_from_cm_bin(res_gbm_sc$CM))
)
cat("\n SPENDINGCLASS (Low/High): Metrics (Test set, th=0.5) \n"); print(sc_perf)

cat("\n SPENDINGCLASS (Low/High): Confusion Matrices (th=0.5) \n")
print(res_glm_sc$CM$table); print(res_cart_sc$CM$table); print(res_rf_sc$CM$table); print(res_gbm_sc$CM$table)

# ROC overlay + CM heatmaps
plot(res_glm_sc$ROC,  col = "red", lwd = 2, main = "ROC - SpendingClass (Low/High)")
plot(res_cart_sc$ROC, col = "orange",   lwd = 2, add = TRUE)
plot(res_rf_sc$ROC,   col = "skyblue",  lwd = 2, add = TRUE)
plot(res_gbm_sc$ROC,  col = "green", lwd = 2, add = TRUE)

legend("bottomright",
       legend = c(sprintf("Logistic (AUC=%.3f)", res_glm_sc$AUC),
                  sprintf("CART (AUC=%.3f)",     res_cart_sc$AUC),
                  sprintf("RF (AUC=%.3f)",       res_rf_sc$AUC),
                  sprintf("GBM (AUC=%.3f)",      res_gbm_sc$AUC)),
       lwd = 2, col = c("red","orange","skyblue","green"))

# Variable-importance (top 10)
print(vi_plot(fit_glm_sc,  "Variable Importance - SpendingClass (Logistic)"))
print(vi_plot(fit_cart_sc, "Variable Importance - SpendingClass (CART)"))
print(vi_plot(fit_rf_sc,   "Variable Importance - SpendingClass (Random Forest)"))
print(vi_plot(fit_gbm_sc,  "Variable Importance - SpendingClass (GBM)"))

# Exploratory K-means Clustering 
cluster_vars <- c("Freq","last_update_days_ago","X1st_update_days_ago")
km_df <- df %>% dplyr::select(dplyr::all_of(cluster_vars)) %>% scale() %>% as.data.frame()

set.seed(1)
wss <- sapply(1:8, function(k) kmeans(km_df, centers = k, nstart = 25)$tot.withinss)
plot(1:8, wss, type = "b", pch = 19, xlab = "k", ylab = "Total within-clusters SS",
     main = "K-means Elbow Plot")

set.seed(2)
km3 <- kmeans(km_df, centers = 3, nstart = 50)
df$Cluster <- factor(km3$cluster)

# Crosstabs (print only)
cm1 <- as.table(with(df, table(Cluster, Purchase)))
cat("\nClusters vs Purchase:\n"); print(cm1)
if ("SpendingClass" %in% names(df)) {
  cm2 <- as.table(with(df[df$Purchase=="Purchase",], table(Cluster, SpendingClass)))
  cat("\nClusters vs SpendingClass (purchasers):\n"); print(cm2)
}

# Mean profile of each cluster
cluster_profile <- df %>%
  group_by(Cluster) %>%
  summarise(across(all_of(cluster_vars), mean, na.rm = TRUE),
            PurchaseRate = mean(Purchase == "Purchase"))

print(cluster_profile)

# Visualization
ggplot(df, aes(x = Cluster, fill = Purchase)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Purchase Rate by Cluster",
       y = "Share", x = "Cluster") +
  theme_minimal()


# Function to collect variable importance from all caret models
varimp_df <- function(fit, model_name, outcome, top_k = 10) {
  vi_try <- try(caret::varImp(fit), silent = TRUE)
  if (inherits(vi_try, "try-error") || is.null(vi_try$importance)) {
    return(dplyr::tibble())
  }
  vi <- as.data.frame(vi_try$importance)
  vi <- tibble::rownames_to_column(vi, "Variable")
  score_cols <- setdiff(names(vi), "Variable")
  vi <- vi %>%
    dplyr::mutate(Score = if (length(score_cols) == 1) .data[[score_cols]] else rowMeans(dplyr::across(all_of(score_cols)), na.rm = TRUE)) %>%
    dplyr::select(Variable, Score) %>%
    dplyr::arrange(dplyr::desc(Score)) %>%
    dplyr::slice(1:min(top_k, dplyr::n())) %>%
    dplyr::mutate(Model = model_name, Outcome = outcome, .before = 1)
  return(vi)
}

# Summary Plot – Variable Importance Across All Models
vi_all <- dplyr::bind_rows(
  varimp_df(fit_glm,   "Logistic",     "Purchase",      top_k = 10),
  varimp_df(fit_cart,  "CART",         "Purchase",      top_k = 10),
  varimp_df(fit_rf,    "Random Forest","Purchase",      top_k = 10),
  varimp_df(fit_gbm,   "GBM",          "Purchase",      top_k = 10),
  varimp_df(fit_glm_sc,"Logistic",     "SpendingClass", top_k = 10),
  varimp_df(fit_cart_sc,"CART",        "SpendingClass", top_k = 10),
  varimp_df(fit_rf_sc, "Random Forest","SpendingClass", top_k = 10),
  varimp_df(fit_gbm_sc,"GBM",          "SpendingClass", top_k = 10)
) %>%
  dplyr::group_by(Outcome, Model) %>%
  dplyr::mutate(NormScore = Score / max(Score, na.rm = TRUE)) %>%
  dplyr::ungroup()

# To keep the heatmap readable, keep variables that were top in any panel
top_vars <- vi_all %>%
  dplyr::group_by(Variable) %>%
  dplyr::summarise(MaxAcross = max(NormScore, na.rm = TRUE), .groups = "drop") %>%
  dplyr::arrange(dplyr::desc(MaxAcross)) %>%
  dplyr::slice(1:min(20, dplyr::n())) %>% # show top 20 overall
  dplyr::pull(Variable)

vi_show <- vi_all %>% dplyr::filter(Variable %in% top_vars)

# Order models nicely
vi_show$Model <- factor(vi_show$Model, levels = c("Logistic","CART","Random Forest","GBM"))

# Order variables within each outcome by average importance
avg_by_outcome <- vi_show %>%
  dplyr::group_by(Outcome, Variable) %>%
  dplyr::summarise(Avg = mean(NormScore, na.rm = TRUE), .groups = "drop")

vi_show <- vi_show %>%
  dplyr::left_join(avg_by_outcome, by = c("Outcome","Variable")) %>%
  dplyr::arrange(Outcome, Avg) %>%
  dplyr::mutate(VarOrder = paste(Outcome, Variable, sep = "||"))

ggplot(vi_show, aes(x = Model, y = reorder(VarOrder, Avg), fill = NormScore)) +
  geom_tile(color = "white") +
  scale_y_discrete(labels = function(z) sub("^[^|]+\\|\\|", "", z)) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  facet_wrap(~ Outcome, scales = "free_y") +
  labs(
    title = "Summary of Variable Importance Across All Models",
    subtitle = "Normalized within each model panel; darker = more important",
    x = NULL, y = NULL, fill = "Rel. Importance"
  ) +
  theme_minimal(base_size = 12)

# Calibration Plot of Predicted vs. Actual Purchase Probability (GBM)
cal_df <- dplyr::tibble(
  prob   = res_gbm$Probs,
  actual = as.integer(test_pur$Purchase == "Purchase")
) %>%
  dplyr::mutate(bin = dplyr::ntile(prob, 10)) %>%
  dplyr::group_by(bin) %>%
  dplyr::summarise(
    mean_pred = mean(prob, na.rm = TRUE),
    obs_rate  = mean(actual, na.rm = TRUE),
    n = dplyr::n(), .groups = "drop"
  )

ggplot(cal_df, aes(x = mean_pred, y = obs_rate)) +
  geom_abline(slope = 1, intercept = 0, linetype = 2) +
  geom_point(size = 2) +
  geom_line() +
  scale_x_continuous(labels = scales::percent_format(accuracy = 1)) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(
    title = "Calibration Predicted vs. Actual Purchase Probability (GBM)",
    subtitle = "Decile-binned on the test set",
    x = "Mean predicted probability",
    y = "Observed purchase rate"
  ) +
  theme_minimal(base_size = 12)

# Residual Distribution of SpendingClass Predictions (GBM)
resid_sc_df <- dplyr::tibble(
  p_high = res_gbm_sc$Probs,
  y_high = as.integer(test_sc$SpendingClass == "High")
) %>%
  dplyr::mutate(residual = y_high - p_high)

ggplot(resid_sc_df, aes(x = residual)) +
  geom_histogram(bins = 30, color = "white") +
  geom_vline(xintercept = 0, linetype = 2) +
  labs(
    title = "Residual Distribution of SpendingClass (GBM)",
    subtitle = "Residual = observed (High=1/Low=0) of predicted probability of High",
    x = "Residual", y = "Count"
  ) +
  theme_minimal(base_size = 12)

# Cluster Membership vs. Predicted Purchase Probability (GBM)
df_pred_pur <- df %>% dplyr::select(-dplyr::any_of(c("Spending","SpendingClass")))
gbm_prob_all <- predict(fit_gbm, newdata = df_pred_pur, type = "prob")[, "Purchase"]

cluster_prob_df <- dplyr::tibble(
  Cluster = df$Cluster,
  p_purchase = gbm_prob_all
) %>%
  dplyr::filter(!is.na(Cluster))

ggplot(cluster_prob_df, aes(x = Cluster, y = p_purchase, fill = Cluster)) +
  geom_violin(trim = FALSE, alpha = 0.7, color = NA) +
  geom_boxplot(width = 0.15, outlier.alpha = 0, fill = "white", color = "black") +
  stat_summary(fun = mean, geom = "point", size = 2, color = "black") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(
    title = "Cluster Membership vs. Predicted Purchase Probability (GBM)",
    x = "Cluster", y = "Predicted purchase probability"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")


