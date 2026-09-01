## ============================================================
## Titanic Dataset — Week 3: Statistical Analysis & Predictive Modeling
## ============================================================
## Steps:
##   1. Hypothesis Testing (Chi-Square, t-Test)
##   2. Normality / Distribution Checks (Shapiro-Wilk, Q-Q plot)
##   3. Logistic Regression Model + Train/Test Split + Confusion Matrix
##   4. 10-Fold Cross-Validation (caret)
##   5. Residual Diagnostics
##   6. ROC Curve, AUC, Precision/Recall/F1
## ============================================================

## ---- 0. Setup ----
# install.packages(c("dplyr", "caret", "pROC"))  # run once if not installed
library(dplyr)
library(caret)
library(pROC)

# df should already contain (adjust to your actual loading step):
#   Survived (0/1), Pclass (factor: 1st/2nd/3rd), Sex, Age, Fare,
#   Age_norm, Fare_norm (normalized versions), HasCabin (0/1)
#
# Example loading + basic feature engineering, if starting from raw titanic.csv:
# df <- read.csv("titanic.csv")
# df$HasCabin  <- ifelse(df$Cabin == "" | is.na(df$Cabin), 0, 1)
# df$Age[is.na(df$Age)] <- median(df$Age, na.rm = TRUE)
# df$Age_norm  <- scale(df$Age)[, 1]
# df$Fare_norm <- scale(df$Fare)[, 1]
# df$Pclass    <- factor(df$Pclass, labels = c("1st", "2nd", "3rd"))


## ============================================================
## STEP 1: HYPOTHESIS TESTING
## ============================================================

## Chi-square test: is survival associated with passenger class?
chisq_result <- chisq.test(table(df$Pclass, df$Survived))
print(chisq_result)

## Welch t-test: does mean fare differ between survivors and non-survivors?
t_test_result <- t.test(Fare ~ Survived, data = df)
print(t_test_result)


## ============================================================
## STEP 2: NORMALITY / DISTRIBUTION ASSUMPTION CHECKS
## ============================================================

shapiro.test(df$Fare)
shapiro.test(df$Age)

qqnorm(df$Fare, main = "Normal Q-Q Plot - Fare")
qqline(df$Fare)


## ============================================================
## STEP 3: LOGISTIC REGRESSION MODEL (TRAIN/TEST SPLIT)
## ============================================================

set.seed(42)
train_index <- sample(nrow(df), 0.7 * nrow(df))
train_data  <- df[train_index, ]
test_data   <- df[-train_index, ]

logistic_model <- glm(
  Survived ~ Pclass + Sex + Age_norm + Fare_norm + HasCabin,
  data   = train_data,
  family = binomial
)
summary(logistic_model)

## Predict on held-out test set
test_data$pred_prob  <- predict(logistic_model, newdata = test_data, type = "response")
test_data$pred_class <- ifelse(test_data$pred_prob > 0.5, 1, 0)

## Confusion matrix + accuracy
conf_matrix <- table(Actual = test_data$Survived, Predicted = test_data$pred_class)
print(conf_matrix)

accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
cat("Model Accuracy:", round(accuracy * 100, 2), "%\n")


## ============================================================
## STEP 4: 10-FOLD CROSS-VALIDATION (caret)
## ============================================================

## IMPORTANT: Survived must be a factor so caret runs classification
## (Accuracy/Kappa) instead of regression (RMSE/R-squared/MAE).
df$Survived_factor <- factor(df$Survived, labels = c("No", "Yes"))

ctrl <- trainControl(method = "cv", number = 10)

cv_model <- train(
  Survived_factor ~ Pclass + Sex + Age_norm + Fare_norm + HasCabin,
  data      = df,
  method    = "glm",
  family    = "binomial",
  trControl = ctrl
)
print(cv_model)


## ============================================================
## STEP 5: RESIDUAL DIAGNOSTICS
## ============================================================

plot(
  fitted(logistic_model),
  residuals(logistic_model, type = "pearson"),
  xlab = "Predicted values",
  ylab = "Pearson Residuals",
  main = "Residuals vs Fitted"
)
abline(h = 0, col = "red", lty = 2)


## ============================================================
## STEP 6: ROC CURVE, AUC, PRECISION / RECALL / F1
## ============================================================

roc_obj <- roc(test_data$Survived, test_data$pred_prob)
plot(roc_obj, main = "ROC Curve for Logistic Regression Model", col = "blue", print.auc = TRUE)

auc_val <- auc(roc_obj)
print(auc_val)

tp <- conf_matrix[2, 2]
tn <- conf_matrix[1, 1]
fp <- conf_matrix[1, 2]
fn <- conf_matrix[2, 1]

precision <- tp / (tp + fp)
recall    <- tp / (tp + fn)
f1_score  <- 2 * (precision * recall) / (precision + recall)

cat("Precision:", round(precision, 4), "\n")
cat("Recall:", round(recall, 4), "\n")
cat("F1-Score:", round(f1_score, 4), "\n")
cat("AUC:", round(as.numeric(auc_val), 4), "\n")
