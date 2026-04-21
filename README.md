# Telecom Churn Hyperparameter Tuning & Nested Cross-Validation

This repository contains an Honors-level assignment focusing on robust model evaluation using GridSearchCV and Nested Cross-Validation.

## Part 1: GridSearchCV Analysis (Random Forest)

### Hyperparameter Impact
In this grid search, **max_depth** appeared to be the hyperparameter with the most significant impact on the F1 score. The "sweet spot" for this dataset was **max_depth: 5**. At this depth, the model captures enough complexity to identify churn patterns without succumbing to the noise of the simulated features. The model preferred a smaller forest (**n_estimators: 50**), suggesting that for this 1000-sample dataset, adding more trees provided diminishing returns.

### Overfitting vs. Underfitting
We observed signs of **underfitting** at `max_depth: 3` (low scores) and potential **overfitting** at higher depths (10+) where the model might begin to track individual sample quirks. The plateau at 50 trees indicates that the model is well-regularized.

### Recommendations
To expand the grid, I would recommend testing `max_depth` in the range of [4, 5, 6, 7] and exploring `max_features` (e.g., 'sqrt' vs 'log2') to see if further decorrelation of the trees can improve the F1 score.

---

## Part 2: Nested Cross-Validation Results

### Comparison Table

| Metric | Random Forest | Decision Tree |
| :--- | :--- | :--- |
| **Mean Inner best_score_** | 0.6860 | 0.5637 |
| **Mean Outer Nested CV Score** | 0.7135 | 0.6013 |
| **Gap (Inner - Outer)** | -0.0275 | -0.0376 |

### Nested CV Analysis
The **Random Forest** (Outer Score: 0.7135) significantly outperformed the **Decision Tree** (Outer Score: 0.6013). This highlights the robustness of ensemble methods; while a single tree is prone to high variance, the Random Forest stabilizes predictions by averaging multiple decorrelated trees.

### Selection Bias & The "Honest Evaluator"
The **Decision Tree had a larger gap magnitude (0.037 vs 0.027)**, confirming that it is more sensitive to data fluctuations and thus less stable during hyperparameter tuning.

**Nested Cross-Validation** acts as an **"honest evaluator"** because it never evaluates a model on data that was used to pick its hyperparameters. By separating the *tuning* (inner loop) from the *performance estimation* (outer loop), Nested CV provides a realistic expectation of how the machine learning pipeline will generalize to truly unseen data.

---

## Files in this Repository
- `churn_nested_cv.py`: Complete Python script for simulation and tuning.
- `rf_heatmap.png`: Visualization of the Random Forest grid search results.
- `README.md`: Analysis and summary of results.
