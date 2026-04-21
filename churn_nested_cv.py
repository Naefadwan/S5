import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

# Set plotting style
sns.set_theme(style="whitegrid")


data = pd.read_csv('data/telecom_churn.csv')
df = data.drop(columns=['customer_id'])

# 1. Preprocessing: Convert total_charges to numeric and handle missing values
df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
df = df.dropna()

# 2. Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

y = df['churned'].values
X = df.drop(columns=['churned']).values

print(f"Dataset Shape: {X.shape}")
print(f"Class Distribution:\n{pd.Series(y).value_counts(normalize=True)}")




# =============================================================================
# PART 1 — GridSearchCV (Random Forest)
# =============================================================================

# 1. Initialize RandomForestClassifier
rf = RandomForestClassifier(class_weight='balanced', random_state=42)

# 2. Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, 20, None],
    'min_samples_split': [2, 5, 10]
}

# 3. Use GridSearchCV
cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(class_weight='balanced', random_state=42),
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1',
    n_jobs=-1,
    return_train_score=True
)

# 4. Fit and report
grid_search.fit(X, y)

print("\n--- Part 1: GridSearchCV Results (Random Forest) ---")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best F1 Score: {grid_search.best_score_:.4f}")

# 5. Create Heatmap
# To visualize 3D params in 2D, we fix min_samples_split to its best value.
# Justification: Fixing the third variable allows us to see the clear interaction 
# between tree depth and forest size, which are typically the most critical 
# hyperparameters for Random Forest performance and overfitting control.
best_mss = grid_search.best_params_['min_samples_split']
cv_results = pd.DataFrame(grid_search.cv_results_)

# Filtering for the best min_samples_split
pivot_data = cv_results[cv_results['param_min_samples_split'] == best_mss].copy()
# Ensure params are treated as strings or numbers properly for pivoting
pivot_data['param_max_depth'] = pivot_data['param_max_depth'].fillna('None')
pivot_table = pivot_data.pivot(index='param_max_depth', columns='param_n_estimators', values='mean_test_score')

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt=".3f")
plt.title(f'Mean F1 Score (min_samples_split={best_mss})')
plt.xlabel('n_estimators')
plt.ylabel('max_depth')
plt.tight_layout()
plt.savefig('rf_heatmap.png')
plt.show()

# =============================================================================
# PART 1 ANALYSIS
# =============================================================================
"""
### Part 1 Analysis

In this grid search, **max_depth** appears to be the hyperparameter with the most significant impact on the F1 score. 
While increasing **n_estimators** generally improves performance, the gains often plateau after 100 trees, 
whereas an incorrect **max_depth** (e.g., too shallow at 3) leads to significant underfitting. 
We see signs of a "sweet spot" at a depth where the F1 score peaks before marginally decreasing or stabilizing, 
suggesting that excessively deep trees (max_depth=None) might start capturing noise if not for the ensemble's 
averaging effect. To expand the grid, I would investigate finer increments for **max_depth** between 10 and 20, 
and perhaps explore **max_features** to further decorrelate the trees, as the current grid suggests we are 
nearing an optimal complexity for this specific feature set.
"""

# =============================================================================
# PART 2 — Nested Cross-Validation
# =============================================================================

def run_nested_cv(model, p_grid, X, y, outer_cv, inner_cv):
    outer_scores = []
    inner_scores = []
    
    # Custom loop to track both inner and outer scores per fold
    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Inner Loop: GridSearchCV
        inner_grid = GridSearchCV(
            estimator=model, 
            param_grid=p_grid, 
            cv=inner_cv, 
            scoring='f1', 
            n_jobs=-1
        )
        inner_grid.fit(X_train, y_train)
        
        # Record best score from inner validation
        inner_scores.append(inner_grid.best_score_)
        
        # Outer Loop: Evaluate on held-out fold
        best_est = inner_grid.best_estimator_
        y_pred = best_est.predict(X_test)
        outer_scores.append(f1_score(y_test, y_pred))
        
    return np.mean(inner_scores), np.mean(outer_scores)

# Setup CVs
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=99)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# A) Random Forest
rf_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, 20, None],
    'min_samples_split': [2, 5, 10]
}
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)

# B) Decision Tree
dt_grid = {
    'max_depth': [3, 5, 10, 20, None],
    'min_samples_split': [2, 5, 10]
}
dt_model = DecisionTreeClassifier(class_weight='balanced', random_state=42)

print("\nRunning Nested CV for Random Forest...")
rf_inner, rf_outer = run_nested_cv(rf_model, rf_grid, X, y, outer_cv, inner_cv)

print("Running Nested CV for Decision Tree...")
dt_inner, dt_outer = run_nested_cv(dt_model, dt_grid, X, y, outer_cv, inner_cv)

# =============================================================================
# RESULTS TABLE
# =============================================================================

results_df = pd.DataFrame({
    'Metric': ['Mean Inner best_score_', 'Mean Outer Nested CV Score', 'Gap (Inner - Outer)'],
    'Random Forest': [rf_inner, rf_outer, rf_inner - rf_outer],
    'Decision Tree': [dt_inner, dt_outer, dt_inner - dt_outer]
})

print("\n--- Part 2: Nested Cross-Validation Results ---")
print(results_df.to_string(index=False))

# =============================================================================
# PART 2 ANALYSIS
# =============================================================================
"""
### Part 2 Analysis

The **Decision Tree** typically exhibits a larger selection bias (Gap) compared to the Random Forest. 
This is because a single decision tree is highly sensitive to the specific training data and the 
search for optimal hyperparameters on small folds is more likely to \"overfit\" the validation set. 
The GridSearchCV `best_score_` is often overly optimistic because it reports the highest score 
encountered during the search, which includes a component of random chance from the validation splits.

Nested Cross-Validation serves as an **honest evaluator** by separating the tuning process (inner loop) 
from the performance estimation (outer loop). While standard GridSearchCV reports how well the model 
performed on the data it was tuned on, Nested CV simulates a truly \"held-out\" test set multiple times. 
It ensures that the reported accuracy reflects the model's ability to generalize to new data, 
effectively penalizing models that only perform well through \"lucky\" hyperparameter selection on specific subsets.
"""
