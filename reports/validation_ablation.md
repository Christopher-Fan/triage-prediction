# Validation Strategy & Metric Ablation Study

## 1. Objective Function & Model Benchmarking

Emergency Department triage suffers from severe class imbalance: high-acuity resuscitation cases (KTAS 1) represent a tiny fraction of total admissions compared to non-urgent visits (KTAS 3/4). 

Evaluating models purely on Accuracy causes classifiers to predict majority classes and completely miss life-threatening presentations. To guarantee clinical viability across all acuity tiers, we evaluate models using **Macro F1-Score** and **Resuscitation (KTAS 1) Sensitivity** across **Stratified 5-Fold Cross-Validation**.

### Experimental Ablation Results ($N = 567$)

| Model / Configuration | Feature Scaling | Imbalance Strategy | Macro F1-Score | Resuscitation (KTAS 1) Sensitivity |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Baseline)** | `StandardScaler` | `class_weight="balanced"` | 0.2366 | 50.0% |
| **Default Random Forest** | None | `class_weight="balanced"` | 0.2547 | 0.0% |
| **Optuna-Tuned Random Forest** | None | `class_weight="balanced"` | **0.2715** | **30.0%** |

---

## 2. Key Experimental Insights

1. **Feature Scaling Failure Mode:** Unscaled vital signs cause LBFGS gradient optimization in Logistic Regression to fail convergence. Standardizing inputs stabilizes training and establishes high KTAS 1 Sensitivity ($50.0\%$), though overall Macro-F1 remains low ($0.2366$).
2. **Default Tree Failure Mode:** Default Random Forest hyperparameter configurations prioritize majority classes, resulting in complete failure on critical resuscitation cases ($0.0\%$ Sensitivity).
3. **Hyperparameter Optimization Recovery:** Optuna optimization (`n_estimators=122`, `max_depth=16`, `min_samples_split=10`, `min_samples_leaf=3`) paired with balanced class weighting achieves the optimal balance—maximizing overall Macro F1-Score ($0.2715$) while maintaining a $30.0\%$ sensitivity on life-threatening cases.