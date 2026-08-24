# %% [markdown]
# Support Vector Machine: Mushroom Classification
#
# This analysis covers EDA, preprocessing, visualization, SVM training, kernel
# comparison, hyperparameter tuning, evaluation, and practical interpretation.

# %%
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
try:
    from IPython.display import display
except ImportError:
    display = print
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
sns.set_theme(style="whitegrid", palette="deep")

# %% [markdown]
## 1. Load and Explore the Dataset

# %%
script_folder = Path(__file__).parent if "__file__" in globals() else Path.cwd()
data_path = script_folder / "mushroom.csv"
if not data_path.exists():
    data_path = Path("mushroom.csv")

df = pd.read_csv(data_path)
print(f"Dataset path: {data_path.resolve()}")
print(f"Shape: {df.shape}")
display(df.head())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isna().sum().sort_values(ascending=False).head())
print(f"\nDuplicate rows: {df.duplicated().sum()}")
print("\nClass counts:\n", df["class"].value_counts())
print("\nNumerical summary:")
display(df.describe(include="all").T)

# %% [markdown]
## 2. Exploratory Data Analysis and Visualization

# %%
numeric_features = df.select_dtypes(include=np.number).columns.tolist()
categorical_features = [column for column in df.columns if column not in numeric_features + ["class"]]

fig, axes = plt.subplots(1, len(numeric_features), figsize=(12, 4))
for axis, feature in zip(np.atleast_1d(axes), numeric_features):
    sns.histplot(data=df, x=feature, hue="class", kde=True, element="step", ax=axis)
    axis.set_title(f"Distribution of {feature}")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, len(numeric_features), figsize=(12, 4))
for axis, feature in zip(np.atleast_1d(axes), numeric_features):
    sns.boxplot(data=df, x="class", y=feature, ax=axis)
    axis.set_title(f"{feature} by class")
plt.tight_layout()
plt.show()

# Plot the most informative categorical features by cardinality and class.
selected_categoricals = sorted(
    categorical_features,
    key=lambda column: df[column].nunique(),
    reverse=True,
)[:6]
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for axis, feature in zip(axes.flat, selected_categoricals):
    counts = pd.crosstab(df[feature], df["class"], normalize="index")
    counts.plot(kind="bar", stacked=True, ax=axis, colormap="Set2")
    axis.set_title(f"Class proportions: {feature}")
    axis.set_ylabel("Proportion")
    axis.tick_params(axis="x", rotation=45)
for axis in axes.flat[len(selected_categoricals):]:
    axis.set_visible(False)
plt.tight_layout()
plt.show()

# %%
class_proportions = df["class"].value_counts(normalize=True).mul(100).round(2)
print("Class proportions (%):\n", class_proportions)
sns.countplot(data=df, x="class", order=df["class"].value_counts().index)
plt.title("Mushroom class distribution")
plt.ylabel("Count")
plt.show()

# One-hot encoding is used only for this exploratory correlation view.
eda_encoded = pd.get_dummies(df.drop(columns="class"), drop_first=False, dtype=int)
eda_encoded["class_encoded"] = df["class"].map({"edible": 0, "poisonous": 1})
correlations = eda_encoded.corr(numeric_only=True)[["class_encoded"]].drop("class_encoded")
strongest_correlations = correlations["class_encoded"].abs().sort_values(ascending=False).head(15)
plt.figure(figsize=(8, 7))
sns.heatmap(
    eda_encoded[strongest_correlations.index.tolist() + ["class_encoded"]].corr(),
    cmap="coolwarm",
    center=0,
    annot=False,
)
plt.title("Correlation heatmap: strongest one-hot relationships")
plt.tight_layout()
plt.show()

# %% [markdown]
# 3. Preprocessing and Train/Test Split
#
# The identifier-like `Unnamed: 0` column is removed because it is not a biological
# mushroom characteristic. Encoding and scaling are placed inside a pipeline so
# that the test set remains unseen during fitting.

# %%
features = df.drop(columns=["class", "Unnamed: 0"], errors="ignore")
target = df["class"]
numeric_features = features.select_dtypes(include=np.number).columns.tolist()
categorical_features = [column for column in features.columns if column not in numeric_features]

X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.20,
    stratify=target,
    random_state=RANDOM_STATE,
)

try:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
except TypeError:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_features),
        ("categorical", encoder, categorical_features),
    ],
)
print(f"Training rows: {len(X_train)} | Testing rows: {len(X_test)}")
print("Training class proportions:\n", y_train.value_counts(normalize=True))

# %% [markdown]
## 4. Baseline SVM Classifier

# %%
baseline_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale")),
    ]
)
baseline_model.fit(X_train, y_train)
baseline_predictions = baseline_model.predict(X_test)

print(classification_report(y_test, baseline_predictions, zero_division=0))
ConfusionMatrixDisplay.from_predictions(y_test, baseline_predictions, cmap="Blues")
plt.title("Baseline RBF SVM confusion matrix")
plt.show()

# %% [markdown]
## 5. Visualize Feature Relationships and SVM Results

# %%
sns.pairplot(df, vars=numeric_features, hue="class", corner=True, height=3)
plt.suptitle("Numeric feature relationships by mushroom class", y=1.02)
plt.show()

# A 2D model provides an interpretable decision-region view of the numeric features.
X_numeric = df[numeric_features]
X_num_train, X_num_test, y_num_train, y_num_test = train_test_split(
    X_numeric,
    target,
    test_size=0.20,
    stratify=target,
    random_state=RANDOM_STATE,
)
visual_model = Pipeline(
    steps=[("scaler", StandardScaler()), ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale"))]
)
visual_model.fit(X_num_train, y_num_train)

x_min, x_max = X_numeric.iloc[:, 0].min() - 1, X_numeric.iloc[:, 0].max() + 1
y_min, y_max = X_numeric.iloc[:, 1].min() - 1, X_numeric.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
grid_predictions = visual_model.predict(pd.DataFrame({numeric_features[0]: xx.ravel(), numeric_features[1]: yy.ravel()}))
class_codes = {label: index for index, label in enumerate(sorted(target.unique()))}
zz = np.array([class_codes[label] for label in grid_predictions]).reshape(xx.shape)
plt.figure(figsize=(9, 6))
plt.contourf(xx, yy, zz, alpha=0.25, cmap="coolwarm")
sns.scatterplot(data=df, x=numeric_features[0], y=numeric_features[1], hue="class", alpha=0.65)
plt.title("RBF SVM decision regions using numeric features")
plt.show()

# %% [markdown]
## 6. Compare SVM Kernels

# %%
def evaluate_model(model, name):
    started = perf_counter()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return {
        "kernel": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, pos_label="poisonous", zero_division=0),
        "recall": recall_score(y_test, predictions, pos_label="poisonous", zero_division=0),
        "f1_score": f1_score(y_test, predictions, pos_label="poisonous", zero_division=0),
        "fit_time_seconds": perf_counter() - started,
    }

kernel_models = {
    "linear": SVC(kernel="linear", C=1.0),
    "polynomial": SVC(kernel="poly", C=1.0, degree=3, gamma="scale"),
    "rbf": SVC(kernel="rbf", C=1.0, gamma="scale"),
}
kernel_results = pd.DataFrame(
    [
        evaluate_model(
            Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)]),
            name,
        )
        for name, model in kernel_models.items()
    ]
).sort_values("f1_score", ascending=False)
display(kernel_results.style.format({column: "{:.4f}" for column in kernel_results.columns if column != "kernel"}))

# %% [markdown]
## 7. Hyperparameter Tuning

# %%
tuning_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", SVC()),
    ]
)
parameter_grid = [
    {"classifier__kernel": ["linear"], "classifier__C": [0.1, 1, 10]},
    {
        "classifier__kernel": ["rbf"],
        "classifier__C": [0.1, 1, 10, 100],
        "classifier__gamma": ["scale", "auto", 0.01],
    },
    {
        "classifier__kernel": ["poly"],
        "classifier__C": [0.1, 1, 10],
        "classifier__degree": [2, 3],
        "classifier__gamma": ["scale"],
    },
]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
grid_search = GridSearchCV(
    tuning_pipeline,
    parameter_grid=parameter_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    refit=True,
)
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)
print(f"Best mean CV F1-score: {grid_search.best_score_:.4f}")

# %% [markdown]
## 8. Tuned Model Evaluation

# %%
tuned_predictions = grid_search.predict(X_test)
print(classification_report(y_test, tuned_predictions, zero_division=0))
print("Test metrics:")
print(
    pd.Series(
        {
            "accuracy": accuracy_score(y_test, tuned_predictions),
            "precision": precision_score(y_test, tuned_predictions, pos_label="poisonous", zero_division=0),
            "recall": recall_score(y_test, tuned_predictions, pos_label="poisonous", zero_division=0),
            "f1_score": f1_score(y_test, tuned_predictions, pos_label="poisonous", zero_division=0),
        }
    ).round(4)
)
ConfusionMatrixDisplay.from_predictions(y_test, tuned_predictions, cmap="Greens")
plt.title("Tuned SVM confusion matrix")
plt.show()

# %% [markdown]
# 9. Permutation Importance
#
# Permutation importance measures how much the fitted pipeline's test score changes
# when each original input column is shuffled. It is a model-agnostic proxy rather
# than a causal explanation.

# %%
permutation = permutation_importance(
    grid_search.best_estimator_,
    X_test,
    y_test,
    scoring="f1",
    n_repeats=5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
importance = pd.Series(permutation.importances_mean, index=features.columns).sort_values(ascending=False)
display(importance.head(15).to_frame("mean_f1_importance"))
sns.barplot(x=importance.head(15).values, y=importance.head(15).index, color="teal")
plt.title("Top original features by permutation importance")
plt.xlabel("Mean decrease in F1 after permutation")
plt.show()

# %% [markdown]
# 10. Analysis and Conclusion
#
# - **EDA:** The target distribution should be checked before training; stratification preserves that distribution in both splits. Numeric feature plots reveal overlap, while categorical class-proportion plots show which mushroom characteristics separate edible and poisonous classes.
# - **SVM strengths:** SVMs work well with many one-hot encoded predictors, can model nonlinear boundaries with the RBF kernel, and maximize a margin that often generalizes effectively on medium-sized datasets.
# - **SVM weaknesses:** One-hot encoding can create a high-dimensional matrix, training and tuning can become expensive as the dataset grows, and the model is less interpretable than a shallow decision tree. Results also depend on preprocessing and hyperparameter choices.
# - **Practical implications:** For mushroom safety screening, recall for the poisonous class is especially important because a false negative can be dangerous. In production, the classifier should support expert review, monitor drift, validate incoming categories, and avoid treating predictions as a substitute for toxicological verification.
# - **Final choice:** Select the tuned model using cross-validated F1, then verify its poisonous-class recall and confusion matrix on the untouched test set. The reported test metrics provide the final comparison against the untuned kernel models.
