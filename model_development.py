import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE  
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Load dataset
df = pd.read_csv("data/expanded_data.csv")  # Using the expanded dataset after merging

# Handle NaN values in the 'processed_message' column
df['processed_message'] = df['processed_message'].fillna('')

# Lemmatization (instead of stemming)
lemmatizer = WordNetLemmatizer()
df['processed_message'] = df['processed_message'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

# Ensure the required columns exist
if 'processed_message' not in df.columns or 'label' not in df.columns:
    raise ValueError("Missing required columns: 'processed_message' or 'label'")

X = df['processed_message']  # Features
y = df['label']  # Target

# Vectorization with TF-IDF (using unigrams and bigrams)
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))  # Added bigrams
X_tfidf = tfidf.fit_transform(X)

# Apply SMOTE for balancing
smote = SMOTE(sampling_strategy='minority', random_state=42)  # Changed sampling strategy to minority
X_resampled, y_resampled = smote.fit_resample(X_tfidf, y)

# Convert to dense format
X_resampled = pd.DataFrame(X_resampled.toarray(), columns=tfidf.get_feature_names_out())
y_resampled = pd.Series(y_resampled)

print("Class distribution after SMOTE:\n", y_resampled.value_counts())

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

# Hyperparameter tuning with GridSearchCV for Logistic Regression
logreg_param_grid = {
    'C': [0.1, 1, 10],
    'class_weight': ['balanced'],
    'penalty': ['l2'],
    'solver': ['liblinear']
}

logreg_grid = GridSearchCV(LogisticRegression(max_iter=1000), logreg_param_grid, cv=5, scoring='accuracy')
logreg_grid.fit(X_train, y_train)

print("Best parameters for Logistic Regression:", logreg_grid.best_params_)

# Hyperparameter tuning with GridSearchCV for Random Forest
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced']
}

rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, scoring='accuracy')
rf_grid.fit(X_train, y_train)

print("Best parameters for Random Forest:", rf_grid.best_params_)

# Initialize Naive Bayes model
nb_param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0]}

nb_grid = GridSearchCV(MultinomialNB(), nb_param_grid, cv=5, scoring='accuracy')
nb_grid.fit(X_train, y_train)

print("Best parameters for Naive Bayes:", nb_grid.best_params_)

# Initialize models
logreg_best = logreg_grid.best_estimator_
rf_best = rf_grid.best_estimator_
bn_best = nb_grid.best_estimator_

# Train models
logreg_best.fit(X_train, y_train)
rf_best.fit(X_train, y_train)
bn_best.fit(X_train, y_train)

# Predictions
y_pred_logreg = logreg_best.predict(X_test)
y_pred_rf = rf_best.predict(X_test)
y_pred_bn = bn_best.predict(X_test)

# Evaluate models with classification report
print("\nLogistic Regression (Test evaluation):")
print(classification_report(y_test, y_pred_logreg, zero_division=1))
print("\nRandom Forest (Test evaluation):")
print(classification_report(y_test, y_pred_rf, zero_division=1))
print("\nNaive Bayes (Test evaluation):")
print(classification_report(y_test, y_pred_bn, zero_division=1))

# Ensemble using VotingClassifier
voting_clf = VotingClassifier(estimators=[
    ('logreg', logreg_best),
    ('rf', rf_best),
    ('nb', bn_best)
], voting='soft', weights=[2, 1, 1])  # Adjust weights for better models

voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_test)

print("\nVoting Classifier (Test evaluation):")
print(classification_report(y_test, y_pred_voting, zero_division=1))

# Confusion Matrix Analysis
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

# Apply visualization to models
models = {"Logistic Regression": logreg_best,
          "Random Forest": rf_best,
          "Naive Bayes": bn_best,
          "Voting Classifier": voting_clf}

for model_name, model in models.items():
    print(f"\nVisualizing Performance for {model_name}...\n")
    y_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, model_name)

# ROC & Precision-Recall Analysis
def plot_roc_pr_curves(model, X_test, y_test, model_name):
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_f1_score = max(f1_scores)

    plt.figure(figsize=(12, 5))

    # Plot ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")

    # Plot Precision-Recall Curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='red', lw=2, label=f'F1 Score = {best_f1_score:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower left")

    plt.show()

# Apply to all models
for model_name, model in models.items():
    print(f"\nPlotting ROC & Precision-Recall Curves for {model_name}...\n")
    plot_roc_pr_curves(model, X_test, y_test, model_name)
