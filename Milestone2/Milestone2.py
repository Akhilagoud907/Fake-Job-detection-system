# ==============================================
# Milestone 2 - Model Training, Evaluation & Plots
# ==============================================

import joblib
from scipy import sparse
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# Sklearn imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc

# TensorFlow for BiLSTM
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==============================================
# Load preprocessed data from Milestone 1
# ==============================================
X_train = sparse.load_npz(r'C:\Temp\X_train_tfidf.npz')
X_test = sparse.load_npz(r'C:\Temp\X_test_tfidf.npz')

X_train_raw = joblib.load(r'C:\Temp\X_train_raw.pkl')
X_test_raw = joblib.load(r'C:\Temp\X_test_raw.pkl')

y_train = joblib.load(r'C:\Temp\y_train.pkl')
y_test = joblib.load(r'C:\Temp\y_test.pkl')

print("Data loaded successfully.")

# ==============================================
# Logistic Regression
# ==============================================
print("\nTraining Logistic Regression...")
lr_params = {'C':[0.01,0.1,1,10], 'penalty':['l2'], 'solver':['liblinear','lbfgs']}
lr = LogisticRegression(max_iter=1000)
grid_lr = GridSearchCV(lr, lr_params, cv=5, scoring='f1', n_jobs=-1)

start = time.time()
grid_lr.fit(X_train, y_train)
end = time.time()

lr_best = grid_lr.best_estimator_
y_pred_lr = lr_best.predict(X_test)
y_proba_lr = lr_best.predict_proba(X_test)[:,1]

print("Logistic Regression Metrics")
print("Best Params:", grid_lr.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr))
print("Recall:", recall_score(y_test, y_pred_lr))
print("F1 Score:", f1_score(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("Training Time:", round(end-start,2), "seconds")

joblib.dump(lr_best, r'C:\Temp\logistic_regression_v1.pkl')
print("Logistic Regression model saved.")

# ==============================================
# Random Forest
# ==============================================
print("\nTraining Random Forest...")
rf_params = {'n_estimators':[100,200], 'max_depth':[None,10,20], 'min_samples_split':[2,5]}
rf = RandomForestClassifier(random_state=42)
grid_rf = GridSearchCV(rf, rf_params, cv=5, scoring='f1', n_jobs=-1)

start = time.time()
grid_rf.fit(X_train, y_train)
end = time.time()

rf_best = grid_rf.best_estimator_
y_pred_rf = rf_best.predict(X_test)
y_proba_rf = rf_best.predict_proba(X_test)[:,1]

print("Random Forest Metrics")
print("Best Params:", grid_rf.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1 Score:", f1_score(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Training Time:", round(end-start,2), "seconds")

# Feature importance (top 10)
if hasattr(rf_best, 'feature_importances_'):
    importances = pd.Series(rf_best.feature_importances_)
    top_features = importances.sort_values(ascending=False).head(10)
    print("Top 10 feature importances:\n", top_features)
    
    plt.figure(figsize=(8,5))
    top_features.plot(kind='barh')
    plt.gca().invert_yaxis()
    plt.xlabel('Importance')
    plt.title('Top 10 Random Forest Feature Importances')
    plt.tight_layout()
    plt.savefig(r'C:\Temp\rf_feature_importance.png')
    plt.show()

joblib.dump(rf_best, r'C:\Temp\random_forest_v1.pkl')
print("Random Forest model saved.")

# Cross-validation
cv_scores = cross_val_score(rf_best, X_train, y_train, cv=5, scoring='f1')
print("Random Forest CV F1 Scores:", cv_scores)
print("Mean F1:", cv_scores.mean(), "Std:", cv_scores.std())

# ==============================================
# BiLSTM Deep Learning
# ==============================================
print("\nTraining BiLSTM model...")

max_words = 10000
max_len = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train_raw)

X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train_raw), maxlen=max_len)
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test_raw), maxlen=max_len)

model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))
model.add(Bidirectional(LSTM(64, return_sequences=False)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    X_train_seq, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=64,
    callbacks=[early_stop]
)

y_pred_dl = (model.predict(X_test_seq) > 0.5).astype(int)
y_proba_dl = model.predict(X_test_seq).ravel()

print("BiLSTM Metrics")
print("Accuracy:", accuracy_score(y_test, y_pred_dl))
print("Precision:", precision_score(y_test, y_pred_dl))
print("Recall:", recall_score(y_test, y_pred_dl))
print("F1 Score:", f1_score(y_test, y_pred_dl))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_dl))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dl))

model.save(r'C:\Temp\bilstm_model_v1.h5')
print("BiLSTM model saved.")

# ==============================================
# Plot ROC Curves
# ==============================================
plt.figure(figsize=(8,6))
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
fpr_dl, tpr_dl, _ = roc_curve(y_test, y_proba_dl)

plt.plot(fpr_lr, tpr_lr, label='Logistic Regression (AUC = %.2f)' % auc(fpr_lr, tpr_lr))
plt.plot(fpr_rf, tpr_rf, label='Random Forest (AUC = %.2f)' % auc(fpr_rf, tpr_rf))
plt.plot(fpr_dl, tpr_dl, label='BiLSTM (AUC = %.2f)' % auc(fpr_dl, tpr_dl))
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.tight_layout()
plt.savefig(r'C:\Temp\roc_curves.png')
plt.show()
print("ROC curves saved to C:\\Temp\\roc_curves.png")

# ==============================================
# Plot BiLSTM training history
# ==============================================
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('BiLSTM Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('BiLSTM Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(r'C:\Temp\bilstm_history.png')
plt.show()
print("BiLSTM training history saved to C:\\Temp\\bilstm_history.png")

# ==============================================
# Model Comparison Table
# ==============================================
results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "BiLSTM"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_dl)
    ],
    "F1 Score": [
        f1_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_dl)
    ],
    "ROC-AUC": [
        roc_auc_score(y_test, y_proba_lr),
        roc_auc_score(y_test, y_proba_rf),
        roc_auc_score(y_test, y_proba_dl)
    ]
})
print("\nModel Comparison:\n", results)
 


