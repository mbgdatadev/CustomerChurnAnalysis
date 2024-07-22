import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# RFM verilerini yükle
rfm_data = pd.read_csv("rfm.csv", index_col=0)

# Segments'ı hedef değişken olarak ayır
X = rfm_data[['recency', 'frequency', 'monetary']]
y = rfm_data['segment']

# Kategorik hedef değişkeni sayısal değerlere dönüştür
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model ve hiperparametreler
model = RandomForestClassifier(random_state=42)

# Hiperparametreler için grid tanımla
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# GridSearchCV ile hiperparametre optimizasyonu
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           cv=5, scoring='accuracy', verbose=2, n_jobs=-1)

# Modeli eğit
grid_search.fit(X_train, y_train)

# En iyi parametreleri yazdır
print("Best Parameters:")
print(grid_search.best_params_)

# En iyi modeli al
best_model = grid_search.best_estimator_

# Model ve label encoder'ı bir arada kaydet
with open('best_model.pkl', 'wb') as file:
    pickle.dump((best_model, label_encoder), file)

# Test seti üzerinde tahmin yap
y_pred = best_model.predict(X_test)

# Sonuçları değerlendirme
test_accuracy = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred, average='weighted')
test_roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test), multi_class='ovr')

print("\nTest Accuracy:", test_accuracy)
print("Test F1 Score:", test_f1)
print("Test ROC AUC Score:", test_roc_auc)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Test metriklerini hesaplayın
test_accuracy = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred, average='weighted')
test_roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test), multi_class='ovr')

# Metrikleri bar grafikte gösterin
metrics = {
    'Accuracy': test_accuracy,
    'F1 Score': test_f1,
    'ROC AUC Score': test_roc_auc
}

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), ax=ax)
ax.set_ylim(0, 1)
ax.set_title('Test Metrics')
ax.set_ylabel('Score')
plt.show()

# Confusion matrix'i gösterin
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('Confusion Matrix')
plt.show()

# Classification report'u heatmap olarak gösterin
classification_rep = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
classification_rep_df = pd.DataFrame(classification_rep).transpose()

plt.figure(figsize=(10, 6))
sns.heatmap(classification_rep_df.iloc[:-1, :-1], annot=True, cmap='Blues')
plt.title('Classification Report')
plt.show()

# Tahminlerle gerçek değerleri gösteren bir grafik ekleyin
results_df = pd.DataFrame({'True': y_test, 'Predicted': y_pred})
plt.figure(figsize=(10, 6))
sns.scatterplot(data=results_df, x='True', y='Predicted', alpha=0.5)
plt.title('True vs Predicted Values')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.show()