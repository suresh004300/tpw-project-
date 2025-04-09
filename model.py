# Step 1: Import Libraries
print("Step 1: Importing Libraries")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import joblib

# Step 2: Load Data
print("\nStep 2: Loading Data")
file_path = r'C:\Users\B.Suneel\OneDrive\Desktop\mtech sem2\own project\stress_dataset.csv'
data = pd.read_csv(file_path)
print(f"Dataset Loaded Successfully from: {file_path}")

# Step 3: Explore Data
print("\nStep 3: Exploring Data")
print(data.info())
print("\nMissing Values:\n", data.isnull().sum())
print("\nClass Distribution Before SMOTE:\n", data['stress'].value_counts())

plt.figure(figsize=(8, 5))
sns.countplot(x='stress', hue='stress', data=data, palette='Set2', legend=False)
plt.title("Class Distribution Before SMOTE")
plt.show()

# Step 4: Clean Data
print("\nStep 4: Cleaning Data")
data = data.dropna()

# Step 5: Prepare Features and Target
print("\nStep 5: Preparing Features and Target")
X = data[['temperature', 'steps', 'humidity', 'heart_rate', 'sleep_count']]
y = data['stress']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.pkl')  # Save Scaler

# Step 6: Apply SMOTE
print("\nStep 6: Applying SMOTE")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print("\nClass Distribution After SMOTE:\n", pd.Series(y_train_resampled).value_counts())

plt.figure(figsize=(8, 5))
sns.countplot(x=pd.Series(y_train_resampled), palette='Set2')
plt.title("Class Distribution After SMOTE")
plt.xlabel("Stress Class")
plt.ylabel("Count")
plt.show()

# Store results
model_names = []
accuracies = []
conf_matrices = []

# Step 7: Train Random Forest Model
print("\nStep 7: Training Random Forest Model")
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

print("\nEvaluating Random Forest Model")
y_pred_rf = rf_model.predict(X_test_scaled)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {acc_rf * 100:.2f}%")
print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf, target_names=['No Stress', 'Avg Stress', 'Stressed'], zero_division=0))

model_names.append("Random Forest")
accuracies.append(acc_rf)
conf_matrices.append(confusion_matrix(y_test, y_pred_rf))

# Save Random Forest model
print("\nSaving Random Forest Model")
joblib.dump(rf_model, 'random_forest_model.pkl')
print("Random Forest model saved as 'random_forest_model.pkl'")

# Step 8: Train Naive Bayes Model
print("\nStep 8: Training Naive Bayes Model")
nb_model = GaussianNB()
nb_model.fit(X_train_resampled, y_train_resampled)

print("\nEvaluating Naive Bayes Model")
y_pred_nb = nb_model.predict(X_test_scaled)
acc_nb = accuracy_score(y_test, y_pred_nb)
print(f"Naive Bayes Accuracy: {acc_nb * 100:.2f}%")
print("\nNaive Bayes Classification Report:\n", classification_report(y_test, y_pred_nb, target_names=['No Stress', 'Avg Stress', 'Stressed'], zero_division=0))

model_names.append("Naive Bayes")
accuracies.append(acc_nb)
conf_matrices.append(confusion_matrix(y_test, y_pred_nb))

# Step 9: Train SVM Model
print("\nStep 9: Training SVM Model")
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train_resampled, y_train_resampled)

print("\nEvaluating SVM Model")
y_pred_svm = svm_model.predict(X_test_scaled)
acc_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {acc_svm * 100:.2f}%")
print("\nSVM Classification Report:\n", classification_report(y_test, y_pred_svm, target_names=['No Stress', 'Avg Stress', 'Stressed'], zero_division=0))

model_names.append("SVM")
accuracies.append(acc_svm)
conf_matrices.append(confusion_matrix(y_test, y_pred_svm))

# Step 10: Visual Comparison
print("\nStep 10: Visual Comparison of Models")

# Bar Plot of Accuracies
plt.figure(figsize=(8, 5))
sns.barplot(x=model_names, y=[acc * 100 for acc in accuracies], palette='Set2')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 100)
for i, acc in enumerate(accuracies):
    plt.text(i, acc * 100 + 1, f"{acc * 100:.2f}%", ha='center')
plt.show()

# Plot Confusion Matrices
for i, cm in enumerate(conf_matrices):
    plt.figure(figsize=(6, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Stress', 'Avg Stress', 'Stressed'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"{model_names[i]} - Confusion Matrix")
    plt.show()

# Step 11: Prediction Function for Random Forest
print("\nStep 11: Making Predictions (Random Forest Only)")

def predict_stress(model, scaler, temperature, steps, humidity, heart_rate, sleep_count):
    new_data = pd.DataFrame([[temperature, steps, humidity, heart_rate, sleep_count]], 
                            columns=['temperature', 'steps', 'humidity', 'heart_rate', 'sleep_count'])
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)[0]
    return {0: "No Stress", 1: "Avg Stress", 2: "Stressed"}[prediction]

# Test Prediction
sample_input = (37.5, 5000, 70, 85, 5.0)
prediction = predict_stress(rf_model, scaler, *sample_input)
print(f"Random Forest Prediction for {sample_input}: {prediction}")
