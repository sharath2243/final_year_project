import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('../dataset/data.csv')

# Include cases where there is no deficiency
data = data[data['Vitamin_Deficiency'].isin(['Vitamin K', 'Vitamin D', 'Vitamin E', 'No Deficiency'])]

# Check class distribution
print("Class distribution:\n", data['Vitamin_Deficiency'].value_counts())

# Initialize label encoders dictionary
encoders = {}

# Encode categorical variables
for column in data.columns[:-1]:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    encoders[column] = le

# Encode the target variable
le_deficiency = LabelEncoder()
data['Vitamin_Deficiency'] = le_deficiency.fit_transform(data['Vitamin_Deficiency'])
encoders['Vitamin_Deficiency'] = le_deficiency

# Split the data into features and target variable
X = data.drop('Vitamin_Deficiency', axis=1)
y = data['Vitamin_Deficiency']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}')

# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le_deficiency.classes_))
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Feature importance
feature_importance = model.feature_importances_
feature_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
feature_df = feature_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_df['Feature'], feature_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.show()

# Save the model and the encoders
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/vitamin_deficiency_model.pkl')
joblib.dump(encoders, 'models/encoders.pkl')
print('Model and LabelEncoders saved to models directory')
