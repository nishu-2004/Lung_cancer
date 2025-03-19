import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import pickle

# Create folder for saving plots
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

# Load dataset
df = pd.read_csv('dataset_med.csv')
df['end_treatment_date']=pd.to_datetime(df['end_treatment_date'])
df['date']=df['end_treatment_date'].dt.day
df['month']=df['end_treatment_date'].dt.month
df['year']=df['end_treatment_date'].dt.year
df=df.drop('end_treatment_date',axis=1)
df['diagnosis_date']=pd.to_datetime(df['diagnosis_date'])
df['date1']=df['diagnosis_date'].dt.day
df['month1']=df['diagnosis_date'].dt.month
df['year1']=df['diagnosis_date'].dt.year
df=df.drop('diagnosis_date',axis=1)

# Select only numerical columns
df = df.select_dtypes(include=[np.number])

# Separate features and target
X = df.drop(columns=['survived'], errors='ignore')
y = df['survived']
y = (y > 0).astype(int) 

# Save feature names
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)
    
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Compute Mutual Information
from sklearn.feature_selection import mutual_info_classif
mi_scores = mutual_info_classif(X, y, discrete_features='auto')

# Convert to DataFrame for easy visualization
mi_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})
mi_df = mi_df.sort_values(by='MI Score', ascending=False)

# Plot MI Scores
plt.figure(figsize=(10, 5))
sns.barplot(x='MI Score', y='Feature', data=mi_df, palette='viridis')
plt.title("Mutual Information Scores for Feature Selection")
plt.savefig(f"{plot_dir}/mutual_info.jpg", format="jpg", dpi=300)
plt.close()

# Print the scores
print(mi_df)

# Heatmap of correlations
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.savefig(f"{plot_dir}/correlation_heatmap.jpg", format="jpg", dpi=300)
plt.close()

# Check class balance before resampling
print("Class Distribution (Before Resampling):\n", y.value_counts())

# Apply SMOTE + Undersampling
smote = SMOTE(sampling_strategy=0.6, random_state=42)
under = RandomUnderSampler(sampling_strategy=0.7, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
X_resampled, y_resampled = under.fit_resample(X_resampled, y_resampled)

print("Class Distribution (After Resampling):\n", pd.Series(y_resampled).value_counts())

# Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
scaler_path = "scaler.pkl"
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Scaler saved successfully!")

# Define the ANN Model
def build_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),

        Dense(1, activation='sigmoid')  # Binary classification
    ])
    return model

model = build_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callbacks
checkpoint_path = "checkpoint.weights.h5"
checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, monitor='val_loss', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Load checkpoint if exists
if os.path.exists(checkpoint_path):
    print("🔄 Loading checkpoint to resume training...")
    model.load_weights(checkpoint_path)
    print("✅ Checkpoint restored successfully!")
else:
    print("⚠️ No checkpoint found. Starting training from scratch...")

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), verbose=1,
                    callbacks=[checkpoint, early_stopping, reduce_lr])

# Plot loss curves
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"{plot_dir}/loss_curve.jpg", format="jpg", dpi=300)
plt.close()

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save the model
model.save('model1.h5')
print("✅ Model saved successfully!")
