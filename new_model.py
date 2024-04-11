# ######################### Random Forest ################################################################################
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report

# # Read the CSV file
# data = pd.read_csv('result_with_e_descriptors_all_data_with_aac.csv')

# # Extract features (AAC and E descriptors) and target variable
# X = data[['E1', 'E2', 'E3', 'E4', 'E5', 'AAC_A', 'AAC_C', 'AAC_D', 'AAC_E', 'AAC_F', 'AAC_G', 'AAC_H',
#           'AAC_I', 'AAC_K', 'AAC_L', 'AAC_M', 'AAC_N', 'AAC_P', 'AAC_Q', 'AAC_R', 'AAC_S', 'AAC_T',
#           'AAC_V', 'AAC_W', 'AAC_Y']]
# y = data['Binary_allergen']

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and fit the Random Forest classifier
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_classifier.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = rf_classifier.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print("\nClassification Report:")
# print(f"Overall Accuracy: {accuracy:.2%} Using Random Forest")

# print(classification_report(y_test, y_pred))




# ######################### TensorFlow ################################################################################
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.preprocessing import StandardScaler
# from tensorflow import keras
# from tensorflow.keras import layers

# # Read the CSV file
# data = pd.read_csv('result_with_e_descriptors_all_data_with_aac.csv')

# # Extract features (AAC and E descriptors) and target variable
# X = data[['E1', 'E2', 'E3', 'E4', 'E5', 'AAC_A', 'AAC_C', 'AAC_D', 'AAC_E', 'AAC_F', 'AAC_G', 'AAC_H',
#           'AAC_I', 'AAC_K', 'AAC_L', 'AAC_M', 'AAC_N', 'AAC_P', 'AAC_Q', 'AAC_R', 'AAC_S', 'AAC_T',
#           'AAC_V', 'AAC_W', 'AAC_Y']]
# y = data['Binary_allergen']

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardize the data
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Build a basic neural network model
# model = keras.Sequential([
#     layers.Input(shape=(X_train.shape[1],)),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(32, activation='relu'),
#     layers.Dense(1, activation='sigmoid')
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2)

# # Evaluate the model
# y_pred_proba = model.predict(X_test_scaled)
# y_pred = (y_pred_proba > 0.5).astype(int)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print("\nClassification Report:")
# print(f"Overall Accuracy: {accuracy:.2%} Using Deep Learning")

# print(classification_report(y_test, y_pred))



# ######################### Deeper neural network ################################################################################
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.preprocessing import StandardScaler
# from tensorflow import keras
# from tensorflow.keras import layers

# # Read the CSV file
# data = pd.read_csv('result_with_e_descriptors_all_data_with_aac.csv')

# # Extract features (AAC and E descriptors) and target variable
# X = data[['E1', 'E2', 'E3', 'E4', 'E5', 'AAC_A', 'AAC_C', 'AAC_D', 'AAC_E', 'AAC_F', 'AAC_G', 'AAC_H',
#           'AAC_I', 'AAC_K', 'AAC_L', 'AAC_M', 'AAC_N', 'AAC_P', 'AAC_Q', 'AAC_R', 'AAC_S', 'AAC_T',
#           'AAC_V', 'AAC_W', 'AAC_Y']]
# y = data['Binary_allergen']

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardize the data
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Build a deeper neural network model
# model = keras.Sequential([
#     layers.Input(shape=(X_train.shape[1],)),
#     layers.Dense(128, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(64, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(32, activation='relu'),
#     layers.Dense(1, activation='sigmoid')
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_split=0.2)

# # Evaluate the model
# y_pred_proba = model.predict(X_test_scaled)
# y_pred = (y_pred_proba > 0.5).astype(int)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print("\nClassification Report:")
# print(f"Overall Accuracy: {accuracy:.2%} Using Deep Learning")

# print(classification_report(y_test, y_pred))


# ######################### Min-Max scaling ################################################################################
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow import keras
# from tensorflow.keras import layers

# # Read the CSV file
# data = pd.read_csv('result_with_e_descriptors_all_data_with_aac.csv')

# # Extract features (AAC and E descriptors) and target variable
# X = data[['E1', 'E2', 'E3', 'E4', 'E5', 'AAC_A', 'AAC_C', 'AAC_D', 'AAC_E', 'AAC_F', 'AAC_G', 'AAC_H',
#           'AAC_I', 'AAC_K', 'AAC_L', 'AAC_M', 'AAC_N', 'AAC_P', 'AAC_Q', 'AAC_R', 'AAC_S', 'AAC_T',
#           'AAC_V', 'AAC_W', 'AAC_Y']]
# y = data['Binary_allergen']

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Min-Max scaling
# scaler = MinMaxScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Build a deeper neural network model
# model = keras.Sequential([
#     layers.Input(shape=(X_train.shape[1],)),
#     layers.Dense(256, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(128, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(64, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(32, activation='relu'),
#     layers.Dense(1, activation='sigmoid')
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(X_train_scaled, y_train, epochs=30, batch_size=32, validation_split=0.2)

# # Evaluate the model
# y_pred_proba = model.predict(X_test_scaled)
# y_pred = (y_pred_proba > 0.5).astype(int)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print("\nClassification Report:")
# print(f"Overall Accuracy: {accuracy:.2%} Using Deep Learning")

# print(classification_report(y_test, y_pred))






# # 93.60%
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, StratifiedKFold
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.preprocessing import StandardScaler
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.callbacks import EarlyStopping

# # Read the CSV file
# data = pd.read_csv('result_with_e_descriptors_all_data_with_aac.csv')

# # Extract features (AAC and E descriptors) and target variable
# X = data[['E1', 'E2', 'E3', 'E4', 'E5', 'AAC_A', 'AAC_C', 'AAC_D', 'AAC_E', 'AAC_F', 'AAC_G', 'AAC_H',
#           'AAC_I', 'AAC_K', 'AAC_L', 'AAC_M', 'AAC_N', 'AAC_P', 'AAC_Q', 'AAC_R', 'AAC_S', 'AAC_T',
#           'AAC_V', 'AAC_W', 'AAC_Y']]
# y = data['Binary_allergen']

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardize the data
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Build a deeper neural network model
# model = keras.Sequential([
#     layers.Input(shape=(X_train.shape[1],)),
#     layers.Dense(256, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(128, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(1, activation='sigmoid')
# ])

# # Compile the model
# model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
#               loss='binary_crossentropy', 
#               metrics=['accuracy'])

# # Early stopping to prevent overfitting
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# # Train the model with k-fold cross-validation
# kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# for train_index, val_index in kf.split(X_train_scaled, y_train):
#     X_train_cv, X_val_cv = X_train_scaled[train_index], X_train_scaled[val_index]
#     y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]

#     model.fit(X_train_cv, y_train_cv, epochs=30, batch_size=64, validation_data=(X_val_cv, y_val_cv), callbacks=[early_stopping], verbose=0)

# # Evaluate the model on the test set
# y_pred_proba = model.predict(X_test_scaled)
# y_pred = (y_pred_proba > 0.5).astype(int)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print("\nClassification Report:")
# print(f"Overall Accuracy: {accuracy:.2%} Using Deep Learning")
# print(classification_report(y_test, y_pred))









import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# Read the CSV file
data = pd.read_csv('result_with_e_descriptors_all_data_with_aac.csv')

# Extract features (AAC and E descriptors) and target variable
X = data[['E1', 'E2', 'E3', 'E4', 'E5', 'AAC_A', 'AAC_C', 'AAC_D', 'AAC_E', 'AAC_F', 'AAC_G', 'AAC_H',
          'AAC_I', 'AAC_K', 'AAC_L', 'AAC_M', 'AAC_N', 'AAC_P', 'AAC_Q', 'AAC_R', 'AAC_S', 'AAC_T',
          'AAC_V', 'AAC_W', 'AAC_Y']]
y = data['Binary_allergen']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a more complex neural network model
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with k-fold cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_index, val_index in kf.split(X_train_scaled, y_train):
    X_train_cv, X_val_cv = X_train_scaled[train_index], X_train_scaled[val_index]
    y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]

    model.fit(X_train_cv, y_train_cv, epochs=50, batch_size=64, validation_data=(X_val_cv, y_val_cv), callbacks=[early_stopping], verbose=0)

# Evaluate the model on the test set
y_pred_proba = model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("\nClassification Report:")
print(f"Overall Accuracy: {accuracy:.2%} Using Deep Learning")
print(classification_report(y_test, y_pred))
