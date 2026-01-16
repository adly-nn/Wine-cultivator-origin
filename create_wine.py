import pandas as pd
import joblib
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Load Data
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target # Classes: 0, 1, 2

# 2. Select Top 5 Features (Easier for web demo)
# These are chemically the strongest indicators of origin
features = ['alcohol', 'flavanoids', 'color_intensity', 'proline', 'od280/od315_of_diluted_wines']
X = df[features]
y = df['target']

# 3. Scale Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Build Neural Network (Multi-Class)
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(5,)))
model.add(Dense(16, activation='relu'))
# Output layer has 3 units (one for each Cultivator)
# Softmax ensures they add up to 100% probability
model.add(Dense(3, activation='softmax')) 

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 5. Train
print("Training Wine Cultivator Model...")
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# 6. Save
model.save('wine_model.h5')
joblib.dump(scaler, 'wine_scaler.pkl')
print("Saved wine_model.h5 and wine_scaler.pkl")