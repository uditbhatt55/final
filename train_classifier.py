import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

EXPECTED_LENGTH = 42

data_dict = pickle.load(open('./data.pickle', 'rb'))

raw_data = data_dict['data']
raw_labels = data_dict['labels']

clean_data = []
clean_labels = []

for sample, label in zip(raw_data, raw_labels):
    if sample is None:
        continue
    if len(sample) != EXPECTED_LENGTH:
        continue
    if any(v is None or np.isnan(v) for v in sample):
        continue
    clean_data.append(sample)
    clean_labels.append(label)

print(f"Total samples after filtering: {len(clean_data)} / {len(raw_data)}")

if len(clean_data) == 0:
    raise RuntimeError("No valid samples found. Check your data.pickle.")

X = np.array(clean_data, dtype=np.float32)
y = np.array(clean_labels)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {acc * 100:.2f}%\n")
print(classification_report(y_test, y_pred))

with open('model.p', 'wb') as f:
    pickle.dump({
        'model': model,
        'scaler': scaler,
        'classes': sorted(list(set(y))),
        'feature_size': EXPECTED_LENGTH   
    }, f)

print("Model saved to model.p")