from sklearn.ensemble import RandomForestClassifier
import joblib
from features import extract_features
import random

# Dummy dataset (for demo only)
X = []
y = []
for i in range(200):
    # create fake URLs
    if i % 2 == 0:
        url = f"https://safe{i}.example.com"
        label = 0
    else:
        url = f"http://login-secure{i}.bad.com/login"
        label = 1
    X.append(extract_features(url))
    y.append(label)

clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X, y)
joblib.dump(clf, "model.joblib")
print("Saved model.joblib")
