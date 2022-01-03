# %% 0. LIBRARIES
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

# %% 1. LOADING DATASET
iris_df = pd.read_csv("../model_building/data/iris_dataset.csv")
X = iris_df.iloc[::, :-1]
y = iris_df.iloc[::, -1]

# %% 2. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.5)

# %% 3. BUILD MODEL
clf = RandomForestClassifier(n_estimators=10)

# %% 4. TRAIN MODEL
clf.fit(X_train, y_train)

# %% 5. GET PREDICTIONS
predicted = clf.predict(X_test)

# %% 6.CHECK ACCURACY
print(accuracy_score(predicted, y_test))

# %% 7.SAVE MODEL
with open('../models/rf.pkl', 'wb') as model_pkl:
    pickle.dump(clf, model_pkl, protocol=2)
# %%
