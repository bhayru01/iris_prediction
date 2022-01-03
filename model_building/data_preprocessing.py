# %% 0. LIBRARIES
import pandas as pd
from sklearn.datasets import load_iris



# %% 1. DATA

# loading the dataset
iris = load_iris()
X = iris.data
y = iris.target



# %% Pre-process
iris_df = pd.DataFrame(X, columns=iris.feature_names)
iris_df['species'] = y



# %% Save
iris_df.to_csv("../model_building/data/iris_dataset.csv", index=False)
# %%
