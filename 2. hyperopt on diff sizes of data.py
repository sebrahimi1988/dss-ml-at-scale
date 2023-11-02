# Databricks notebook source
# MAGIC %md ## Hyperopt: best practices for datasets of different sizes
# MAGIC
# MAGIC This notebook provides guidelines for using the Hyperopt class `SparkTrials` when working with datasets of different sizes:
# MAGIC * small (~10MB or less)
# MAGIC * medium (~100MB)
# MAGIC * large (~1GB or more)
# MAGIC
# MAGIC The notebook uses randomly generated datasets. The goal is to tune the regularization parameter `alpha` in a LASSO model.
# MAGIC
# MAGIC Requirements:
# MAGIC * Two workers

# COMMAND ----------

import numpy as np
import os, shutil, tempfile
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK
from sklearn import linear_model, datasets, model_selection

# COMMAND ----------

# Some utility methods

def gen_data(bytes):
  """
  Generates train/test data with target total bytes for a random regression problem.
  Returns (X_train, X_test, y_train, y_test).
  """
  n_features = 100
  n_samples = int((1.0 * bytes / (n_features + 1)) / 8)
  X, y = datasets.make_regression(n_samples=n_samples, n_features=n_features, random_state=0)
  return model_selection.train_test_split(X, y, test_size=0.2, random_state=1)

def train_and_eval(data, alpha):
  """
  Trains a LASSO model using training data with the input alpha and evaluates it using test data.
  """
  X_train, X_test, y_train, y_test = data  
  model = linear_model.Lasso(alpha=alpha)
  model.fit(X_train, y_train)
  loss = model.score(X_test, y_test)
  return {"loss": loss, "status": STATUS_OK}

def tune_alpha(objective):
  """
  Uses Hyperopt's SparkTrials to tune the input objective, which takes alpha as input and returns loss.
  Returns the best alpha found.
  """
  best = fmin(
    fn=objective,
    space=hp.uniform("alpha", 0.0, 10.0),
    algo=tpe.suggest,
    max_evals=4,
    trials=SparkTrials(parallelism=2))
  return best["alpha"]

# COMMAND ----------

# MAGIC %md ### Small datasets (~10MB or less)
# MAGIC
# MAGIC When a dataset is small, you can load it on the driver and call it from the objective function directly.  
# MAGIC `SparkTrials` automatically broadcasts the data and the objective function to workers.  
# MAGIC There is negligible overhead.

# COMMAND ----------

# Generate dataset, ~10MB
data_small = gen_data(10 * 1024 * 1024) 

# COMMAND ----------

def objective_small(alpha):
  # For small data, you can call it directly
  return train_and_eval(data_small, alpha)

tune_alpha(objective_small)

# COMMAND ----------

# MAGIC %md ### Medium datasets (~100MB)
# MAGIC
# MAGIC Calling a medium dataset directly from the objective function can be inefficient.  
# MAGIC If you change the objective function code, the data would have to be broadcast again.  
# MAGIC Databricks recommends broadcasting the data explicitly using Spark and getting back its value from the broadcasted variable on workers.

# COMMAND ----------

# Generate dataset, ~100MB
data_medium = gen_data(100 * 1024 * 1024)
# For medium data, you might broadcast it first
bc_data_medium = sc.broadcast(data_medium)

# COMMAND ----------

def objective_medium(alpha):
  # Load broadcasted value onto workers
  data = bc_data_medium.value
  return train_and_eval(data, alpha)

tune_alpha(objective_medium)

# COMMAND ----------

# MAGIC %md ### Large datasets (~1GB or more)
# MAGIC
# MAGIC Broadcasting a large dataset requires significant cluster resources.  
# MAGIC Consider storing the data on DBFS and loading it back onto workers using the DBFS local file interface.

# COMMAND ----------

# Some utility methods

def save_to_dbfs(data):
  """
  Saves input data (a tuple of numpy arrays) to a temporary file on DBFS and returns its path.
  """
  # Save data to a local file first
  data_filename = "data.npz"
  local_data_dir = tempfile.mkdtemp()
  local_data_path = os.path.join(local_data_dir, data_filename)
  np.savez(local_data_path, *data)
  
  # Move the data to DBFS, which is shared among cluster nodes
  dbfs_tmp_dir = "/dbfs/ml/tmp/hyperopt"
  os.makedirs(dbfs_tmp_dir, exist_ok=True)
  dbfs_data_dir = tempfile.mkdtemp(dir=dbfs_tmp_dir)  
  dbfs_data_path = os.path.join(dbfs_data_dir, data_filename)  
  shutil.move(local_data_path, dbfs_data_path)
  return dbfs_data_path

def load(path):
  """
  Loads saved data (a tuple of numpy arrays).
  """
  return list(np.load(path).values())

# COMMAND ----------

# Generate dataset, ~1000MB
data_large = gen_data(1000 * 1024 * 1024) 
# For large data, save it to DBFS first
data_large_path = save_to_dbfs(data_large)

# COMMAND ----------

def objective_large(alpha):
  # Load data back from DBFS onto workers
  data = load(data_large_path)
  return train_and_eval(data, alpha)

tune_alpha(objective_large)

# COMMAND ----------

# Delete the large dataset
shutil.rmtree(data_large_path, ignore_errors=True)