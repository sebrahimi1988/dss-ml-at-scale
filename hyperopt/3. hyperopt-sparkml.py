# Databricks notebook source
# MAGIC %md 
# MAGIC # Tuning distributed training algorithms: Hyperopt and Apache Spark MLlib
# MAGIC
# MAGIC Databricks Runtime for Machine Learning includes [Hyperopt](https://github.com/hyperopt/hyperopt), a library for ML hyperparameter tuning in Python, and [Apache Spark MLlib](https://spark.apache.org/docs/latest/ml-guide.html), a library of distributed algorithms for training ML models (also often called "Spark ML").  This example notebook shows how to use them together.
# MAGIC
# MAGIC ## Use case
# MAGIC
# MAGIC Distributed machine learning workloads in Python for which you want to tune hyperparameters.
# MAGIC
# MAGIC ## In this example notebook
# MAGIC
# MAGIC The demo shows how to tune hyperparameters for an example machine learning workflow in MLlib.  You can follow this example to tune other distributed machine learning algorithms from MLlib or from other libraries.
# MAGIC
# MAGIC This guide includes two sections to illustrate the process you can follow to develop your own workflow:
# MAGIC * Run distributed training using MLlib.  In this section, you get the MLlib model training code working without hyperparameter tuning.
# MAGIC * Use Hyperopt to tune hyperparameters in the distributed training workflow.  In this section, you wrap the MLlib code with Hyperopt for tuning.
# MAGIC
# MAGIC ## Requirements
# MAGIC * To run the notebook, create a cluster with
# MAGIC  - At least one worker (or a cluster in "Single Node" mode)
# MAGIC  - Databricks Runtime 7.3 LTS ML or above

# COMMAND ----------

# MAGIC %md ## MLflow autologging
# MAGIC
# MAGIC This notebook demonstrates how to track model training and tuning with MLflow. Starting with MLflow version 1.17.0, you can use MLflow autologging with `pyspark.ml`. If your cluster is running Databricks Runtime for ML 8.2 or below, you can upgrade the MLflow client to add this `pyspark.ml` support. Upgrading is not required to run the notebook.    
# MAGIC
# MAGIC To upgrade MLflow to a version that supports `pyspark.ml` autologging, uncomment and run the following cell.

# COMMAND ----------

# %pip install --upgrade mlflow==1.18.0

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1. Run distributed training using MLlib
# MAGIC
# MAGIC This section shows a simple example of distributed training using MLlib. For more information and examples, see these resources:
# MAGIC * Databricks documentation on MLlib ([AWS](https://docs.databricks.com/applications/machine-learning/train-model/mllib/index.html)|[Azure](https://docs.microsoft.com/en-us/azure/databricks/applications/machine-learning/train-model/mllib/)|[GCP](https://docs.gcp.databricks.com/applications/machine-learning/train-model/mllib/index.html))
# MAGIC * [Apache Spark MLlib programming guide](https://spark.apache.org/docs/latest/ml-guide.html)
# MAGIC * [Apache Spark MLlib Python API documentation](https://spark.apache.org/docs/latest/api/python/reference/pyspark.ml.html)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load data
# MAGIC
# MAGIC This notebook uses the classic MNIST handwritten digit recognition dataset.  The examples are vectors of pixels representing images of handwritten digits.  For example:
# MAGIC
# MAGIC ![Image of a digit](http://training.databricks.com/databricks_guide/digit.png)
# MAGIC ![Image of all 10 digits](http://training.databricks.com/databricks_guide/MNIST-small.png)
# MAGIC
# MAGIC These datasets are stored in the popular LibSVM dataset format.  The following cell shows how to load them using MLlib's LibSVM dataset reader utility.

# COMMAND ----------

full_training_data = spark.read.format("libsvm").load("/databricks-datasets/mnist-digits/data-001/mnist-digits-train.txt")
test_data = spark.read.format("libsvm").load("/databricks-datasets/mnist-digits/data-001/mnist-digits-test.txt")

# Cache data for multiple uses.
full_training_data.cache()
test_data.cache()

print(f"There are {full_training_data.count()} training images and {test_data.count()} test images.")

# COMMAND ----------

# Randomly split the training data for use in tuning.
training_data, validation_data = full_training_data.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

# MAGIC %md Display the data.  Each image has the true label (the `label` column) and a vector of `features` that represent pixel intensities.

# COMMAND ----------

display(training_data)

# COMMAND ----------

# MAGIC %md ### Create a function to train a model
# MAGIC
# MAGIC In this section, you define a function to train a decision tree.  Wrapping the training code in a function is important for passing the function to Hyperopt for tuning later.
# MAGIC
# MAGIC Details: The tree algorithm needs to know that the labels are categories 0-9, rather than continuous values.  This example uses the `StringIndexer` class to do this.  A `Pipeline` ties this feature preprocessing together with the tree algorithm.  ML Pipelines are tools Spark provides for piecing together Machine Learning algorithms into workflows.  To learn more about Pipelines, check out other ML example notebooks in Databricks and the [ML Pipelines user guide](http://spark.apache.org/docs/latest/ml-guide.html).

# COMMAND ----------

import mlflow

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer

# COMMAND ----------

# As explained in Cmd 2 of this notebook, MLflow autologging for `pyspark.ml` requires MLflow version 1.17.0 or above.
# This try-except logic allows the notebook to run with older versions of MLflow.
try:
  import mlflow.pyspark.ml
  mlflow.pyspark.ml.autolog()
except:
  print(f"Your version of MLflow ({mlflow.__version__}) does not support pyspark.ml for autologging. To use autologging, upgrade your MLflow client version or use Databricks Runtime for ML 8.3 or above.")

# COMMAND ----------

def train_tree(minInstancesPerNode, maxBins):
  '''
  This train() function:
   - takes hyperparameters as inputs (for tuning later)
   - returns the F1 score on the validation dataset

  Wrapping code as a function makes it easier to reuse the code later with Hyperopt.
  '''
  # Use MLflow to track training.
  # Specify "nested=True" since this single model will be logged as a child run of Hyperopt's run.
  with mlflow.start_run(nested=True):
    
    # StringIndexer: Read input column "label" (digits) and annotate them as categorical values.
    indexer = StringIndexer(inputCol="label", outputCol="indexedLabel")
    
    # DecisionTreeClassifier: Learn to predict column "indexedLabel" using the "features" column.
    dtc = DecisionTreeClassifier(labelCol="indexedLabel",
                                 minInstancesPerNode=minInstancesPerNode,
                                 maxBins=maxBins)
    
    # Chain indexer and dtc together into a single ML Pipeline.
    pipeline = Pipeline(stages=[indexer, dtc])
    model = pipeline.fit(training_data)

    # Define an evaluation metric and evaluate the model on the validation dataset.
    evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", metricName="f1")
    predictions = model.transform(validation_data)
    validation_metric = evaluator.evaluate(predictions)
    mlflow.log_metric("val_f1_score", validation_metric)

  return model, validation_metric

# COMMAND ----------

# MAGIC %md Run the training function to make sure it works.
# MAGIC It's a good idea to make sure training code runs before adding in tuning.

# COMMAND ----------

initial_model, val_metric = train_tree(minInstancesPerNode=200, maxBins=2)
print(f"The trained decision tree achieved an F1 score of {val_metric} on the validation data")

# COMMAND ----------

# MAGIC %md ## Part 2. Use Hyperopt to tune hyperparameters
# MAGIC
# MAGIC In this section, you create the Hyperopt workflow. 
# MAGIC * Define a function to minimize
# MAGIC * Define a search space over hyperparameters
# MAGIC * Specify the search algorithm and use `fmin()` to tune the model
# MAGIC
# MAGIC For more information about the Hyperopt APIs, see the [Hyperopt documentation](http://hyperopt.github.io/hyperopt/).

# COMMAND ----------

# MAGIC %md ### Define a function to minimize
# MAGIC
# MAGIC * Input: hyperparameters
# MAGIC * Internally: Reuse the training function defined above.
# MAGIC * Output: loss

# COMMAND ----------

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

def train_with_hyperopt(params):
  """
  An example train method that calls into MLlib.
  This method is passed to hyperopt.fmin().
  
  :param params: hyperparameters as a dict. Its structure is consistent with how search space is defined. See below.
  :return: dict with fields 'loss' (scalar loss) and 'status' (success/failure status of run)
  """
  # For integer parameters, make sure to convert them to int type if Hyperopt is searching over a continuous range of values.
  minInstancesPerNode = int(params['minInstancesPerNode'])
  maxBins = int(params['maxBins'])

  model, f1_score = train_tree(minInstancesPerNode, maxBins)
  
  # Hyperopt expects you to return a loss (for which lower is better), so take the negative of the f1_score (for which higher is better).
  loss = - f1_score
  return {'loss': loss, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md ### Define the search space over hyperparameters
# MAGIC
# MAGIC This example tunes two hyperparameters: `minInstancesPerNode` and `maxBins`. See the [Hyperopt documentation](https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions) for details on defining a search space and parameter expressions.

# COMMAND ----------

import numpy as np
space = {
  'minInstancesPerNode': hp.uniform('minInstancesPerNode', 10, 200),
  'maxBins': hp.uniform('maxBins', 2, 32),
}

# COMMAND ----------

# MAGIC %md ### Tune the model using Hyperopt `fmin()`
# MAGIC
# MAGIC - Set `max_evals` to the maximum number of points in hyperparameter space to test (the maximum number of models to fit and evaluate). Because this command evaluates many models, it can take several minutes to execute.
# MAGIC - You must also specify which search algorithm to use. The two main choices are:
# MAGIC   - `hyperopt.tpe.suggest`: Tree of Parzen Estimators, a Bayesian approach which iteratively and adaptively selects new hyperparameter settings to explore based on previous results
# MAGIC   - `hyperopt.rand.suggest`: Random search, a non-adaptive approach that randomly samples the search space

# COMMAND ----------

# MAGIC %md 
# MAGIC **Important:**  
# MAGIC When using Hyperopt with MLlib and other distributed training algorithms, do not pass a `trials` argument to `fmin()`. When you do not include the `trials` argument, Hyperopt uses the default `Trials` class, which runs on the cluster driver. Hyperopt needs to evaluate each trial on the driver node so that each trial can initiate distributed training jobs.  
# MAGIC
# MAGIC Do not use the `SparkTrials` class with MLlib. `SparkTrials` is designed to distribute trials for algorithms that are not themselves distributed. MLlib uses distributed computing already and is not compatible with `SparkTrials`.

# COMMAND ----------

algo=tpe.suggest

with mlflow.start_run():
  best_params = fmin(
    fn=train_with_hyperopt,
    space=space,
    algo=algo,
    max_evals=8
  )

# COMMAND ----------

# Print out the parameters that produced the best model
best_params

# COMMAND ----------

# MAGIC %md ### Retrain the model on the full training dataset
# MAGIC
# MAGIC For tuning, this workflow split the training dataset into training and validation subsets. Now, retrain the model using the "best" hyperparameters on the full training dataset.

# COMMAND ----------

best_minInstancesPerNode = int(best_params['minInstancesPerNode'])
best_maxBins = int(best_params['maxBins'])

final_model, val_f1_score = train_tree(best_minInstancesPerNode, best_maxBins)

# COMMAND ----------

# MAGIC %md Use the test dataset to compare evaluation metrics for the initial and "best" models.

# COMMAND ----------

evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", metricName="f1")

initial_model_test_metric = evaluator.evaluate(initial_model.transform(test_data))
final_model_test_metric = evaluator.evaluate(final_model.transform(test_data))

print(f"On the test data, the initial (untuned) model achieved F1 score {initial_model_test_metric}, and the final (tuned) model achieved {final_model_test_metric}.")

# COMMAND ----------


