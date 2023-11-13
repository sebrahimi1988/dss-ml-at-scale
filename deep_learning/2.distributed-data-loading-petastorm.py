# Databricks notebook source
# MAGIC %md
# MAGIC # Distributed data loading with Petastorm for distributed training
# MAGIC
# MAGIC [Petastorm](https://github.com/uber/petastorm) is an open source data access library. This library enables single-node or distributed training and evaluation of deep learning models directly from datasets in Apache Parquet format and datasets that are already loaded as Apache Spark DataFrames.
# MAGIC
# MAGIC This example shows how to use Petastorm  with TorchDistributor to train on `imagenet` data with Pytorch Lightning. 
# MAGIC
# MAGIC ![alt text](https://github.com/sebrahimi1988/dss-ml-at-scale/blob/main/deep_learning/Pytorch-data-loading.png?raw=true)
# MAGIC
# MAGIC ## Requirements
# MAGIC - Databricks Runtime ML 13.0 and above
# MAGIC - (Recommended) GPU instances

# COMMAND ----------

# MAGIC %pip install pytorch-lightning pillow deltalake

# COMMAND ----------

import io
import os
import time
import logging
from math import ceil
import mlflow

import warnings

warnings.filterwarnings("ignore")

from PIL import Image
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms, models
import torchmetrics.functional as FM
from pyspark.sql.functions import col
from petastorm import TransformSpec
from petastorm.reader import make_batch_reader
from petastorm.pytorch import DataLoader
from pyspark.ml.torch.distributor import TorchDistributor
from deltalake import DeltaTable

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup MLflow experiment
# MAGIC
# MAGIC In the following example, you specify the host and token for the notebook so you can reference them later in this guide. It also manually creates the experiment so that you can get the ID and send it to the worker nodes for scaling. 

# COMMAND ----------

username = spark.sql("SELECT current_user()").first()["current_user()"]

experiment_path = f"/Users/{username}/pytorch-distributor"

db_host = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .extraContext()
    .apply("api_url")
)
db_token = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
)

experiment = mlflow.set_experiment(experiment_path)

log_path = (
    f"/dbfs/Users/{username}/pl_training_logger"  # change this to location on DBFS
)

# COMMAND ----------

# MAGIC %md ## Load the dataset from Delta table
# MAGIC
# MAGIC The `imagenet` dataset is downloaded from the Kaggle challenge: [ImageNet Object Localization Challenge](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/overview). The data is processed and stored as a Delta table with the following schema.
# MAGIC ```
# MAGIC path:string
# MAGIC modificationTime:timestamp
# MAGIC length:long
# MAGIC content:binary
# MAGIC annotation:string
# MAGIC object_id:string
# MAGIC ```

# COMMAND ----------

train_delta_path = "/dbfs/tmp/dl/cv_datasets/imagenet_train.delta"
val_delta_path = "/dbfs/tmp/dl/cv_datasets/imagenet_val.delta"

# COMMAND ----------

# DBTITLE 1,Get list of Parquet files
training_dt = DeltaTable(train_delta_path)
train_parquet_files = training_dt.file_uris()
train_parquet_files = [
    parquet_file.replace("/dbfs", "file:///dbfs")
    for parquet_file in train_parquet_files
]
train_rows = training_dt.get_add_actions().to_pandas()["num_records"].sum()

val_dt = DeltaTable(val_delta_path)
val_parquet_files = val_dt.file_uris()
val_parquet_files = [
    parquet_file.replace("/dbfs", "file:///dbfs") for parquet_file in val_parquet_files
]
val_rows = val_dt.get_add_actions().to_pandas()["num_records"].sum()

# COMMAND ----------

unique_object_ids = (
    spark.read.format("delta")
    .load("/tmp/example-directory/cv_datasets/imagenet_train.delta")
    .select("object_id")
    .distinct()
    .collect()
)
object_id_to_class_mapping = {
    unique_object_ids[idx].object_id: idx for idx in range(len(unique_object_ids))
}

# COMMAND ----------

# MAGIC %md ## Set up the model
# MAGIC
# MAGIC The following  uses `resnet50` from `torchvision` and encapsulates it into [pl.LightningModule](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html). 

# COMMAND ----------

class ImageNetClassificationModel(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float,
        momentum: float = 0.9,
        logging_level=logging.INFO,
        device_id: int = 0,
        device_count: int = 1,
        feature_column: str = "content",
        label_column: str = "object_id",
    ):

        super().__init__()
        self.learn_rate = learning_rate
        self.momentum = momentum
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.state = {"epochs": 0}
        self.logging_level = logging_level
        self.device_id = device_id
        self.device_count = device_count
        self.feature_column = feature_column
        self.label_column = label_column

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learn_rate)

        return optimizer

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs

    def training_step(self, batch, batch_idx):
        X, y = (
            batch[self.feature_column],
            batch[self.label_column],
        )
        pred = self(X)
        loss = F.cross_entropy(pred, y)
        self.log("train_loss", loss)

        if self.logging_level == logging.DEBUG:
            if batch_idx == 0:
                print(f" - [{self.device_id}] training batch size: {y.shape[0]}")
            print(f" - [{self.device_id}] training batch: {batch_idx}, loss: {loss}")

        return loss

    def on_train_epoch_start(self):
        print(f"Epoch {self.state['epochs']} started at {time.time()} seconds")
        # No need to re-load data here as `train_dataloader` will be called on each epoch
        if self.logging_level in (logging.DEBUG, logging.INFO):
            print(f"++ [{self.device_id}] Epoch: {self.state['epochs']}")
        self.state["epochs"] += 1

    def validation_step(self, batch, batch_idx):
        X, y = (
            batch[self.feature_column],
            batch[self.label_column],
        )
        pred = self(X)
        loss = F.cross_entropy(pred, y)
        acc = FM.accuracy(pred, y, task="multiclass", num_classes=1000)

        # Roll validation up to epoch level
        self.log("val_loss", loss)
        self.log("val_acc", acc)

        if self.logging_level == logging.DEBUG:
            print(
                f" - [{self.device_id}] val batch: {batch_idx}, size: {y.shape[0]}, loss: {loss}, acc: {acc}"
            )

        return {"loss": loss, "acc": acc}

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## DataLoader Class
# MAGIC
# MAGIC This class holds all the logic for processing and loading the dataset.
# MAGIC
# MAGIC The default value `None` for `num_epochs` in `make_batch_reader` function is used in order to generate an infinite number of data batches to avoid handling the last, likely incomplete, batch. This is especially important for distributed training where the number of data records seen on all workers need to be identical per step. Given that the length of each data shard may not be identical, setting `num_epochs` to any specific number would fail to meet the guarantee and can result in an error. Even though this may not be really important for training on a single device, it determines the way epochs are controlled. Otherwise, training runs infinitely on an infinite dataset, which means there would be only 1 epoch if other means of controlling the epoch duration are not used.
# MAGIC
# MAGIC Using the default value `num_epochs=None` is also important for the validation process. At the time this notebook was developed, Pytorch Lightning Trainer runs a final check for completeness prior to any training, unless instructed otherwise. That check initializes the validation data loader and reads the `num_sanity_val_steps` batches from it before the first training epoch. Training does not reload the validation dataset for the actual validation phase of the first epoch which results in an error. To work around this error, you can avoid doing any checks by setting `num_sanity_val_steps=0`, and using `limit_val_batches` parameter of the Trainer class to avoid the infinitely running validation.

# COMMAND ----------

class ImageNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_parquet_files,
        val_parquet_files,
        device_id: int = 0,
        device_count: int = 1,
        batch_size: int = 16,
        num_epochs: int = 1,
        workers_count: int = 1,
        reader_pool_type: str = "dummy",
        result_queue_size: int = 1,
        feature_column: str = "content",
        label_column: str = "object_id",
        object_id_to_class_mapping: dict = object_id_to_class_mapping,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.train_dataloader_context = None
        self.val_dataloader_context = None

    def create_dataloader_context(self, input_parquet_files):
        petastorm_reader_kwargs = {
            "transform_spec": self._get_transform_spec(),
            "cur_shard": self.hparams.device_id,
            "shard_count": self.hparams.device_count,
            "workers_count": self.hparams.workers_count, # Start multiple workers to read in parallel
            "reader_pool_type": self.hparams.reader_pool_type,
            "results_queue_size": self.hparams.result_queue_size,
            "num_epochs": None,
        }
        return DataLoader(
            make_batch_reader(input_parquet_files, **petastorm_reader_kwargs),
            self.hparams.batch_size,
        )

    def train_dataloader(self):
        if self.train_dataloader_context is not None:
            self.train_dataloader_context.__exit__(None, None, None)
        self.train_dataloader_context = self.create_dataloader_context(
            self.hparams.train_parquet_files
        )
        return self.train_dataloader_context.__enter__()

    def val_dataloader(self):
        if self.val_dataloader_context is not None:
            self.val_dataloader_context.__exit__(None, None, None)
        self.val_dataloader_context = self.create_dataloader_context(
            self.hparams.val_parquet_files
        )
        return self.val_dataloader_context.__enter__()

    def teardown(self, stage=None):
        # Close all readers (especially important for distributed training to prevent errors)
        self.train_dataloader_context.__exit__(None, None, None)
        self.val_dataloader_context.__exit__(None, None, None)

    def preprocess(self, img):
        image = Image.open(io.BytesIO(img)).convert("RGB")

        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        return transform(image)

    def _transform_rows(self, batch):

        batch = batch[[self.hparams.feature_column, self.hparams.label_column]]
        # To keep things simple, use the same transformation both for training and validation
        batch[self.hparams.feature_column] = batch[self.hparams.feature_column].map(
            lambda x: self.preprocess(x).numpy()
        )
        batch[self.hparams.label_column] = batch[self.hparams.label_column].map(
            lambda object_id: self.hparams.object_id_to_class_mapping[object_id]
        )
        return batch

    def _get_transform_spec(self):
        return TransformSpec(
            self._transform_rows,
            edit_fields=[
                (self.hparams.feature_column, np.float32, (3, 224, 224), False),
                (self.hparams.label_column, np.int32, (1,), False),
            ],
            selected_fields=[self.hparams.feature_column, self.hparams.label_column],
        )

# COMMAND ----------

# MAGIC %md ## Create the training function
# MAGIC
# MAGIC The TorchDistributor API has support for single node multi-GPU training as well as multi-node training. The following `pl_train` function takes the parameters `num_tasks` and `num_proc_per_task`.
# MAGIC
# MAGIC For additional clarity:
# MAGIC - `num_tasks` is the number of **Spark Tasks** you want for distributed training.
# MAGIC - `num_proc_per_task` is the number of devices/GPUs you want per **Spark task** for distributed training.
# MAGIC
# MAGIC If you are running single node multi-GPU training on the driver, set `num_tasks` to 1 and `num_proc_per_task` to the number of GPUs that you want to use on the driver.
# MAGIC
# MAGIC If you are running multi-node training, set `num_tasks` to the number of **Spark tasks** you want to use and `num_proc_per_task` to the value of `spark.task.resource.gpu.amount` (which is usually 1).
# MAGIC
# MAGIC Therefore, the total number of GPUs used is `num_tasks * num_proc_per_task`.
# MAGIC
# MAGIC ***
# MAGIC
# MAGIC Petastorm uses the device ID and device count that is passed from the main training loop to shard data for multi-GPU training. It is crucial to specify appropriate values for Petastorm arguments such as `workers_count`, `reader_pool`, and `result_queue_size` to prevent out-of-memory (OOM) exceptions. For instance, `result_queue_size` determines the number of row groups loaded into the queue. If the size of the Parquet row groups is large, setting `result_queue_size` to a higher number can easily lead to OOM. Consider the following scenario: a row group with 1000 rows and a row size of 0.1 MB. If the default `result_queue_size` (50) and workers_count (10) are used, this would result in 50 GB of data in memory (10 workers x 50 result queue size x 1000 rows per row group x 0.1 MB).

# COMMAND ----------

BATCH_SIZE = 212
MAX_EPOCHS = 2

# petastorm specific parameters
READER_POOL_TYPE = "thread"
WORKERS_COUNT = 2
RESULTS_QUEUE_SIZE = 20


def main_training_loop(num_tasks, num_proc_per_task):
    import warnings

    warnings.filterwarnings("ignore")

    ############################
    ##### Setting up MLflow ####
    # This allows for different processes to find mlflow
    os.environ["DATABRICKS_HOST"] = db_host
    os.environ["DATABRICKS_TOKEN"] = db_token

    # NCCL P2P can cause issues with incorrect peer settings, so let's turn this off to scale for now
    os.environ["NCCL_P2P_DISABLE"] = "1"

    mlf_logger = pl.loggers.MLFlowLogger(experiment_name=experiment_path)

    WORLD_SIZE = num_tasks * num_proc_per_task
    node_rank = int(os.environ.get("NODE_RANK", 0))

    datamodule = ImageNetDataModule(
        train_parquet_files=train_parquet_files,
        val_parquet_files=val_parquet_files,
        batch_size=BATCH_SIZE,
        workers_count=WORKERS_COUNT,
        reader_pool_type=READER_POOL_TYPE,
        device_id=node_rank,
        device_count=WORLD_SIZE,
        result_queue_size=RESULTS_QUEUE_SIZE,
    )

    model = ImageNetClassificationModel(
        learning_rate=1e-5,
        device_id=node_rank,
        device_count=WORLD_SIZE,
    )

    train_steps_per_epoch = ceil(train_rows // (BATCH_SIZE * WORLD_SIZE))
    val_steps_per_epoch = ceil(val_rows // (BATCH_SIZE * WORLD_SIZE))

    if num_tasks > 1:
        strategy = "ddp"
    else:
        strategy = "auto"

    trainer = pl.Trainer(
        accelerator="gpu",
        strategy=strategy,
        devices=num_proc_per_task,
        num_nodes=num_tasks,
        max_epochs=MAX_EPOCHS,
        limit_train_batches=train_steps_per_epoch,  # this is the way to end the epoch
        val_check_interval=train_steps_per_epoch,  # this value must be the same as `limit_train_batches`
        num_sanity_val_steps=0,  # this must be zero to prevent a Petastorm error about Data Loader not being read completely
        limit_val_batches=val_steps_per_epoch,  # any value would work here but there is point in validating on repeated set of data
        reload_dataloaders_every_n_epochs=1,  # need to set this to 1 to reload dataloaders for every epoch
        use_distributed_sampler=False,  # data distribution is handled by petastorm, hence distributed sampler has to be disabled
        default_root_dir=log_path,
        logger=mlf_logger,
        enable_checkpointing=True,
    )
    print(f"strategy is {trainer.strategy}")

    trainer.fit(model, datamodule)

    return model, trainer.checkpoint_callback.best_model_path

# COMMAND ----------

# MAGIC %md ### Train the model locally with 1 GPU
# MAGIC
# MAGIC Note that `nnodes` = 1 and `nproc_per_node` = 1.

# COMMAND ----------

NUM_TASKS = 1
NUM_PROC_PER_TASK = 1

model, ckpt_path = main_training_loop(NUM_TASKS, NUM_PROC_PER_TASK)

# COMMAND ----------

# MAGIC %md ## Single node multi-GPU setup
# MAGIC
# MAGIC For the distributor API, you want to set `num_processes` to the total amount of GPUs that you plan on using. For single node multi-gpu, this is limited by the number of GPUs available on the driver node.
# MAGIC
# MAGIC As mentioned before, single node multi-gpu (with `NUM_PROC` GPUs) setup involves setting `trainer = pl.Trainer(accelerator='gpu', devices=NUM_PROC, num_nodes=1, **kwargs)`.

# COMMAND ----------

NUM_TASKS = 1
NUM_GPUS_PER_WORKER = torch.cuda.device_count()  # CHANGE AS NEEDED
USE_GPU = NUM_GPUS_PER_WORKER > 0
NUM_PROC_PER_TASK = NUM_GPUS_PER_WORKER
NUM_PROC = NUM_TASKS * NUM_PROC_PER_TASK

(model, ckpt_path) = TorchDistributor(
    num_processes=NUM_PROC, local_mode=True, use_gpu=USE_GPU
).run(main_training_loop, NUM_TASKS, NUM_PROC_PER_TASK)

# COMMAND ----------

# MAGIC %md ## Multi-node setup
# MAGIC
# MAGIC For the distributor API, you want to set `num_processes` to the total amount of GPUs that you plan on using. For multi-node, this is equal to `num_spark_tasks * num_gpus_per_spark_task`. Additionally, note that `num_gpus_per_spark_task` usually equals 1 unless you configure that value specifically.
# MAGIC
# MAGIC Multi-node with `num_proc` GPUs setup involves setting `trainer = pl.Trainer(accelerator='gpu', devices=1, num_nodes=num_proc, **kwargs)`.

# COMMAND ----------

from pyspark.ml.torch.distributor import TorchDistributor

NUM_WORKERS = 4  # CHANGE AS NEEDED
NUM_GPUS_PER_WORKER = 4
NUM_TASKS = NUM_WORKERS * NUM_GPUS_PER_WORKER
NUM_PROC_PER_TASK = 1
NUM_PROC = NUM_TASKS * NUM_PROC_PER_TASK

model, ckpt_path = TorchDistributor(
    num_processes=NUM_PROC, local_mode=False, use_gpu=True
).run(main_training_loop, NUM_TASKS, NUM_PROC_PER_TASK)
