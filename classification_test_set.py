from pathlib import Path
from math import ceil

import pandas as pd
import torch
import pytorch_lightning as pl

from classification.train_base import MultiPartitioningClassifier # class defining our model
from classification.dataset import FiveCropImageDataset # class for preparing the images before giving them to the NN

# where model's params and hyperparams are saved
checkpoint = "models/base_M/epoch=014-val_loss=18.4833.ckpt"
hparams = "models/base_M/hparams.yaml"
# where images are saved
image_dir = "resources/images/im2gps"
meta_csv = "resources/images/im2gps_places365.csv"

print("Loading the model from checkpoint", checkpoint)
# load_from_checkpoint is a static method from pytorch lightning, inherited by MultiPartitioningClassifier
# it permits to load a model previously saved, in the form of a checkpoint file, and one with hyperparameters
# MultiPartitioningClassifier is the class defining our model
model = MultiPartitioningClassifier.load_from_checkpoint(
    checkpoint_path=checkpoint,
    hparams_file=hparams,
    map_location=None
)

#to allow GPU
want_gpu = True
if want_gpu and torch.cuda.is_available():
    gpu = 1
else:
    gpu = None

# the class Trainer from pythorch lightining is the one responsible for training a deep NN
# in particular, it can initialize the model, run forward and backward passes, optimize, print statistics, early stop...
wanted_precision = 32 #16 for half precision (how many bits for each number)
trainer = pl.Trainer(gpus=gpu, precision=wanted_precision)

print("Initializing the test set")
dataset = FiveCropImageDataset(meta_csv, image_dir)

batch_size = 64
dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=ceil(batch_size / 5),  #you divide by 5 because for each image you generate 5 different crops
                    shuffle=False,
                    num_workers=4 #number ot threads used for parallelism (cores of CPU?)
                )


print("Running the model on the test set")
results = trainer.test(model, test_dataloaders=dataloader, verbose=False)

# formatting results into a pandas dataframe
df = pd.DataFrame(results[0]).T
#df["dataset"] = image_dir
df["partitioning"] = df.index
df["partitioning"] = df["partitioning"].apply(lambda x: x.split("/")[-1])
df.set_index(keys=["partitioning"], inplace=True) #keys=["dataset", "partitioning"] in case
print(df)

#saving the results on a csv
fout = 'test_results.csv'
df.to_csv(fout)
