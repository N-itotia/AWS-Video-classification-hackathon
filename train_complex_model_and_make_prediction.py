import os
import torch
import pytorch_lightning
import dill
import itertools

import torch.nn as nn
import torch.nn.functional as F

import pytorchvideo.data
import torch.utils.data
import pytorchvideo.models.resnet

# from pathos.multiprocessing import ProcessingPool as Pool
# from habana_frameworks.torch.utils.library_loader import load_habana_module


from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    CenterCrop,
    RandomHorizontalFlip
)


class KineticsDataModule(pytorch_lightning.LightningDataModule):
    videos_root = os.path.join(os.getcwd())
    dataset_location = os.path.join(videos_root, "data")
    
    cpuCount = int(os.cpu_count())
    workers_int = cpuCount-2

    # Dataset configuration
    _DATA_PATH = dataset_location
    _CLIP_DURATION = 2  # Duration of sampled clip for each video
    _BATCH_SIZE = 8
    _NUM_WORKERS = 20 # Number of parallel processes fetching data Change to 1/4 number of CPUs
    


    def train_dataloader(self):
        """
        Create the Kinetics train partition from the list of video labels
        in {self._DATA_PATH}/train.csv. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """
        # lambda x: x+1
        def tmp_lambda_func(x):
            return ( x / 255.0)
        
        
        train_transform = Compose(
            [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                    UniformTemporalSubsample(8),
                    Lambda(tmp_lambda_func
                           ),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(244),
                    RandomHorizontalFlip(p=0.5),
                    ]
                ),
                ),
            ]
        )
        train_dataset = LimitDataset(pytorchvideo.data.Kinetics(
            data_path=os.path.join(self._DATA_PATH, "train"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("random", self._CLIP_DURATION),
            decode_audio=False,
            transform=train_transform
            )
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )


    def val_dataloader(self):
        """
        Create the Kinetics validation partition from the list of video labels
        in {self._DATA_PATH}/val
        """
                # It creates a dataset and a dataloader for the validation set.
                # lambda x: x+1
        def tmp_lambda_func(x):
            return (x / 255.0)
        
        val_transform = Compose(
            [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                    UniformTemporalSubsample(8),
                    Lambda(tmp_lambda_func),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    ShortSideScale(size=256),
                    CenterCrop(244),
                    ]
                ),
                ),
            ]
        )
        
        val_dataset = pytorchvideo.data.Kinetics(
            data_path=os.path.join(self._DATA_PATH, "validation"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self._CLIP_DURATION),
            decode_audio=False,
            transform=val_transform
        )
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )
        
def make_kinetics_resnet():
  return pytorchvideo.models.resnet.create_resnet(
      input_channel=3, # RGB input from Kinetics
      model_depth=50, # For the tutorial let's just use a 50 layer network
      model_num_class=31, # Kinetics has 400 classes so we need out final head to align
      norm=nn.BatchNorm3d,
      activation=nn.ReLU,
  )

class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
  def __init__(self):
      super().__init__()
      self.model = make_kinetics_resnet()

  def forward(self, x):
      return self.model(x)

  def training_step(self, batch, batch_idx):
      # The model expects a video tensor of shape (B, C, T, H, W), which is the
      # format provided by the dataset
      y_hat = self.model(batch["video"])

      # Compute cross entropy loss, loss.backwards will be called behind the scenes
      # by PyTorchLightning after being returned from this method.
      loss = F.cross_entropy(y_hat, batch["label"])

      # Log the train loss to Tensorboard
      self.log("train_loss", loss.item())

      return loss

  def validation_step(self, batch, batch_idx):
      y_hat = self.model(batch["video"])
      loss = F.cross_entropy(y_hat, batch["label"])
      self.log("val_loss", loss)
      return loss

  def configure_optimizers(self):
      """
      Setup the Adam optimizer. Note, that this function also can return a lr scheduler, which is
      usually useful for training video models.
      """
      return torch.optim.Adam(self.parameters(), lr=1e-1)
  
class LimitDataset(torch.utils.data.Dataset):
    """
    To ensure a constant number of samples are retrieved from the dataset we use this
    LimitDataset wrapper. This is necessary because several of the underlying videos
    may be corrupted while fetching or decoding, however, we always want the same
    number of steps per epoch.
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(dataset), 2)
        )

    def __getitem__(self, index):
        return next(self.dataset_iter)

    def __len__(self):
        return self.dataset.num_videos
  
'''
Train the model
'''
def train():
    classification_module = VideoClassificationLightningModule()
    data_module = KineticsDataModule()
    # device = torch.device("hpu")
    # model_inputs = inputs.to(device)
    # model(model_inputs)
    # load_habana_module()
    cpuCount = int(os.cpu_count())
    workers_int = cpuCount-4
    trainer = pytorch_lightning.Trainer(accelerator='cpu', num_processes = workers_int) #gpus=torch.cuda.device_count(),
    trainer.fit(classification_module, data_module)
    
if __name__ == "__main__":
    train()
