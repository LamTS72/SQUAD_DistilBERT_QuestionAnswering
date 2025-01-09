from datasets import load_dataset, concatenate_datasets, DatasetDict
import torch
from configs.squad_config import ConfigDataset, ConfigModel
class CustomDataset():
    def __init__(self,
                 path_dataset=ConfigDataset.PATH_DATASET,
                 revision=ConfigDataset.REVISION,
                 train_size=ConfigModel.TRAIN_SIZE,
                 flag_info=True
                ):
      self.raw_data = load_dataset(path_dataset)
      self.size = len(self.raw_data["train"]) + len(self.raw_data["validation"])
      if flag_info:
        print("-"*50, "Information of Dataset", "-"*50)
        print(self.raw_data)
        print("-"*50, "Information of Dataset", "-"*50)


    def __len__(self):
      return self.size

    def __getitem__(self, index):
      dataset = self.raw_data["train"]
      data = dataset[index]["question"]
      context = dataset[index]["context"]
      target = dataset[index]["answers"]
      return {
          "question": data,
          "context": context,
          "answers": target
      }

