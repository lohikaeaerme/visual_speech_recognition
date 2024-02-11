from torch.utils.data import Dataset
import pandas as pd
import os


class VideoDataset(Dataset):

    def __init__(self, dataframe, transform=None):

        self.dataframe = dataframe #new Dateframe already loaded in train.py and no root
        self.transform = transform
        #self.root = root

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        label = self.dataframe.iloc[index].label
        #new no root
        #video = os.path.join(self.root, self.dataframe.iloc[index].path) 
        #video = self.transform(video)

        video = self.transform(self.dataframe.iloc[index].path)

        return video, label