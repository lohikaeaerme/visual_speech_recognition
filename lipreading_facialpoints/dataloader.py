import torchvision
from torch.utils.data import DataLoader

from . import dataset
from . import transforms


def generate_dataloader(batch_size, num_frames, df,data_dict):
    ds = dataset.VideoDataset(df, transform=torchvision.transforms.Compose([transforms.VideoFolderPathToTensor(data_dict, num_frames)]))

    return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=4)


def get_dataloader(batch_size, num_frames, df_train, df_test, data_dict):
    return {
        'train': generate_dataloader(batch_size, num_frames, df_train, data_dict),
        'test': generate_dataloader(batch_size, num_frames, df_test, data_dict)}