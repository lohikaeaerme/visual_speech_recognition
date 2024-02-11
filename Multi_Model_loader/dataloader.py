import torchvision
from torch.utils.data import DataLoader

from . import dataset
from . import transforms


def generate_dataloader(batch_size, num_frames, img_size,df):
    ds = dataset.VideoDataset(df,
                                    transform=torchvision.transforms.Compose([transforms.VideoFolderPathToTensor(num_frames,img_size=img_size)]))

    return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=4)


def get_dataloader(batch_size, num_frames, df_train, df_test, img_size = 224):
    return {
        'train': generate_dataloader(batch_size, num_frames, img_size, df_train),
        'test': generate_dataloader(batch_size, num_frames, img_size, df_test)}