import torch
import torchvision
import cv2
import os
from torchvision.transforms import v2


class VideoFolderPathToTensor(object):

    def __init__(self, num_frames, img_size=224, max_len=None):
        self.max_len = max_len
        self.num_frames = num_frames
        self.img_size = img_size

    def __call__(self, path):

        #new Frames now read from Video -------------------------
        #file_names = sorted([os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
        #frames_path = [os.path.join(path, f) for f in file_names]
#
        #frame = cv2.imread(frames_path[0])

        video = cv2.VideoCapture('/media/fabian/Elch/Bachelorarbeit/' + path)
        sucess, frame = video.read()
        #if not sucess:
        #   print('Error reading Video from File :', path)

        height, width, channels = (self.img_size,self.img_size,3) #frame.shape
        #num_frames = len(frames_path)
        #---------------------------------

        transform = v2.Compose([
            v2.ToPILImage(),
            v2.ToTensor(),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            v2.Resize([self.img_size, self.img_size],antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomResizedCrop(size=(self.img_size, self.img_size), antialias=True),
        ])

        # EXTRACT_FREQUENCY = 18
        EXTRACT_FREQUENCY = 1

        # num_time_steps = int(num_frames / EXTRACT_FREQUENCY)

        #new now 120 Frame pro Video
        num_time_steps = self.num_frames
        # num_time_steps = 4

        # (3 x T x H x W), https://pytorch.org/docs/stable/torchvision/models.html
        frames = torch.FloatTensor(channels, num_time_steps, self.img_size, self.img_size) #new 

        for index in range(0, num_time_steps):
            #if not sucess:
                #new check if Video reading worked
                #print('Error reading Frame : ', index , ' in Video ' , path) 
            if sucess:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                frame = frame.permute(2, 0, 1)  # (H x W x C) to (C x H x W)
                frame = frame / 255
#                if frame.shape[2] != 224:
#                    frame = frame[:, :, 80:560]
                frame = transform(frame)
                frames[:, index, :, :] = frame.float()

            sucess, frame = video.read() #new Frame from Video
        return frames.permute(1, 0, 2, 3)