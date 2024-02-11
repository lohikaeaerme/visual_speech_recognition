import torch
import torchvision
import cv2
import os


class VideoFolderPathToTensor(object):


    def __init__(self,data_dict, num_frames, max_len=None):
        self.max_len = max_len
        self.num_frames = num_frames
        self.data_dict = data_dict

    def __call__(self,path):

        #with open(path) as file:
        #    text = file.read()

        #numbers = text.split(',')
        if path in self.data_dict:
            data = self.data_dict[path]

        else :
            print('missing key :' + path)
            print(len(self.data_dict.keys()))
            with open(path) as file:
                text = file.read()

                numbers = text.split(',')
                data = []
                for number in numbers:
                    if number != ' ':
                        data.append(int(number))

        inputlen = self.num_frames*68*2
        input = torch.FloatTensor(inputlen)
        
        for index,number in enumerate(data):
            if index < inputlen:
                input[index] = (number + 50 ) / 550

        return input