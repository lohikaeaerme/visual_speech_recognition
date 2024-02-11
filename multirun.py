from Multi_Model_loader.train import train
from lipreading_restcn.res_tnc import ResTCN
from vision_transformer.vivit import ViViT
from lstm.lstm_julia.models import cnnlstm
from soundnet.audioMNIST_solution_julia.model import SoundClassifier
from lipreading_facialpoints.facialpoint_model import facialpoint_model
from lipreading_facialpoints import dataloader as facialDataloader
from sklearn.model_selection import train_test_split
import pandas as pd
csv_path = '/media/fabian/Elch/Bachelorarbeit/dataset_3/lip_videos_words/dataset_no_A.csv'
print('start')

def generate_facial_dataloader(csv_path,num_frames,batch_size=16):
    df = pd.read_csv(csv_path,converters={'keypoints' : lambda x: [int(y) for y in x.strip("[]").replace("'","").split(", ")] })
    data_dict = {}
    for i,row in df.iterrows():
        #print('did some thing here : ' + str(i) + ' ' + row.path)
        data_dict[row.path] = row.keypoints

    print(len(data_dict.keys()))

    df_train, df_test = train_test_split(df,test_size=0.2) 

    dataloader = facialDataloader.get_dataloader(batch_size,
                                num_frames,                 
                                df_train,
                                df_test,
                                data_dict)

    dataset_sizes = {x: len(dataloader[x].dataset) for x in ['train', 'test']}
    print(dataset_sizes, flush=True)
    return dataloader

for lr in [ 0.001, 0.0005, 0.0001, 0.01 ]:
    train(ResTCN(),csv_path,'RestTCN',90,'d3', lr=lr,num_epochs=5,batch_size=4)
    train(cnnlstm.CNNLSTM(num_classes=10),csv_path,'lstm',90,'d3',num_epochs=5,lr=lr)


for lr in [  0.00001, 0.0001, 0.001, 0.0005, 0.000005, 0.01]:
    train(ViViT(224, 16, 10, 90), csv_path, 'Vivit', 90, 'd3', lr=lr, num_epochs=5,batch_size=4)



for lr in [ 0.001, 0.0005, 0.0001, 0.01 ]:

    facial_dataloader = generate_facial_dataloader('dataset_3/lip_points/dataset_with_keypoints.csv',90)
    train(facialpoint_model(90*2*68,10), csv_path, 'FacialPoint', 90, 'd3', lr=lr, num_epochs=5,dataloader=facial_dataloader)
    train(SoundClassifier(),csv_path,'sound',90,'d3',lr=lr,num_epochs=5)



train(ResTCN(),"dataset_3/lip_point_videos_224/dataset_lippoints_224_no_A.csv",'RestTCN',90,'fp224', lr=0.001,num_epochs=5,batch_size=4)
train(ResTCN(),"dataset_3/lip_point_videos_64/dataset_lippoints_64_no_A.csv",'RestTCN',90,'fp64', lr=0.001,num_epochs=5, img_size=64)
train(ResTCN(),'dataset_3/lip_point_videos_32/dataset_lippoints_32_no_A.csv','RestTCN',90,'fp32', lr=0.001,num_epochs=5, img_size=32)


# Scheduler Step
# 

print('done')