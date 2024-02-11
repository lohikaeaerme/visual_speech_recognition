from Multi_Model_loader.train import train
from lstm.lstm_julia.models import cnnlstm 

csv_path = 'dataset_3/lip_videos_words/dataset_no_A.csv'
train(cnnlstm.CNNLSTM(num_classes=10),csv_path,'longlstm',90,'d3',num_epochs=100,lr=0.0005)