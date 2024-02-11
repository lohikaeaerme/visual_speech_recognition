import os
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import torch
from torch import nn, optim
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import sys
import os
import gc
from .dataloader import get_dataloader
from datetime import datetime
def train( model, csv_path, model_name, num_frames, dataset_name, lr = .0001, batch_size=16, num_epochs=5, use_adam=True, dataloader=None, img_size = 224):

    
    output_name = f'{model_name}_{ "adam" if use_adam else "SGD" }_dataset_{dataset_name}_lr{lr}'
    output_path = f'Multi_Model_loader/Output/{output_name}'

    if os.path.exists(f'{output_path}.txt'):
        return

    sys.stdout = open(f'{output_path}.txt','wt')
    os.mkdir(output_path)

    torch.manual_seed(0)
    use_cuda = True
    #new Device changed for m1 
    device = torch.device("cuda" if use_cuda else "cpu")
    #device = torch.device("mps")
    print("Device being used:", device, flush=True)

    #new load csv to generate Test and Train data
    df = pd.read_csv(csv_path)
    df_train, df_test = train_test_split(df,test_size=0.2) 

    if (dataloader == None):
        print('init Dataloader')
        dataloader = get_dataloader(batch_size,
                                    num_frames,
                                    df_train,
                                    df_test,
                                    img_size=img_size)

    dataset_sizes = {x: len(dataloader[x].dataset) for x in ['train', 'test']}
    print(dataset_sizes, flush=True)


    #model = ResTCN().to(device)
    model = model.to(device)
    #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    #scheduler = StepLR(optimizer, step_size=5, gamma=.1)


    print(len(df_train))
    if(use_adam):
        optimizer = torch.optim.Adam(model.parameters(),lr=lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
                                                    steps_per_epoch=int(len(df_train) / batch_size),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        scheduler = StepLR(optimizer, step_size=5, gamma=.1)

    criterion = nn.CrossEntropyLoss().to(device)
    softmax = nn.Softmax(dim = -1) #new added Dimension -1 to stop warning

    for epoch in range(num_epochs):
        torch.save(model.state_dict(),f'{output_path}/model_lr{lr}.pt')
        checkpoint = { 
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'learning_rate':lr,
            'adam': use_adam,
            'lr_sched': scheduler}
        torch.save(checkpoint, f'{output_path}/checkpoint_lr{lr}.pth')


        print(f'start ecpoch {epoch} time {datetime.now().time()}')
        for phase in ['train', 'test']:

            running_loss = .0
            y_trues = np.empty([0])
            y_preds = np.empty([0])

            if phase == 'train':
                model.train()
            else:
                model.eval()

            for inputs, labels in tqdm(dataloader[phase], disable=True):
                inputs = inputs.to(device)
                labels = labels.long().squeeze().to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    outputs = outputs.squeeze()
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        #print('Loss backward ' + str(loss.item()))
                        #print('for label ' + str(labels[0]) + ' predicted ' + str(outputs[0]))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()


                running_loss += loss.item() * inputs.size(0)
                #print("running loss increment = " +  str(loss.item()) + '*' + str(inputs.size(0)) + ' = ' + str(loss.item() * inputs.size(0)))

                
                if len(outputs.shape) != 2:
                    outputs = torch.unsqueeze(outputs,0)
                preds = torch.max(softmax(outputs), 1)[1] 
                #print('true Label' + str(labels) + 'prediction ' + str(preds) + 'output softmax ' + str(softmax(outputs)) + ' max ' + str(torch.max(softmax(outputs), 1)))
                y_trues = np.append(y_trues, labels.data.cpu().numpy())
                y_preds = np.append(y_preds, preds.cpu())

            if phase == 'train':
                 scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            #print('running Loss : ' + str(running_loss) + ' data Size : ' + str(dataset_sizes[phase]))

            print("[{}] Epoch: {}/{} Loss: {} LR: {}".format(
                phase, epoch + 1, num_epochs, epoch_loss, scheduler.get_last_lr()), flush=True)
            print('\nconfusion matrix\n' + str(confusion_matrix(y_trues, y_preds)))
            print('\naccuracy\t' + str(accuracy_score(y_trues, y_preds)))
        
    torch.save(model.state_dict(),f'{output_path}/model_{num_epochs}_lr{lr}.pt')

    checkpoint = { 
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'learning_rate':lr,
        'adam': use_adam,
        'lr_sched': scheduler}
    torch.save(checkpoint, f'{output_path}/checkpoint_{num_epochs}_lr{lr}.pth')

    labels.cpu()
    model.cpu()
    criterion.cpu()
    inputs.cpu()
    del model, criterion, inputs,labels, preds
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':

    for lr in [ .001,.00001 , .0005 , .00005, .0002 ,.00002,.0001,]:
        print("----------------------------------------------------------")
        print(f'starting with learningrate : {lr}')
        main(lr)