import matplotlib.pyplot as plt
import argparse
import random
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import os
import copy
import cv2
import utils.eval_indicators as ei
from mydataset.mydataset import mytrain_dataset,my_dataset_test
from utils.tools import read_tif,create_model,compute_weight,get_inviad_mask
from tqdm import tqdm
#Set the random number seed
SEED = 971104
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED) #
DEVICE = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(name):
    # Get input image and pseudo label
    img="../Sentinel2/img/"+name+".tif"
    pseudo_label="../Sentinel2/UUCP/"+name+"/over/"+name+".tif"
    # GT is only used to remove the ocean and does not participate in training. It can also be omitted
    GT="../Sentinel2/island_mask/"+name+".tif"
    easy_sample="../Sentinel2/UUCP/"+name+"/over/certain.tif"
    hard_sample="../Sentinel2/UUCP/"+name+"/over/potential.tif"
    certain_no_water="../Sentinel2/UUCP/"+name+"/over/certain_no_water.tif"
    potential_no_water="../Sentinel2/UUCP/"+name+"/over/potential_no_water.tif"
    MG2="../Sentinel2/MG2/"+name+".tif"
    ##Get training images and pseudo labels
    img=read_tif(img)
    pseudo_label=read_tif(pseudo_label)
    easy_sample=read_tif(easy_sample)
    hard_sample=read_tif(hard_sample)
    GT=read_tif(GT)
    certain_no_water=read_tif(certain_no_water)
    potential_no_water=read_tif(potential_no_water)
    if os.path.exists(MG2):
        MG2 = read_tif(MG2)
    else:
        MG2 = np.zeros_like(pseudo_label).astype(np.int16)
    return img,pseudo_label,GT,easy_sample,hard_sample,certain_no_water,potential_no_water,MG2

def main(args,name):
    create_path(args.save_path)
    #Read input images and pseudo labels
    img,pseudo_label,GT,easy_samples,hard_samples,certain_no_water,potential_no_water,MG2,=load_data(name=name)
    pseudo_label,inviad_mask=get_inviad_mask(pseudo_label.copy(),MG2.copy(),GT.copy(),img.copy(),name)
    loss_weight=compute_weight(img.copy(),inviad_mask.copy())
    print("loss weight over")

    pseudo_label_copy2=pseudo_label.copy()
    pseudo_label_copy2[pseudo_label_copy2 == 0] = 2
    pseudo_label_copy2[certain_no_water == 0] = 0
    pseudo_label_copy2[potential_no_water == 0] = 0
    pseudo_label_copy2[easy_samples == 1] = 1
    pseudo_label_copy2[hard_samples == 1] = 1
    pseudo_label_copy2[inviad_mask==1]=2
    pseudo_label_copy2[pseudo_label == 3] = 0

    #Constructing training and testing datasets
    img, pseudo_label= img.astype(np.float32), pseudo_label.astype(np.int64)
    train_dataset=mytrain_dataset(img.copy(),pseudo_label,loss_weight,easy_samples,hard_samples,certain_no_water,potential_no_water,mode="pre_train",patchsz=args.train_patch)
    dataset=train_dataset.get_dataset()
    train_loader =DataLoader(train_dataset,batch_size=4,shuffle=True,drop_last=True)
    print("dataset over")

    #Build the model
    cfg = {"model":"CAMFNet"}
    model=create_model(cfg)
    # can load parameters trained on water datasets or other areas to help the model converge faster or be more stable
    # model.load_state_dict(torch.load("pretrained.pth",map_location=DEVICE),strict=False)
    model=model.to(DEVICE)
    celoss = nn.CrossEntropyLoss(ignore_index=2, reduction="none")
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr,momentum=0.9,weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=10, gamma=0.1)  # * iters
    #Set the initial confidence matrix
    confidence_weight=np.zeros_like(pseudo_label,dtype=np.float64)
    label_refurbishment=np.zeros_like(pseudo_label,dtype=np.float64)
    pre_train_epoch=args.pre_epochs
    for epochs in range(pre_train_epoch):
        print("\n==> Epoch %i, learning rate = %.6f" %
              (epochs, optimizer.param_groups[0]["lr"]))
        model,optimizer,train_loss=pre_train(model,criterion=celoss,optimizer=optimizer,dataloader=train_loader,device=DEVICE)
        scheduler.step()
        if epochs>=10:
            # Calculate label confidence. After 10 epochs, the model reaches a certain accuracy and then starts calculating label confidence.
            model.eval()
            # Re-predict the entire image
            stride = 256
            patch = 256
            h,w=img.shape[1],img.shape[2]
            with torch.no_grad():
                large_img=np.expand_dims(img.copy()/10000.,axis=0)
                large_img=torch.tensor(large_img,dtype=torch.float32).to(DEVICE)
                label = pseudo_label.copy()
                pred_output = torch.zeros(1,args.num_class, h, w).to(DEVICE)
                h_s = h // stride + 1
                w_s = w // stride + 1
                for i in range(h_s):
                    for j in range(w_s):
                        input = large_img[:, :, stride * i:(stride * i + patch),
                                stride * j:(stride * j + patch)]
                        if input.shape[2] < patch and input.shape[3] < patch:
                            input = large_img[:, :, h - patch:h, w - patch:w]
                            outs = model(input)
                            pred_output[:, :, h - patch:h, w - patch:w] = pred_output[:, :, h - patch:h,
                                                                          w - patch:w] + outs
                            continue
                        if input.shape[2] < patch and input.shape[3] == patch:
                            input = large_img[:, :, h - patch:h, stride * j:(stride * j + patch)]
                            outs = model(input)
                            pred_output[:, :, h - patch:h, stride * j:(stride * j + patch)] = pred_output[:, :,
                                                                                              h - patch:h,
                                                                                              stride * j:(
                                                                                                          stride * j + patch)] + outs
                            continue
                        if input.shape[2] == patch and input.shape[3] < patch:
                            input = large_img[:, :, stride * i:(stride * i + patch), w - patch:w]
                            outs = model(input)
                            pred_output[:, :, stride * i:(stride * i + patch), w - patch:w] = pred_output[:, :,
                                                                                              stride * i:(
                                                                                                          stride * i + patch),
                                                                                              w - patch:w] + outs
                            continue
                        outs = model(input)
                        pred_output[:, :, stride * i:(stride * i + patch),
                        stride * j:(stride * j + patch)] = pred_output[:, :, stride * i:(stride * i + patch),
                                                           stride * j:(stride * j + patch)] + outs

            pred = torch.argmax(pred_output.data, 1)
            pred=np.squeeze(pred,axis=0)
            pred=pred.cpu().numpy()
            #Count the number of times the model incorrectly predicts the pseudo-label
            pred[inviad_mask==1]=0
            confidence_weight[pred != label] = confidence_weight[pred != label] + 0.1
            label_refurbishment=label_refurbishment+pred

    #In the second stage, the confidence_weight is first used to update the label, and then the model is updated using the updated label and loss weight.
    update_label=pseudo_label.copy()
    update_label[confidence_weight>=0.9]=1-pseudo_label_copy2[confidence_weight>=0.9]
    update_label[update_label==-1]=2
    update_label[inviad_mask==1]=0
    update_label_copy=update_label.copy()
    label_refurbishment[label_refurbishment<5]=0
    label_refurbishment[label_refurbishment>=5]=1

    update_label[update_label_copy==2]=label_refurbishment[update_label_copy==2]
    update_label,_=get_inviad_mask(update_label,MG2,GT.copy(),img,name)
    img, update_label = img.astype(np.float32), update_label.astype(np.int64)
    train_dataset = mytrain_dataset(img, update_label, loss_weight, easy_samples, hard_samples,certain_no_water,potential_no_water,dataset=dataset,mode="train",
                                    patchsz=args.fine_tuning_patch)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,drop_last=True)

    # Set loss function, optimizer, etc.
    celoss = nn.CrossEntropyLoss(ignore_index=2, reduction="none")
    optimizer = optim.SGD(model.parameters(), lr=args.fine_tuning_lr, momentum=0.9, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=5, gamma=0.1)  # * iters
    epochs = args.epochs
    for epoch in range(epochs):
        print("\n==> Epoch %i, learning rate = %.4f" %
              (epoch, optimizer.param_groups[0]["lr"]))
        model, optimizer, train_loss = train(model, criterion=celoss, optimizer=optimizer, dataloader=train_loader,
                                                 device=DEVICE)
        scheduler.step()
        #
        if epoch==0:
            torch.save(model.state_dict(),
                       os.path.join(args.save_path, '%s_%first.4f.pth' % (cfg["model"],  epoch)))
        if epoch==epochs-1:
            torch.save(model.state_dict(),
                        os.path.join(args.save_path, '%s_%.4f.pth' % (cfg["model"],  epoch)))

def pre_train(model,criterion,optimizer,dataloader,device):
    '''
    :param model:
    :param criterion:
    :param optimizer:
    :param dataloader:
    :param device: cuda
    :return:
    '''
    model.train()
    model.to(device)
    trainLoss = []
    total_loss = 0.0
    tbar = tqdm(dataloader)
    for step,  (neighbor_region, target,weight) in enumerate(tbar):
        input=neighbor_region.to(device)
        input=input.permute(0,3,1,2)
        out = model(input)

        target=target.to(device)
        target=target.permute(0,3,1,2)
        target=torch.squeeze(target)
        loss=criterion(out,target)
        loss=torch.mean(loss)

        trainLoss.append(loss.item())
        total_loss=total_loss+loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tbar.set_description('Loss: %.3f' % (total_loss / (step + 1)))

    return model, optimizer,float(np.mean(trainLoss)),

def train(model,criterion,optimizer,dataloader,device):
    '''
    :param model:
    :param criterion:
    :param optimizer:
    :param dataloader:
    :param device: cuda
    :return:
    '''
    model.train()
    model.to(device)
    trainLoss = []
    total_loss = 0.0
    tbar = tqdm(dataloader)
    for step,  (neighbor_region, target,weight) in enumerate(tbar):
        input=neighbor_region.to(device)
        input=input.permute(0,3,1,2)
        weight=weight.to(device)
        weight=weight.permute(0,3,1,2)
        out = model(input)

        target=target.to(device)
        target=target.permute(0,3,1,2)
        target=torch.squeeze(target)
        loss=criterion(out,target)
        loss=loss*weight
        loss=torch.mean(loss)
        trainLoss.append(loss.item())
        total_loss=total_loss+loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tbar.set_description('Loss: %.3f' % (total_loss / (step + 1)))

    return model, optimizer,float(np.mean(trainLoss)),

if __name__ == '__main__':
    # Setting global parameters
    Name=["T30TXQ_20180911_Bordeaux_summer","T30TXQ_20190223_Bordeaux_winter","T30TXR_20190223_Gironde_winter","T30TYQ_20190222_Marmande_winter",
          "T30UXU_20180708_Bretagne_summer","T30UXU_20190223_Bretagne_winter","T31TCH_20181023_Ariege_summer","T31TCH_20190322_Ariege_winter",
          "T31TCM_20180819_Chateauroux_summer","T31TCM_20190225_Chateauroux_winter","T31TFJ_20180927_Camargue_summer","T31TFJ_20190331_Camargue_winter",
          "T31TGL_20180828_Alpes_summer","T32ULU_20180912_Alsace_summer","T32ULU_20190321_Alsace_winter","Guangzhou","Wuhan","Guangzhou","Wuhan"]
    for name in Name:
        print(name)
        DATASET = name
        parser = argparse.ArgumentParser()
        parser.add_argument("--base_lr", type=float, default=0.001,)
        parser.add_argument("--pre_epochs", type=int, default=20,)
        parser.add_argument("--epochs", type=int, default=10,)
        parser.add_argument("--fine_tuning_lr", type=float, default=0.001,)
        parser.add_argument("--in_channels", type=int, default=6,)
        parser.add_argument("--num_class", type=int, default=2, )
        parser.add_argument("--batch_size", type=int, default=4, )
        parser.add_argument("--train_patch", type=int, default=256, )
        parser.add_argument("--fine_tuning_patch", type=int, default=256,)
        parser.add_argument("--test_patch", type=int, default=256,)
        parser.add_argument('--save-path', type=str, default="../output/" + DATASET + '/models')
        args = parser.parse_args()
        main(args, name=name)



