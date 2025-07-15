import argparse
import random
import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils.eval_indicators as ei
from mydataset.mydataset import my_dataset_test
from utils.tools import read_tif,create_model,get_inviad_mask
import os
#Set the random number seed
SEED = 971108
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#Set global parameters
parser = argparse.ArgumentParser()
parser.add_argument("--in_channels",type=int,default=6)
parser.add_argument("--num_class",type=int,default=2)
args = parser.parse_args()

def load_data(name):
    # Get the input image and label
    img="../Sentinel2/img/"+name+".tif"
    GT="../Sentinel2/island_mask/"+name+".tif"
    MG2="../MG2/"+name+".tif"
    img=read_tif(img)
    GT=read_tif(GT)
    if os.path.exists(MG2):
        MG2 = read_tif(MG2)
    else:
        MG2 = np.zeros_like(GT).astype(np.int16)
    return img,GT,MG2

def test(args,name,model_name,inviad_mask,cfg_model,testloader):
    cfg = {"model":cfg_model}
    model=create_model(cfg)
    model.load_state_dict(torch.load(model_name,map_location=DEVICE),strict=False)
    model=model.to(DEVICE)
    model.eval()
    pred_label=[]
    target_lable=[]
    stride=256
    patch=256
    with torch.no_grad():
        for _, (img, label) in enumerate(testloader):
            img = img.permute(0, 3, 1, 2)
            img = img.to(DEVICE)
            label = label.to(DEVICE)
            pred_output = torch.zeros((label.shape[0], args.num_class, label.shape[1], label.shape[2])).to(DEVICE)
            b, h, w = label.shape
            h_s = h // stride+1
            w_s = w // stride+1
            for i in range(h_s):
                for j in range(w_s):
                    input = img[:, :, stride * i:(stride * i + patch),
                            stride * j:(stride * j + patch)].to(DEVICE)
                    if input.shape[2] < patch and input.shape[3] < patch:
                        input = img[:, :, h - patch:h, w - patch:w].to(DEVICE)
                        outs = model(input)
                        pred_output[:, :, h - patch:h, w - patch:w] = pred_output[:,:,h - patch:h, w - patch:w] + outs
                        continue
                    if input.shape[2] < patch and input.shape[3] == patch:
                        input = img[:, :, h - patch:h, stride * j:(stride * j + patch)].to(DEVICE)
                        outs = model(input)
                        pred_output[:, :, h - patch:h, stride * j:(stride * j + patch)] = pred_output[:,:,h - patch:h, stride * j:(stride * j + patch)] + outs
                        continue
                    if input.shape[2] == patch and input.shape[3] < patch:
                        input = img[:, :, stride * i:(stride * i + patch), w - patch:w].to(DEVICE)
                        outs = model(input)
                        pred_output[:, :, stride * i:(stride * i + patch), w - patch:w] = pred_output[:,:,stride * i:(stride * i + patch), w - patch:w] + outs
                        continue
                    outs = model(input)
                    pred_output[:, :, stride * i:(stride * i + patch), stride * j:(stride * j + patch)] = pred_output[:,:,stride * i:(stride * i + patch),stride * j:(stride * j + patch)] + outs
            pred = torch.argmax(pred_output.data, 1)
            pred=np.squeeze(pred,axis=0)
            pred[inviad_mask==1]=0
            pred_label.append(np.array(pred.cpu()))
            target_lable.append(np.array(label.cpu()))

    metric = ei.SegmentationMetric(2)  # two classes
    ignore_labels = [2,255]
    for i in range(len(pred_label)):
        pred_label_tensor = torch.tensor(pred_label[i], dtype=torch.long)
        target_lable_i=target_lable[i]
        target_lable_tensor = torch.tensor(target_lable_i, dtype=torch.long)
        target_lable_tensor=torch.squeeze(target_lable_tensor)
        hist = metric.addBatch(pred_label_tensor, target_lable_tensor, ignore_labels)

    # OA
    OA = metric.OA()
    # UA
    UA=metric.UA()
    # PA
    PA=metric.PA()
    # kappa
    kappa = metric.Kappa()
    # F1 water
    F1 = metric.F1_score(1)
    #
    print(metric.confusionMatrix)
    print("OA:{},UA:{},PA:{},kappa:{},F1:{}".format(OA,UA,PA,kappa,F1))

if __name__ == '__main__':
    img_name = ["T30TXQ_20180911_Bordeaux_summer"]
    models_name=["../output/T30TXQ_20180911_Bordeaux_summer/models/model.pth",]
    for i in range(len(img_name)):
        name=img_name[i]
        model_name=models_name[i]
        test_data, test_label, MG2 = load_data(name)
        temp=np.zeros_like(test_label)
        _,inviad_mask = get_inviad_mask(temp, MG2, test_label.copy(), test_data,name)
        # Build training and testing datasets
        test_data, test_label = test_data.astype(np.float32), test_label.astype(np.int64)
        test_dataset = my_dataset_test(test_data, test_label, 256)
        testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
        cfg_model = "CAMFNet"
        test(args, name,model_name, inviad_mask, cfg_model=cfg_model, testloader=testloader)
