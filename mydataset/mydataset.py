import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import albumentations as A

SEED = 971103
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class mytrain_dataset(Dataset):
    def __init__(self, data, label,weight,certain,potential,certain_no_water,potential_no_water,mode,dataset=None,patchsz=512):
        '''
        :param data: Input Data
        :param label: Training Labels
        :param weight: Label weight loss
        :param certain: easy sample
        :param potential: Difficult samples
        :param patchsz: Patch size
        '''
        self.data=data.transpose(1, 2, 0)
        for i in range(data.shape[0]):
            label[data[i,:,:]<0]=2
        self.data=self.Normalize(self.data)
        label_copy=label.copy()
        if mode=="pre_train":
            label[label == 0] = 2
            label[certain_no_water == 0] = 0
            label[potential_no_water == 0] = 0
            label[certain == 1] = 1
            label[potential == 1] = 1
            label[label_copy == 2] = 2
        label[label_copy == 3] = 0
        self.easy_difficult = len(np.where(certain == 1)[0]) / len(np.where(potential == 1)[0])
        self.label = label

        water_sum = len(np.where(label == 1)[0])
        self.water_rate=water_sum/(np.sum([label_copy!=2]))

        certain_no_water[label==2]=2
        potential_no_water[label==2]=2
        self.weight=weight
        self.patchsz=patchsz
        self.mode=mode
        self.label[data[3,:,:]==-10000]=2
        self.data_copy=self.addMirror(self.data)
        self.label_copy=self.addMirror(self.label)
        self.weight_copy=self.addMirror(self.weight)

        #Set mask, invalid value mask
        self.inviald_mask=np.zeros((self.data.shape[0],self.data.shape[1]))
        self.inviald_mask[self.data[:,:,3]==-10000]=1
        certain[self.inviald_mask==1]=2
        potential[self.inviald_mask==1]=2
        self.patchsz=patchsz

        #Select sample points
        self.certain=list(zip(*np.where(certain==1)))
        self.potential=list(zip(*np.where(potential==1)))
        self.certain_no_water=list(zip(*np.where(certain_no_water==0)))
        self.potential_no_water=list(zip(*np.where(potential_no_water==0)))
        #First select a certain proportion of easy samples and difficult samples
        self.certain=random.sample(self.certain,int(len(self.certain)))
        self.potential=random.sample(self.potential,int(len(self.potential)))

        #easy samples and difficult samples constitute the training data set
        if self.mode=="pre_train":
            if len(self.certain)>=1000:
                self.dataset=random.sample(self.certain,1000)
            elif len(self.certain)!=0:
                self.dataset=self.certain
            if len(self.potential)>=500:
                self.dataset=self.dataset+random.sample(self.potential,500)
            elif len(self.potential)!=0:
                self.dataset=self.dataset+self.potential
            self.dataset=self.dataset+random.sample(self.certain_no_water,100)+random.sample(self.potential_no_water,100)
            water_sum=0
            no_water_sum=0
            for i in range(len(self.dataset)):
                l = self.dataset[i][0]
                c = self.dataset[i][1]
                target_label=self.label_copy[l:l + self.patchsz, c:c + self.patchsz,:]
                water_sum=water_sum+len(np.where(target_label==1)[0])
                no_water_sum = no_water_sum + len(np.where(target_label == 0)[0])
            self.rate=water_sum/no_water_sum

        if self.mode=="train":
            self.dataset=dataset
            water_sum=0
            no_water_sum=0
            for i in range(len(self.dataset)):
                l = self.dataset[i][0]
                c = self.dataset[i][1]
                target_label=self.label_copy[l:l + self.patchsz, c:c + self.patchsz,:]
                water_sum=water_sum+len(np.where(target_label==1)[0])
                no_water_sum = no_water_sum + len(np.where(target_label == 0)[0])
            self.rate=water_sum/no_water_sum

    def Normalize(self, data):
        data=data/10000.
        return data

    def train_transforms(self, image, mask):
        """
        Preprocessing and augmentation on training data (image and label)
        """
        train_transform = A.Compose(
            [
                A.HorizontalFlip(p=0.8),
                A.VerticalFlip(p=0.8),
                A.RandomRotate90(p=0.8),
            ]
        )
        transformed = train_transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]

        return image, mask

    # Add mirroring. Mirror filling is a border filling method because the filling method of edge pixels must be considered during convolution operation.
    def addMirror(self, data):
        dx = self.patchsz // 2
        if len(data.shape)==2:
            data=np.expand_dims(data,axis=2)
        h, w, bands = data.shape
        mirror = None
        if dx != 0:
            mirror = np.zeros((h + 2 * dx, w + 2 * dx, bands))
            mirror[dx:-dx, dx:-dx, :] = data
            for i in range(dx):
                # Fill the upper left part of the mirror
                mirror[:, i, :] = mirror[:, 2 * dx - i, :]
                mirror[i, :, :] = mirror[2 * dx - i, :, :]
                # Fill the lower right part with mirror image
                mirror[:, -i - 1, :] = mirror[:, -(2 * dx - i) - 1, :]
                mirror[-i - 1, :, :] = mirror[-(2 * dx - i) - 1, :, :]
        return mirror

    def __len__(self):
        return len(self.dataset)
    def get_dataset(self):
        return self.dataset
    def get_rate(self):
        return self.rate,self.water_rate,self.easy_difficult

    def __getitem__(self, index):
        l, c = self.dataset[index]
        neighbor_region = self.data_copy[l:l + self.patchsz, c:c + self.patchsz, :]
        target = self.label_copy[l:l + self.patchsz, c:c + self.patchsz,:]
        if self.mode=="pre_train":
            neighbor_region, target = self.train_transforms(neighbor_region, target)
            image = torch.tensor(neighbor_region, dtype=torch.float32)
            mask = torch.tensor(target, dtype=torch.int64)
            weight = self.weight_copy[l:l + self.patchsz, c:c + self.patchsz, :]
            return image, mask,torch.tensor(weight,dtype=torch.float32)

        weight=self.weight_copy[l:l + self.patchsz, c:c + self.patchsz,:]
        return torch.tensor(neighbor_region, dtype=torch.float32),torch.tensor(target, dtype=torch.int64),torch.tensor(weight,dtype=torch.float32)

class my_dataset_test(Dataset):
    def __init__(self, data,label,patchsz=512):
        self.data=data.transpose(1, 2, 0)
        #Set mask, invalid value mask
        self.inviald_mask=np.zeros((self.data.shape[0],self.data.shape[1]))
        self.inviald_mask[self.data[:,:,3]==-10000]=1
        self.label=label
        self.label[self.data[:,:,3]==-10000]=2
        self.label[self.label==255]=2
        self.data=self.Normalize(self.data)
        self.patchsz=patchsz
        self.data=[self.data]
        self.label=[self.label]

    def Normalize(self, data):
        data=data/10000.
        return data

    def addMirror(self, data):
        dx = self.patchsz // 2
        if len(data.shape)==2:
            data=np.expand_dims(data,axis=2)
        h, w, bands = data.shape
        mirror = None
        if dx != 0:
            mirror = np.zeros((h + 2 * dx, w + 2 * dx, bands))
            mirror[dx:-dx, dx:-dx, :] = data
            for i in range(dx):
                mirror[:, i, :] = mirror[:, 2 * dx - i, :]
                mirror[i, :, :] = mirror[2 * dx - i, :, :]
                mirror[:, -i - 1, :] = mirror[:, -(2 * dx - i) - 1, :]
                mirror[-i - 1, :, :] = mirror[-(2 * dx - i) - 1, :, :]
        return mirror

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image=torch.tensor(self.data[index].copy(),dtype=torch.float32)
        mask=torch.tensor(self.label[index].copy(),dtype=torch.float32)
        return image,mask