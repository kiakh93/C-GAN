import random
import os
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, lr_transforms=None, hr_transforms=None):
        
        
        
        #inputs
        self.filesI = os.path.join(root, 'LHE')
        self.list_I = os.listdir(self.filesI)
        #targets
        self.filesT = os.path.join(root, 'HE')
        self.list_T = os.listdir(self.filesT)
        #conditions
        self.filesT2 = os.path.join(root, 'Class')
        self.list_T2 = os.listdir(self.filesT2)


        
    def __getitem__(self, index):

        I_name = os.path.join(self.filesI,self.list_I[index%len(self.list_I)])
        im = sio.loadmat(I_name)
        img_lr = im['X']
        
        
        img_lr[np.isnan(img_lr)] = 0
        img_lr[img_lr<0] = 0
        T_name = os.path.join(self.filesT,self.list_T[index%len(self.list_T)])
        im = sio.loadmat(T_name)
        img_hr = im['xx']
        img_hr[np.isnan(img_hr)] = 0
        img_hr[img_hr<0] = 0
        
        #Random channel selection
        CH = random.randint(0,2)
        
        
        RAND = random.uniform(0, 4)
        trans1 = transforms.ToTensor()
        img_lr = np.float32(img_lr)
        img_hr = np.float32(img_hr)
        #Gamma correction
        img_hr = trans1(img_hr)**RAND
        img_lr = trans1(img_lr)**RAND
        img_hr = 1 - img_hr
        img_lr = 1 - img_lr

            
        name = self.list_I[index%len(self.list_I)];

        #conditions
        T2_name = os.path.join(self.filesT2,self.list_T2[index%len(self.list_I)])
        im = sio.loadmat(T2_name)
        img = im['multi']
        trans1 = transforms.ToTensor()
        img = trans1(img)
        seg = img
        
        return {'lr': img_lr[CH,:,:].unsqueeze(0), 'hr': img_hr[CH,:,:].unsqueeze(0), 'name': name, 'seg':seg[1:4,:,:]}
        #return {'lr': img_lr, 'name': name}

    def __len__(self):
        return len(self.list_I)
