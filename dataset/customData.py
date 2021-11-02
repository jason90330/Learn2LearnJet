'''
For Jets artifact data
'''
from torch.utils.data import Dataset
from imutils import paths
from PIL import Image 
import numpy as np
import torch
import random
import os

def pil_loader(path):    # 一般採用pil_loader函式。
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def load_mat(path):
    mat = np.loadtxt(path)
    return torch.tensor(mat,dtype=torch.float32)

class customData(Dataset):
    def __init__(self, img_path, txt_path, phase = '',data_transforms=None, loader = default_loader):
        # with open(txt_path) as input_file:
            # foldLists = input_file.readlines()
        img_paths = list(paths.list_images(img_path))
        random.Random(4).shuffle(img_paths)
        # self.img_name = [os.path.join(img_path, (line.strip().split(' ')[0][11:])) for line in lines]
        self.img_name = []
        self.img_label = []
        lengthOfTrain = int(len(img_paths)*0.8)
        artifact_types =   { 
                    'C_0201': 1,
                    'C_0402': 2,
                    'C_0402-P': 3,
                    'C_0603': 4,
                    'C_0603_LW1P8X1P0_H1P0': 5,
                    'C_0603-P': 6,
                    }
        if phase=='train':
            for path in img_paths[:lengthOfTrain]:#不能先吃
                trainFolder = path.strip("\n")
                if "OK" in path:
                    name = path.split('/')[-3]#只保留類別名稱
                    if name in artifact_types:
                        label = artifact_types[name]
                        self.img_name.append(path)
                        self.img_label.append(label)
                        continue
                else:
                    # print(path)
                    continue                        
        '''
        elif phase=="test":
            for path in img_paths:
                try:
                    for idx, line in enumerate(foldLists):
                        trainFolder = line.strip("\n")
                        if "Live" in path and "Train" not in path:
                            self.img_name.append(path)
                            self.img_label.append(0)
                            break                            
                        elif trainFolder in path or "Train" in path:
                            break # 任何一個 train folder 有出現在 path 當中，代表不是 test data
                        elif idx == len(foldLists)-1: # 所有 train folder 都沒出現在 path 當中，代表是 test data
                            name = path.split('/')[-3]#只保留類別名稱
                            self.img_name.append(path)
                            label = spoof_types[name]
                            self.img_label.append(label)
                            break
                        else:
                            continue
                except:
                    print(path)
        '''                          

        # self.img_label = [int(line.strip().split(' ')[-1]) for line in lines]
        self.data_transforms = data_transforms
        self.phase = phase
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        try:
            img_name = self.img_name[item]
            label = self.img_label[item]
            img = self.loader(img_name)

            if self.data_transforms is not None:
                try:
                    img = self.data_transforms(img)
                except:
                    print("Cannot transform image: {}".format(img_name))
            return img, label
        except:
            print(item)