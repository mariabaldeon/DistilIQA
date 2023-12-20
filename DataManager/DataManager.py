
import SimpleITK
from os import listdir
from os.path import join
import json
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import albumentations as A
from torch.utils import data
import torch



def read_image(path): 
    reader = SimpleITK.ImageFileReader()
    reader.SetFileName(path)
    img = reader.Execute()
    image_data = SimpleITK.GetArrayFromImage(img)
    return image_data
    

def data_augmentation():
    """
    Operations for data augmentation during training
    """

    aug = A.Compose([A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                 A.Transpose(p=0.5), 
                A.RandomScale(p=0.5),
                A.Resize(512,512)])
    return aug

def window_level(width, level, img): 
    """
    Applies window leveling at level and width for img
    assumes img is in format numpy
    """
    # Calculate the lower and upper bounds for the window
    lower_bound = int(level - width / 2)
    upper_bound = int(level + width / 2)

    # Apply the window leveling to the CT image
    windowed_image = np.clip(img, lower_bound, upper_bound)

    # Normalize the windowed image to 0-255 for display
    windowed_image = ((windowed_image - lower_bound) / width)

    return windowed_image

class DatasetManager(Dataset): 
    """ 
    Class that returns the CT image and quality score
    directory_images= path to the directory with the CT images
    json_path= path to the JSON file with the path and score for each image
    image_size= size of the image, expects a tuple
    transform= bool, if the data augmentation should be applied
    train= bool, if the returned dataset should be for training or validation
    train_ratio= float, ratio of images to be used for training. The rest is used for validation
    fold = fold used for validation, int between 1 and 5. 
    all_imgs = if you use all training images for training. No division for validation. 
    """
    def __init__(self, directory_images: str, json_path: str, image_size: tuple=(512,512), transform: bool= False,
                 train: bool=True, train_ratio: float = 0.80, fold: int =5,
                 all_imgs:bool=False):  
        # Dictionary to save path to image and class
        self.dataset={}
        self.paths=[]
        
        f = open(json_path)
        dict_data = json.load(f)
        size_dataset=len(dict_data)
        validation_cases=size_dataset-int(size_dataset*train_ratio)
        data= list(dict_data.items())
        
        if not all_imgs:
            validation_patients=sorted(data[(fold-1)*validation_cases:fold*validation_cases])
            #print("validation_patients", validation_patients)
            #print("number of validation_patients", len(validation_patients))
            
            if train: 
                data=sorted(list(set(list(dict_data.items())).difference(validation_patients)))
            else:
                data=validation_patients
        
        for key, score in data:
            img_path=os.path.join(directory_images,key)
            self.dataset[img_path]= score   
            self.paths.append(img_path)
            
        #Data augmentation 
        self.transform=transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):

        image_path= self.paths[idx]
        image=read_image(image_path)
        image=image[0].astype('float32')
        image=window_level(350, 40, image)

        if self.transform :
            aug = data_augmentation()
            augmented = aug(image=image)
            image= augmented['image']
        
        #Add channel dimension (C, H, W)
        image=image[None,...]
        image = torch.from_numpy(image.astype(np.float32))
        score= torch.from_numpy(np.array([self.dataset[image_path]]).astype(np.float32))
        
        #Return image and score
        return image, score, image_path
        
if __name__=="__main__": 

    DIRECTORY_IMAGES="/home/mgbaldeon/CTAnalysis/Github/Dataset/LDCT-and-Projection-data"
    JSON_PATH="/home/mgbaldeon/CTAnalysis/Github/Dataset/LDCT-and-Projection-data/train.json"
        
    source_train = DatasetManager(DIRECTORY_IMAGES, JSON_PATH, (512,512), True, False)
    print(len(source_train))
    trainloader = data.DataLoader(source_train, batch_size=3,shuffle=True)

    for i, imgs in enumerate(trainloader):
        images, score = imgs
        images=images.data.cpu().numpy()
        print(score)
        print(images.shape)





