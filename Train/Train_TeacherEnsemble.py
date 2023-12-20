
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import pandas as pd
from DataManager.DataManager import DatasetManager
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.stats import pearsonr, spearmanr, kendalltau
import os.path as osp
from Model.ViT import DefineModel
from torchvision.models import vit_b_16, ViT_B_16_Weights



def eval_metric(total_pred, total_gt): 
    aggregate_results=dict()
    aggregate_results["plcc"] = abs(pearsonr(total_pred, total_gt)[0])
    aggregate_results["srocc"] = abs(spearmanr(total_pred, total_gt)[0])
    aggregate_results["krocc"] = abs(kendalltau(total_pred, total_gt)[0])
    aggregate_results["overall"] = abs(pearsonr(total_pred, total_gt)[0]) + abs(spearmanr(total_pred, total_gt)[0]) + abs(kendalltau(total_pred, total_gt)[0])
    return aggregate_results["overall"]

class TrainTeacher(): 
    """
    Trains the teacher ensemble
    size = size of input image
    learning_rate = learning rate for optimization 
    num_epochs = number of epochs to train the network
    device = name of the GPU device to train the network
    batch_size = size of the batch for optimization
    directory_images = path to the CT images for training
    json_path = path to the json file with the path to the images and quality score
    conv_stem = number of convolutional blocks in the convolutional stem
    num_classes = number of scores to predict (1 for each teacher ensemble member)
    in_channels = channel dimensions for the input image
    lambda1 = weight for the L1 loss function 
    lambda 2 = weight for the L2 loss function
    n_features = number of features for the first convolutional block in the convolutional stem 
    """
    
    def __init__(self, size: tuple, learning_rate: float, num_epochs: int, device: str, batch_size:int, 
                 directory_images: str, json_path: str, conv_stem: int, num_classes:int=1, in_channels:int=1, 
                 lambda1: int=1, lambda2: int=1, n_features: int=8):
        self.size=size
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.n_features = n_features
        self.num_epochs = num_epochs
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.device = torch.device("cpu" if not torch.cuda.is_available() else device)
        print("Device running the experiments ", self.device)
        self.batch_size=batch_size
        self.directory_images=directory_images
        self.json_path=json_path
        self.conv_stem=conv_stem
        self.loss_funcL2 = nn.MSELoss()
        self.loss_funcL1 = nn.L1Loss()
    
    def initialize_vit_model(self):
        """
        Initialize vit model with pretrained weights from ImageNet
        """ 
        vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        model= DefineModel(vit, vit=True, classes=self.num_classes, in_channels=self.in_channels, n_features=self.n_features, n_conv_stems=self.conv_stem).set_model()
        model.to(self.device)
        return model
    
    def initialize_files_logging(self, fold): 
        """
        Create logging files/folders
        """
        snapshot_dir = './snapshots_fold_'+str(fold)+'/'
        os.makedirs(snapshot_dir, exist_ok=True)
        train_info = pd.DataFrame(columns=['Epoch','overall_perf', 'train_loss','val_overall_perf', 'val_loss'])
        return snapshot_dir, train_info
    
    def initialize_datasets(self, transform, train, batch_size, fold, shuffle): 
        """
        Initialize training/validation datasets
        """
        dataset = DatasetManager(self.directory_images, self.json_path, self.size, transform = transform, train=train, fold=fold)
        loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle )  
        return loader, dataset   
    
    def calculate_loss(self, pred_score, score): 
        """
        L1 and L2 loss function to optimize network
        """
        loss_pred_classL1 = self.loss_funcL1(pred_score, score)
        loss_pred_classL2 = self.loss_funcL2(pred_score, score)
        return (self.lambda1*loss_pred_classL1)+(self.lambda2*loss_pred_classL2)
    
    def forward_pass(self, dataset, model, loss_list, pred_list, gt_list): 
        img, score, _ = dataset
        img, score  = img.to(self.device), score.to(self.device).float()
        pred_score  = model(img)
        total_loss = self.calculate_loss(pred_score, score)
        loss_list.append(total_loss.tolist())
        pred_list.extend(pred_score.flatten().tolist())
        gt_list.extend(score.flatten().tolist())
        return total_loss, loss_list, pred_list, gt_list

    def train_model_fold(self, fold): 
        model = self.initialize_vit_model()     
        snapshot_dir, train_info = self.initialize_files_logging(fold)
        train_loader, source_train = self.initialize_datasets(transform=True, train=True, batch_size= self.batch_size, fold=fold, shuffle=True)
        val_loader, source_val = self.initialize_datasets(transform=False, train=False, batch_size= self.batch_size, fold=fold, shuffle=False)

        print("Number of images on training set ", len(source_train))
        print("Number of images on validation set", len(source_val))

        val_metric=0

        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min = 1e-6)

        for i_iter in range(self.num_epochs):
            train_loss=[]
            val_loss = []
            pred_all_score=[]
            gt_all_score=[]
            val_pred_all_score=[]
            val_gt_all_score=[]

            for _, dataset in enumerate(train_loader):
                model.train()
                optimizer.zero_grad()
                total_loss, train_loss, pred_all_score, gt_all_score= self.forward_pass(dataset, model, train_loss, pred_all_score, gt_all_score)
                total_loss.backward()
                optimizer.step()
            metric_pred_class=eval_metric(pred_all_score, gt_all_score)
            scheduler.step()  

            for _, dataset in enumerate(val_loader):
                model.eval()
                with torch.set_grad_enabled(False):
                    total_loss, val_loss, val_pred_all_score, val_gt_all_score= self.forward_pass(dataset, model, val_loss, val_pred_all_score, val_gt_all_score)
            val_metric_pred_class = eval_metric(val_pred_all_score, val_gt_all_score)

            train_info = pd.concat([train_info, pd.DataFrame({'Epoch': i_iter,'overall_perf': metric_pred_class, 
                                        'train_loss': np.mean(train_loss),'val_overall_perf': val_metric_pred_class ,
                                        'val_loss':  np.mean(val_loss)}, index=[0])], ignore_index=True)
            train_info.to_csv('train_info_fold_'+str(fold)+'.csv')

            
            if val_metric_pred_class>val_metric: 
                val_metric=val_metric_pred_class
                print('taking snapshot ...')
                torch.save(model.state_dict(), osp.join(snapshot_dir, 'model_' + str(i_iter) + '.pth'))
    
    def train_teacher(self):
         for fold in range(1,6): 
             self.train_model_fold(fold)
             







