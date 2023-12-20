
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
from Train.Train_TeacherEnsemble import eval_metric 
from torchvision.models import vit_b_16, ViT_B_16_Weights


class TrainStudent(): 
    """
    Trains student network
    size = size of input image
    learning_rate = learning rate for optimization 
    num_epochs = number of epochs to train the network
    device = name of the GPU device to train the network
    batch_size = size of the batch for optimization
    directory_images = path to the CT images for training
    json_path = path to the json file with the path to the images and quality score
    conv_stem = number of convolutional blocks in the convolutional stem
    distill_weight = weight to give the distillation term in the loss function
    weights = list where each items is the path to a weight of the teacher ensemble
    in_channels = channel dimensions for the input image
    lambda1 = weight for the L1 loss function 
    lambda2 = weight for the L2 loss function
    n_features = number of features for the first convolutional block in the convolutional stem 
    """
    
    def __init__(self, size: tuple, learning_rate: float, num_epochs: int, device: str, batch_size:int, 
                 directory_images: str, json_path: str, conv_stem: int, distill_weight: int, weights: list, in_channels:int=1, 
                 lambda1: int=1, lambda2: int=1, n_features: int=8):
        self.size=size
        self.learning_rate = learning_rate
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
        self.distill_weight = distill_weight
        self.weights = weights
        self.loss_funcL2 = nn.MSELoss()
        self.loss_funcL1 = nn.L1Loss()
    
    def initialize_vit_model(self, num_classes):
        """
        Initialize vit model with pretrained weights from ImageNet
        """ 
        vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        model= DefineModel(vit, vit=True, classes=num_classes, in_channels=self.in_channels, n_features=self.n_features, n_conv_stems=self.conv_stem).set_model()
        model.to(self.device)
        return model
    
    def create_model(self, weights):
        """
        Initialize vit model with weights 
        weights= path to the weights to use to initialize network
        """ 
        vit=vit_b_16()
        model = DefineModel(model=vit, vit=True, classes=1, in_channels=self.in_channels, n_features=self.n_features, n_conv_stems=self.conv_stem ).set_model()
        model.load_state_dict(torch.load(weights, map_location=self.device), strict=True)
        model.to(self.device)
        return model
    
    def create_teacher_network(self): 
        """
        Initializes the teacher network
        """
        teacher=[]
        for t in range(len(self.weights)): 
            teacher.append(self.create_model(self.weights[t]).eval())
        return teacher
    
    def initialize_files_logging(self): 
        """
        Create logging files/folders
        """
        snapshot_dir = './snapshots_student/'
        os.makedirs(snapshot_dir, exist_ok=True)
        train_info = pd.DataFrame(columns=['Epoch','overall_perf', 'loss','teacher_overall_perf'])
        return snapshot_dir, train_info

    def predict_teacher(self, teacher, img):
        """
        returns the prediction of the teacher ensemble as a vector
        shape = (batch, ensemble_predictions)
        """
        with torch.no_grad():
            pred=teacher[0](img)
            for i in range(1,len(teacher)):
                predi=teacher[i](img)
                pred=torch.cat([pred, predi], dim=-1)
        return pred
    
    def initialize_datasets(self, transform, train, batch_size, shuffle, all_imgs, fold=1): 
        """
        Initialize training/validation datasets
        """
        dataset = DatasetManager(self.directory_images, self.json_path, self.size, transform = transform, train=train, fold=fold, all_imgs=all_imgs)
        loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle )  
        return loader, dataset   
    
    def calculate_loss(self, pred_score, score): 
        """
        L1 and L2 loss function to optimize network
        """
        loss_pred_classL1 = self.loss_funcL1(pred_score, score)
        loss_pred_classL2 = self.loss_funcL2(pred_score, score)
        return (self.lambda1*loss_pred_classL1)+(self.lambda2*loss_pred_classL2)
    

    def train_student(self): 
        snapshot_dir, train_info = self.initialize_files_logging()
        # Student network
        model = self.initialize_vit_model(5) 
        # Teacher network 
        teacher=self.create_teacher_network()
        
        train_loader, source_train = self.initialize_datasets(transform=True, train=True, batch_size= self.batch_size, shuffle=True, all_imgs=True) 
        print("Number of images on training set ", len(source_train))

        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min = 1e-6)
        best_metric=0
        
        for i_iter in range(self.num_epochs):
            train_loss=[]
            pred_all_score=[]
            gt_all_score=[]
            teacher_score=[]

            for _, dataset in enumerate(train_loader):
                model.train()
                optimizer.zero_grad()
                img, score, _ = dataset
                img, score= img.to(self.device), score.to(self.device)

                pred_score_vector = model(img)
                pred_score=torch.mean(pred_score_vector, dim=1, keepdim=True)
                
                # Loss ground truth
                total_loss=self.calculate_loss(pred_score, score)
                
                # Prediction from teacher network
                pred_teacher=self.predict_teacher(teacher, img)
                pred_teacher_mean=torch.mean(pred_teacher, dim=1, keepdim=True)        
                
                # If the teacher network has a good prediction, use for training
                if torch.all(torch.abs(score-pred_teacher_mean) <= 0.05):
                    total_loss=+self.distill_weight*(self.calculate_loss(pred_score_vector, pred_teacher))
                    
                # Backpropagate the error in the generator/segmentor
                total_loss.backward()
                optimizer.step()
                
                train_loss.append(total_loss.tolist())
                pred_all_score.extend(pred_score.flatten().tolist())
                gt_all_score.extend(score.flatten().tolist())
                teacher_score.extend(pred_teacher_mean.flatten().tolist())
            metric_pred_score=eval_metric(pred_all_score, gt_all_score)
            metric_teacher=eval_metric(teacher_score, gt_all_score)
            scheduler.step()  

            train_info = pd.concat([train_info, pd.DataFrame({'Epoch': i_iter,'overall_perf': metric_pred_score, 
                                    'loss': np.mean(train_loss), 'teacher_overall_perf':metric_teacher}, index=[0])], ignore_index=True)
            train_info.to_csv('train_info_student.csv')

            
            if metric_pred_score>best_metric: 
                best_metric=metric_pred_score
                print('taking snapshot ...')
                torch.save(model.state_dict(), osp.join(snapshot_dir, 'student_' + str(i_iter) + '.pth'))
             







