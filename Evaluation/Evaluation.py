import numpy as np
import torch
from torch.utils import data
import pandas as pd
from DataManager.DataManager import DatasetManager
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_absolute_error, mean_squared_error
from Model.ViT import DefineModel
from torchvision.models import vit_b_16


class Evaluation:
    """
    Evaluates the image quality prediction using the pearson correlation, 
    spearmen correlation, kendalltau, mean absolute error, and mean square error
    """
    def __init__(self, directory_images: str, path_json: str, path_weights: str, 
              image_size: tuple, device:str, n_conv_stems:int, 
              members: int, distillation: bool, in_channels:int=1, n_features: int=8): 
        self.directory_images=directory_images
        self.path_json= path_json
        self.path_weights= path_weights
        self.image_size= image_size
        self.device=torch.device("cpu" if not torch.cuda.is_available() else device)
        self.n_conv_stems= n_conv_stems
        self.members= members
        self.distillation = distillation
        self.in_channels = in_channels
        self.n_features = n_features
        
        
    def eval_metric(self, total_pred, total_gt): 
        plcc = abs(pearsonr(total_pred, total_gt)[0])
        srocc = abs(spearmanr(total_pred, total_gt)[0])
        krocc = abs(kendalltau(total_pred, total_gt)[0])
        overall = abs(pearsonr(total_pred, total_gt)[0]) + abs(spearmanr(total_pred, total_gt)[0]) + abs(kendalltau(total_pred, total_gt)[0])
        mae = mean_absolute_error(total_gt, total_pred)
        mse = mean_squared_error(total_gt, total_pred)
        return overall, plcc, srocc, krocc, mae, mse
    
    def initialize_model(self): 
        architecture = vit_b_16()
        self.model= DefineModel(model=architecture, vit=True, classes=self.members, in_channels=self.in_channels, n_features=self.n_features, n_conv_stems=self.n_conv_stems).set_model()
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(self.path_weights, map_location=self.device), strict=True)
        self.model.eval()
    
    def set_dataset(self): 
        test_dataset= DatasetManager(self.directory_images, self.path_json, self.image_size, transform= False, train=False, all_imgs=True)
        self.test_loader = data.DataLoader(test_dataset, batch_size=1)
        print("number of images to test: ", len(self.test_loader))
    
    def predict_score(self):
        eval_info = pd.DataFrame(columns=['patient','true', 'pred'])
        with torch.no_grad():
            self.true=[]
            self.pred=[]
            for _, dataset in enumerate(self.test_loader):
                img, score, image_path = dataset
                img, score  = img.to(self.device), score.to(self.device).float()
                pred_score  = self.model(img)
                if self.distillation: 
                    pred_score = torch.mean(pred_score)
                self.true.append(score.flatten().tolist())
                self.pred.append(pred_score.flatten().tolist())
                print(score.flatten().tolist())
                print(pred_score.flatten().tolist())
                eval_info = pd.concat([eval_info, pd.DataFrame({'patient': image_path,'true':score.flatten().tolist(), 'pred': pred_score.flatten().tolist()}, index=[0])], ignore_index=True)
                eval_info.to_csv('eval_info.csv')
    
    def calculate_eval_metrics(self): 
        overall, plcc, srocc, krocc, mae, mse=self.eval_metric(np.squeeze(self.pred), np.squeeze(self.true))
        print(overall, plcc, srocc, krocc, mae, mse)
        accumulate_metrics = pd.DataFrame({"overall ": overall,
                                         "plcc": plcc,
                                         "srocc": srocc, 
                                           "krocc": krocc, 
                                           "mae": mae, 
                                           "mse": mse}, index=[0])
        accumulate_metrics.to_csv('accumulate_metrics.csv')
    
    def evaluate_model(self):
        self.initialize_model()
        self.set_dataset()
        self.predict_score()
        self.calculate_eval_metrics()


if __name__=="__main__": 
    size=(512, 512)
    path_weight = 'weights/ViT_199.pth'
    device="cuda:0"
    device = torch.device("cpu" if not torch.cuda.is_available() else device)
    print(device)
    DIRECTORY_IMAGES="/home/mgbaldeon/CTAnalysis/LDCT-and-Projection-data"
    JSON_PATH="/home/mgbaldeon/CTAnalysis/LDCT-and-Projection-data/test.json"
    #  model name, name for class, number of conv stems at the beggining
    n_conv_stems=5
    # The number of predictions in the model (ie: if ConvViT 1, Distill ConvViT 5)
    members = 5
    distillation=True
    evaluation_object=Evaluation(directory_images=DIRECTORY_IMAGES, 
                                path_json= JSON_PATH, 
                                path_weights=path_weight, 
                                image_size= size, 
                                device="cpu", 
                                n_conv_stems=n_conv_stems, 
                                members= members, 
                                distillation= distillation)
    evaluation_object.evaluate_model()

