import argparse
import numpy as np
from Preprocessing.GenerateImages import ComposeDataset, save_json, open_json, save_set_images_
from Train.Train_TeacherEnsemble import TrainTeacher 
from Train.Train_StudentNetwork import TrainStudent 
from Evaluation.Evaluation import Evaluation

parser = argparse.ArgumentParser(prog="DistilIQA")
parser.add_argument('--task', choices=['preprocessing', 'train_teacher', 'train_student', 'evaluate'], required=True, help='task to do: preprocess LDCT and Projection dataset, train teacher ensemble, train student network (DistilIQA) or evaluate' )
parser.add_argument('--data_preprocess_path', type=str, default='./Dataset/LDCT-and-Projection-data', help='Path to the LDCT and Projection dataset dataset')
parser.add_argument('--img_extension', type=str, default='dcm', help='Image file extension')
parser.add_argument('--train_cases', type=list, default=['C002', 'C004', 'C016', 'C021', 'C030', 'C050', 'C052', 'C067', 'C081'], help='Name of the cases for the training set')
parser.add_argument('--test_cases', type=list, default=['C012', 'C077'], help='Name of the cases for the testing set')
parser.add_argument('--img_size', type=tuple, default=(512, 512), help='Input image size')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--num_epochs_teacher', type=int, default=500, help='Number of epochs to train the teacher network')
parser.add_argument('--num_epochs_student', type=int, default=200, help='Number of epochs to train the student network')
parser.add_argument('--device', type=str, default="cuda:0", help='GPU where to run experiments')
parser.add_argument('--batch_size_teacher', type=int, default=10, help='Batch size to train teacher network')
parser.add_argument('--batch_size_student', type=int, default=25, help='Batch size to train teacher network')
parser.add_argument('--conv_stem', type=int, default=5, help='Number of convolutional blocks in the convolutional stem')
parser.add_argument('--distill_weight', type=float, default=0.75, help='Weight for the distillation term in the loss function')
parser.add_argument('--weights_path', type=list, default=["Train/weights/fold1.pth", "Train/weights/fold2.pth", "Train/weights/fold3.pth", "Train/weights/fold4.pth", "Train/weights/fold5.pth"], help='Path to the weights of each teacher ensemble member')
parser.add_argument('--student_weight_path', type=str, default= "Evaluation/weights/student_199.pth", help='Path to the student network weights for evaluation')
args = parser.parse_args()

if __name__ =="__main__":
    
    if args.task == 'preprocessing': 
        # Create dataset with images at lower doses
        dt=ComposeDataset(args.data_preprocess_path, args.img_extension)
        dt.create_dataset_dictionary()
        json_gt=dt.compose_dataset_save_gt()
        assert(len(json_gt)==2*len(dt.dataset_path))
        
        # Save information about whole dataset, training set, and testing set in json
        save_json(args.data_preprocess_path, 'dataset.json', json_gt)
        json_dataset=open_json(args.data_preprocess_path+"/dataset.json")
        save_set_images_(args.train_cases, args.data_preprocess_path, "train.json", json_dataset)
        save_set_images_(args.test_cases, args.data_preprocess_path, "test.json", json_dataset)

    if args.task == 'train_teacher': 
        teacher_network = TrainTeacher(args.img_size, args.lr, args.num_epochs_teacher, args.device, 
                                       args.batch_size_teacher, args.data_preprocess_path, args.data_preprocess_path+"/train.json", 
                                       args.conv_stem)
        teacher_network.train_teacher()
    
    if args.task == 'train_student': 
        student_network = TrainStudent(args.img_size, args.lr, args.num_epochs_student, args.device, 
                                       args.batch_size_student, args.data_preprocess_path, args.data_preprocess_path+"/train.json", 
                                       args.conv_stem, args.distill_weight, args.weights_path)
        student_network.train_student()
        
    if args.task == 'evaluate': 
        evaluation_object=Evaluation(args.data_preprocess_path, args.data_preprocess_path+"/test.json", args.student_weight_path,
                         args.img_size, args.device, args.conv_stem, 5, True)
        evaluation_object.evaluate_model()
        
        