#!/usr/bin/env python
# coding: utf-8

import os
import re
import pandas as pd
import SimpleITK
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
import numpy as np
import json


def read_image(path): 
    """
    Read the dicom image and returns numpy array
    """
    reader = SimpleITK.ImageFileReader()
    reader.SetFileName(path)
    img = reader.Execute()
    image_data = SimpleITK.GetArrayFromImage(img)
    return image_data


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


def save_dicom_image(np_img, path):
    """
    Saves numpy image as .dcm in path
    """
    img = SimpleITK.GetImageFromArray(np_img[0])
    SimpleITK.WriteImage(img, path)


def open_json(path):
    """
    Open json file with name in path
    """
    with open(path, 'r') as file:
        json_file = json.load(file)
    return json_file


def save_json(file_path, name, data): 
    """
    Saves json file of data with "name" in file path
    """
    with open(file_path+"/"+name, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def save_set_images_(set_prefixes, file_path, json_name, json_dataset): 
    """
    Saves the images that are from the training and testing set in a json file
    set_prexis: list with the cases that are part of the set
    json_name: name for the json file with the training/testing set images
    json_dataset: json with the list of all the images in the dataset
    """
    file_entries = {key: value for key, value in json_dataset.items() if any(key.startswith(prefix) for prefix in set_prefixes)}
    save_json(file_path, json_name, file_entries)
    print("Files in {} saved successfully.".format(json_name))


class ComposeDataset:
    """
    Reads the chest CT images at low and full dose and 
    produces the images at 50% and 75% dose. 
    root_directory= directory where the CT images at full and low dose are saved
    file_extension= file extension of the images to read, expects dicom images .dcm
    """
    def __init__(self, root_directory, file_extension=".dcm"): 
        self.root_directory=root_directory
        self.file_extension=file_extension
        self.list_FullDose=[]
        self.dataset_path={}
    
    def find_files_with_extension(self):
        """
        Saves the path to all files with the file_extension and saves in list
        """
        file_paths = []

        # Walk through the directory tree
        for root, _, files in os.walk(self.root_directory):
            for file in files:
                if file.endswith(self.file_extension):
                    # Construct the full path to the file
                    file_path = os.path.join(root, file)

                    # Add the relative path (without the root_directory) to the list
                    relative_path = os.path.relpath(file_path, self.root_directory)
                    file_paths.append(relative_path)

        return file_paths

    def find_pattern(self,text, pattern):
        """
        Searches for the pattern in the input text
        If it finds, returns the matches string
        If it doesn´t raises an exception 
        """
        match = re.search(pattern, text)

        if match:
            # Extract the matched string
            matched_string = match.group()
            return matched_string
        else:
            raise Exception("The path does not have the expected format")
    
    def return_key_from_path(self, path): 
        """
        Returns the key for each image. 
        The key is tuple with three components
        (case_number, img_number, dose)
        Assumes the path to the images has the same format as the TCIA images
        """
        pattern_patient_id = r'C\d{3}'
        patient=self.find_pattern(path, pattern_patient_id)

        pattern_img_num = r'\d{1}-\d{3}.dcm'
        img_num=self.find_pattern(path, pattern_img_num)

        pattern_dose = r'\b(Full Dose|Low Dose)\b'
        match=self.find_pattern(path, pattern_dose)
        if match=="Full Dose": 
            dose="FullDose" 
            self.list_FullDose.append((patient,img_num,dose))
        else: 
            dose="LowDose"
        return (patient,img_num,dose)
    
    def save_dataset_information(self, dataset_path): 
        """
        Saves quantitative information about each image in the dataset
        dataset_path: a dictionary with keys (case_number, image_number, dose)
        and the value is the path to the image
        """
        img_info = pd.DataFrame(columns=['Pat',"img_num", "dose","x", "y", "dim",
                                "max_pix", "min_pix", "mean_pix", 
                                "std_pix"])
        for key in dataset_path: 
            patient,img_num,dose=key
            path=dataset_path[key]
            img=read_image(os.path.join(self.root_directory, path))
            c,x,y=img.shape
            img_info = img_info.append({'Pat': patient,"img_num": img_num,
            "dose":dose,
             "x": x,"y": y, "dim": c,
              "max_pix": np.max(img), "min_pix": np.min(img), 
            "mean_pix": np.mean(img), "std_pix":np.std(img)
                                       }, ignore_index=True)
        img_info.to_csv('img_info.csv')   
        
    def return_lowdose_fulldose_path(self, fulldose_key, dataset_path): 
        """
        Using the key from the full dose image, returns the path to 
        the full dose image and corresponding low dose image
        If the corresponding low dose image is not found, it returns an error
        """
        
        patient_id, img_num, _ = fulldose_key
        path_full_dose= dataset_path[fulldose_key]
        try: 
            path_low_dose= dataset_path[(patient_id, img_num, "LowDose")]
        except KeyError: 
            raise Exception("Pairing FullDose-LowDose image not found with keys:"+ str((patient_id, img_num, "LowDose")))
        return path_full_dose, path_low_dose
    
    def get_folder_path(self, file_path, subfolder_depth=2):
        """
        Returns the base folder (2 in depth) from the file_path path
        e.g. input= /home/ct_images/case02/fullDose/1-023.dcm , returns= /home/ct_images/case02
        """
        for i in range(subfolder_depth): 
            # Use os.path.dirname to get the directory containing the file
            file_path = os.path.dirname(file_path)
        return file_path

    def simulate_ld(self, I_qd, I_fd, Dose=1.):
        '''
        Taken from : https://github.com/ayaanzhaque/Noise2Quality
        Given: low dose and full dose CT images, required Dose level.
        Return: Images at the Dose level
        '''
        if Dose==1:
            return I_fd
        elif Dose==0.10:
            return I_qd
        else:
            a = np.sqrt(((1/Dose)-1)/3)

            I_noise = I_qd - I_fd

            return (I_fd+(a*I_noise)).astype(np.int32)
        

    def create_directory_dif_dose(self, base_path, name_folder): 
        """
        Creates the directory with the new doses (50%, 75%) at base path
        """
        new_directory_dose = os.path.join(base_path, name_folder)
        if not os.path.exists(new_directory_dose):
            os.makedirs(new_directory_dose)
        return new_directory_dose 
    

    def calculate_SSIM(self, full, low): 
        """
        Applies window leveling to full and low dose CT image and calculate SSIM between both
        """
        full_wl= window_level(350, 40, full)
        low_wl=window_level(350, 40, low)
        SSIM = structural_similarity(full_wl, low_wl, data_range=1, channel_axis=0)
        return SSIM
    
    def create_dataset_dictionary(self):
        """
        Creates a dictionary with all the images
        The key is a tuple with three components (case_number, img_number, dose)
        The value is the path to that image
        """
        output=self.find_files_with_extension()
        print("number of images to read= ", len(output))
        for path in output:
            print(path)
            key=self.return_key_from_path(path)
            self.dataset_path[key]=path
        print("number of key-value pairs in dictionary= ", len(self.dataset_path))
        
    
    def generate_images_lower_dose(self, base_path, dose, low, full, img_num): 
        """
        Generates and saves images at a different dose
        base_path=path where to create the directory to the dose
        dose= dose at which create the new images
        low=image at low dose
        full= image at full dose
        img_num= name/slice of the image to save (e.g. 1-023.dcm)
        """
        new_path_dose = self.create_directory_dif_dose(base_path, str(dose)+"Dose")
        low_img=self.simulate_ld(low, full, Dose=dose)
        save_dicom_image(low_img, os.path.join(new_path_dose, img_num))
        return low_img
    
    def compose_dataset_save_gt(self): 
        """
        Generates images at 25% and 50% dose
        Calculates SSIM for each image
        returns dictionary with the path to each image and SSIM 
        """
        json_path_gt={}
        for case in self.list_FullDose: 
            path_full_dose, path_low_dose= self.return_lowdose_fulldose_path(case, self.dataset_path)
            _,img_num, _ = case

            # path to create folders at 50% and 75% full dose
            base_path=self.get_folder_path(os.path.join(self.root_directory, path_full_dose))

            full =read_image(os.path.join(self.root_directory, path_full_dose))
            low =read_image(os.path.join(self.root_directory, path_low_dose))
            low_50 =self.generate_images_lower_dose(base_path, 0.5, low, full, img_num)
            low_75 =self.generate_images_lower_dose(base_path, 0.75, low, full, img_num)

            SSIM_low=self.calculate_SSIM(full, low)
            SSIM_50=self.calculate_SSIM(full, low_50)
            SSIM_75=self.calculate_SSIM(full, low_75)

            json_path_gt[path_full_dose]=1
            json_path_gt[path_low_dose]=SSIM_low
            json_path_gt[os.path.join(self.get_folder_path(path_full_dose),"0.5Dose")+"/"+str(img_num)]=SSIM_50
            json_path_gt[os.path.join(self.get_folder_path(path_full_dose),"0.75Dose")+"/"+str(img_num)]=SSIM_75    
        return json_path_gt

if __name__ =='__main__':
    root_directory =  "Dataset/LDCT-and-Projection-data"
    file_extension=".dcm"
    train_prefixes = ["C002", "C004", "C016", "C021", "C030", "C050", "C052", "C067", "C081"]
    test_prefixes = ["C012", "C077"]
    
    # Create dataset with images at lower doses
    dt=ComposeDataset(root_directory, file_extension)
    dt.create_dataset_dictionary()
    json_gt=dt.compose_dataset_save_gt()
    assert(len(json_gt)==2*len(dt.dataset_path))
    
    # Save information about whole dataset, training set, and testing set in json
    save_json(root_directory, 'dataset.json', json_gt)
    json_dataset=open_json(root_directory+"/dataset.json")
    save_set_images_(train_prefixes, root_directory, "train.json", json_dataset)
    save_set_images_(test_prefixes, root_directory, "test.json", json_dataset)
    




