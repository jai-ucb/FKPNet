#import stat
import os
#import time
import multiprocessing
import subprocess
#from time import sleep
from tools import *
import numpy as np
import pandas as pd


class Aug_Image(multiprocessing.Process):
# ==============================================================
#
# ==============================================================
    def set_config(self,config):
        self.config = config
        
# ==============================================================
#
# ==============================================================
    def run(self):
        self.start_batch()
        
# ==============================================================
#
# ==============================================================
    def start_batch(self):
        
        print ("JOB:BATCH CONFIG | " + str(self.config))
                
        train_data = pd.read_csv(self.config[0])
        train_data = train_data.dropna()
        train_data.reset_index(drop=True,inplace=True)
        
        # Add Hash for each Image as Key
        train_data['unique_id'] = train_data.Image.map(hash)
        
        train_data['Image'] = train_data['Image'].apply(lambda im: np.fromstring(im, sep=' '))
                
        # Add Aug Type
        train_data['aug'] = ''
        
        image_idx = 0
        
        unique_id = np.array(train_data["unique_id"][image_idx])
        aug_image, aug_points = image_augmentation_rotation(train_data,image_idx,15)
        train_data_aug = add_image_points_to_df(train_data,aug_image, aug_points,image_idx,"rotation_15",unique_id)
        
        for i in range(self.config[3],self.config[4],1):
            print ("JOB:Batch %0.0f " %self.config[1] + " Image Number = %0.0f" %i)
            
            image_idx = i            
            
            unique_id = np.array(train_data["unique_id"][image_idx])
            aug_image, aug_points = image_augmentation_rotation(train_data,image_idx,25)
            new_df = add_image_points_to_df(train_data,aug_image, aug_points,image_idx,"rotation_25",unique_id)
            train_data_aug = pd.concat([train_data_aug,new_df]) 
            
            unique_id = np.array(train_data["unique_id"][image_idx])
            aug_image, aug_points = image_augmentation_rotation(train_data,image_idx,20)
            new_df = add_image_points_to_df(train_data,aug_image, aug_points,image_idx,"rotation_20",unique_id)
            train_data_aug = pd.concat([train_data_aug,new_df]) 
            
            unique_id = np.array(train_data["unique_id"][image_idx])
            aug_image, aug_points = image_augmentation_rotation(train_data,image_idx,15)
            new_df = add_image_points_to_df(train_data,aug_image, aug_points,image_idx,"rotation_15",unique_id)
            train_data_aug = pd.concat([train_data_aug,new_df]) 
            
            unique_id = np.array(train_data["unique_id"][image_idx])
            aug_image, aug_points = image_augmentation_rotation(train_data,image_idx,10)
            new_df = add_image_points_to_df(train_data,aug_image, aug_points,image_idx,"rotation_10",unique_id)
            train_data_aug = pd.concat([train_data_aug,new_df]) 
            
            unique_id = np.array(train_data["unique_id"][image_idx])
            aug_image, aug_points = image_augmentation_rotation(train_data,image_idx,-20)
            new_df = add_image_points_to_df(train_data,aug_image, aug_points,image_idx,"rotation_-20",unique_id)
            train_data_aug = pd.concat([train_data_aug,new_df]) 
            
            unique_id = np.array(train_data["unique_id"][image_idx])
            aug_image, aug_points = image_augmentation_rotation(train_data,image_idx,-10)
            new_df = add_image_points_to_df(train_data,aug_image, aug_points,image_idx,"rotation_-10",unique_id)
            train_data_aug = pd.concat([train_data_aug,new_df])
            
            unique_id = np.array(train_data["unique_id"][image_idx])
            aug_image, aug_points = image_augmentation_rotation(train_data,image_idx,-5)
            new_df = add_image_points_to_df(train_data,aug_image, aug_points,image_idx,"rotation_-5",unique_id)
            train_data_aug = pd.concat([train_data_aug,new_df])
            
            unique_id = np.array(train_data["unique_id"][image_idx])
            aug_image, aug_points = image_augmentation_rotation(train_data,image_idx,5)
            new_df = add_image_points_to_df(train_data,aug_image, aug_points,image_idx,"rotation_5",unique_id)
            train_data_aug = pd.concat([train_data_aug,new_df]) 
            
            unique_id = np.array(train_data["unique_id"][image_idx])
            aug_image, aug_points = image_augmentation_mirror(train_data,image_idx)
            new_df = add_image_points_to_df(train_data,aug_image, aug_points,image_idx,"mirror",unique_id)
            train_data_aug = pd.concat([train_data_aug,new_df]) 
            
            unique_id = np.array(train_data["unique_id"][image_idx])
            aug_image, aug_points = image_augmentation_pixelate(train_data,image_idx,.5,.5)
            new_df = add_image_points_to_df(train_data,aug_image, aug_points,image_idx,"pixel.5.5",unique_id)
            train_data_aug = pd.concat([train_data_aug,new_df]) 

            unique_id = np.array(train_data["unique_id"][image_idx])
            aug_image, aug_points = image_augmentation_pixelate(train_data,image_idx,.4,.4)
            new_df = add_image_points_to_df(train_data,aug_image, aug_points,image_idx,"pixel.4.4",unique_id)
            train_data_aug = pd.concat([train_data_aug,new_df]) 
            
            unique_id = np.array(train_data["unique_id"][image_idx])
            aug_image, aug_points = image_augmentation_pixelate(train_data,image_idx,.3,.3)
            new_df = add_image_points_to_df(train_data,aug_image, aug_points,image_idx,"pixel.3.3",unique_id)
            train_data_aug = pd.concat([train_data_aug,new_df]) 
                                                
            unique_id = np.array(train_data["unique_id"][image_idx])
            aug_image, aug_points = image_augmentation_shift(train_data,image_idx,5,-10)
            new_df = add_image_points_to_df(train_data,aug_image, aug_points,image_idx,"shift_5_-10",unique_id)
            train_data_aug = pd.concat([train_data_aug,new_df]) 
            
            unique_id = np.array(train_data["unique_id"][image_idx])
            aug_image, aug_points = image_augmentation_shift(train_data,image_idx,-5,-5)
            new_df = add_image_points_to_df(train_data,aug_image, aug_points,image_idx,"shift_=5_-5",unique_id)
            train_data_aug = pd.concat([train_data_aug,new_df]) 
            
            unique_id = np.array(train_data["unique_id"][image_idx])
            aug_image, aug_points = image_augmentation_shift(train_data,image_idx,5,5)
            new_df = add_image_points_to_df(train_data,aug_image, aug_points,image_idx,"shift_5_5",unique_id)
            train_data_aug = pd.concat([train_data_aug,new_df])             
            
            unique_id = np.array(train_data["unique_id"][image_idx])
            aug_image, aug_points = image_augmentation_shift(train_data,image_idx,-3,-3)
            new_df = add_image_points_to_df(train_data,aug_image, aug_points,image_idx,"shift_-3_-3",unique_id)
            train_data_aug = pd.concat([train_data_aug,new_df])             
            
            unique_id = np.array(train_data["unique_id"][image_idx])
            aug_image, aug_points = image_augmentation_shift(train_data,image_idx,5,0)
            new_df = add_image_points_to_df(train_data,aug_image, aug_points,image_idx,"shift_5_0",unique_id)
            train_data_aug = pd.concat([train_data_aug,new_df])             
                        
            unique_id = np.array(train_data["unique_id"][image_idx])
            aug_image, aug_points = image_augmentation_zoom(train_data,image_idx,1.8,1.8)
            new_df = add_image_points_to_df(train_data,aug_image, aug_points,image_idx,"zoom_1.8_1.8",unique_id)
            train_data_aug = pd.concat([train_data_aug,new_df]) 
            
            unique_id = np.array(train_data["unique_id"][image_idx])
            aug_image, aug_points = image_augmentation_zoom(train_data,image_idx,1.5,1.5)
            new_df = add_image_points_to_df(train_data,aug_image, aug_points,image_idx,"zoom_1.5_1.5",unique_id)
            train_data_aug = pd.concat([train_data_aug,new_df]) 
                        
            unique_id = np.array(train_data["unique_id"][image_idx])
            aug_image, aug_points = image_augmentation_zoom(train_data,image_idx,1.2,1.3)
            new_df = add_image_points_to_df(train_data,aug_image, aug_points,image_idx,"zoom_1.2_1.3",unique_id)
            train_data_aug = pd.concat([train_data_aug,new_df]) 
            
            unique_id = np.array(train_data["unique_id"][image_idx])
            aug_image, aug_points = image_augmentation_zoom(train_data,image_idx,1,.95)
            new_df = add_image_points_to_df(train_data,aug_image, aug_points,image_idx,"zoom_1_.95",unique_id)
            train_data_aug = pd.concat([train_data_aug,new_df]) 

            unique_id = np.array(train_data["unique_id"][image_idx])
            aug_image, aug_points = image_augmentation_zoom(train_data,image_idx,0.95,.95)
            new_df = add_image_points_to_df(train_data,aug_image, aug_points,image_idx,"zoom_0.95_.95",unique_id)
            train_data_aug = pd.concat([train_data_aug,new_df]) 
                                                            
        train_data_aug['Image'] = [' '.join(map(str, l)) for l in train_data_aug['Image']]
                        
        fname='train_data_aug'+str(self.config[1])+'.csv'     
        train_data_aug.to_csv(fname,index=False)

# ==============================================================
#
# ==============================================================
