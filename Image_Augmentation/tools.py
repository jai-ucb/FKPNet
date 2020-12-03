import numpy as np
import pandas as pd
import cv2
from math import sin, cos, pi

# ==================================================================================
# INPUT:
# df              = data frame with feature points and image data
# image_idx       = index in the dataframe of the image to use
# aug_type_detail = degrees for rotation
# show_image_flag = T to show image
#
# OUTPUT:
# rot_image       = rotated image
# rot_points      = all feature points rotated
# ==================================================================================
def image_augmentation_rotation(df,image_idx,aug_type_detail):    
        
    image = df["Image"][image_idx]
    image = image.reshape(96,96)               
    feature_points = df.iloc[image_idx, 0:30].tolist()
    aug_points = []
    
    #Create Rotation Matrix And Rotate Image
    rot_matrix = cv2.getRotationMatrix2D((48,48),aug_type_detail,1.0)
    rot_image = cv2.warpAffine(image,rot_matrix,(96,96))
    ox = 48
    oy = 48
    angle = -aug_type_detail * pi/180
        
    #Rotate all feature points
    for i in range(0,30,2):
    
        px = feature_points[i]
        py = feature_points[i+1]
    
        x = ox + cos(angle) * (px - ox) - sin(angle) * (py - oy)
        y = oy + sin(angle) * (px - ox) + cos(angle) * (py - oy)
        
        aug_points.append(x)    
        aug_points.append(y)
                        
    return rot_image,aug_points
# ==================================================================================
# INPUT:
# ==================================================================================
def add_image_points_to_df(df_source, aug_image,aug_points,image_idx,aug_type,unique_id):
    
    #Convert aug_points into DataFrame
    df = pd.DataFrame(aug_points)
    df_points = df.T
        
    #Convert aug_image into DataFrame    
    aug_image = aug_image.flatten()
    stringList = [] 
    
    aug_image = [int(i) for i in aug_image] 
    
    for x in aug_image: 
            stringList.append(str(x)) 
    
    df_image = pd.DataFrame(stringList)        
    df_image = df_image.T
    
    df_image['Image2'] = df_image.apply(','.join, axis=1)
    df_image["Image"] = df_image['Image2'].str.split(',')
    df_image_only = df_image[['Image']]
        
    return_df = pd.merge(df_points,df_image_only,left_index=True,right_index=True)
    return_df['unique_id'] = unique_id
    return_df['aug'] = aug_type
    
    col_names = df_source.columns.values.tolist()
    return_df.columns = col_names
                    
    return return_df
# ==================================================================================
# INPUT:
# df              = data frame with feature points and image data
# image_idx       = index in the dataframe of the image to use
# show_image_flag = T to show image
#
# OUTPUT:
# aug_image       = mirror image
# aug_points      = all feature points mirrored
# ==================================================================================
def image_augmentation_mirror(df,image_idx):    
    
    image = np.array(df["Image"][image_idx])
    image = image.reshape(96,96)
    feature_points = df.iloc[image_idx, 0:30].tolist()
    aug_points = []
    
    #Mirror Image
    aug_image = np.flip(image,axis=1)
       
    #Rotate all feature points
    for i in range(0,30,2):
        
        x = 96 - feature_points[i]
        y = feature_points[i+1]
        aug_points.append(x)    
        aug_points.append(y)    
                                           
    return aug_image,aug_points
# ==================================================================================
# INPUT:
# df              = data frame with feature points and image data
# image_idx       = index in the dataframe of the image to use
# x_zoom          = zoom factor
# y_zoom          = zoom factor
#
# OUTPUT:
# aug_image       = aug image
# aug_points      = all feature points 
# ==================================================================================
def image_augmentation_pixelate(df,image_idx,x_zoom,y_zoom):    
    
    image = np.array(df["Image"][image_idx])
    image = image.reshape(96,96)
    feature_points = df.iloc[image_idx, 0:30].tolist()
    aug_points = []
    
    h = int(x_zoom * 96)
    w = int(y_zoom * 96)
    
    #Aug Image
    zoom_image = cv2.resize(image,(w, h), interpolation = cv2.INTER_CUBIC)
    aug_image = cv2.resize(zoom_image,(96,96), interpolation = cv2.INTER_CUBIC)
       
    #Rotate all feature points
    for i in range(0,30,2):
        
        x = feature_points[i]
        y = feature_points[i+1]
        aug_points.append(x)    
        aug_points.append(y)    
                                           
    return aug_image,aug_points
# ==================================================================================
# INPUT:
# df              = data frame with feature points and image data
# image_idx       = index in the dataframe of the image to use
# shift           = how much to shift image in x direction
#
# OUTPUT:
# aug_image       = mirror image
# aug_points      = all feature points mirrored
# ==================================================================================
def image_augmentation_shift(df,image_idx,x_shift,y_shift):    
    
    image = np.array(df["Image"][image_idx])
    image = image.reshape(96,96)
    feature_points = df.iloc[image_idx, 0:30].tolist()
    aug_points = []
    
    #Aug Image
    M = np.float32([[1,0,x_shift],[0,1,y_shift]])            
    aug_image = cv2.warpAffine(image,M,(96,96))
       
    #Rotate all feature points
    for i in range(0,30,2):
        
        x = feature_points[i] + x_shift
        y = feature_points[i+1] + y_shift
        aug_points.append(x)    
        aug_points.append(y)    
                                           
    return aug_image,aug_points
# ==================================================================================
# INPUT:
# df              = data frame with feature points and image data
# image_idx       = index in the dataframe of the image to use
# shift           = how much to shift image in x direction
#
# OUTPUT:
# aug_image       = mirror image
# aug_points      = all feature points mirrored
# ==================================================================================
def image_augmentation_zoom(df,image_idx,x_zoom,y_zoom):    
    
    image = np.array(df["Image"][image_idx])
    image = image.reshape(96,96)
    feature_points = df.iloc[image_idx, 0:30].tolist()
    aug_points = []
    
    #Aug Image
    rows = 96
    cols = 96
    ch = 1

    scale_x = int(96 / x_zoom)
    scale_y = int(96 / y_zoom)
    
    pts1 = np.float32([[0,0],[96,0],[0,96],[96,96]])
    pts2 = np.float32([[0,0],[scale_x,0],[0,scale_y],[scale_x,scale_y]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    aug_image = cv2.warpPerspective(image,M,(96,96))
    
    #Rotate all feature points
    for i in range(0,30,2):
        
        x = feature_points[i] /x_zoom
        y = feature_points[i+1] /y_zoom
        aug_points.append(x)    
        aug_points.append(y)    
                                           
    return aug_image,aug_points
