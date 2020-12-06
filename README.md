<p align="center">
    <img src="uc-berkeley-logo-seal.jpg" alt="Logo" width="200" height="200">
</p>

<p align="center">
  <p align="center"><strong>Final Project: Facial Keypoint Detection</strong></p>
  <p align="center"><strong>W207: Applied Machine Learning</strong></p>
  <p align="center"><strong>Jaishankar Raju | Jae Park | Piotr Parkitny</strong></p>
</p>

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About Repository](#about-repository)
* [Presentation](#presentation)
* [EDA](#eda)
* [Preprocessing](#preprocessing)
* [Image Augmentation](#image-augmentation)
* [Model](#model)

<!-- about repository -->
About Repository
------------

This repository is used to host all the code related to the final project along with the link to the final presentation.


<!-- presentation -->
Presentation
------------

The final presentation for this project is hosted on google drive with the link here -> [Link to Presentation](https://docs.google.com/presentation/d/1zQLQ3WyMFHNmQW__SyUghMysKJFmwZ9nAzqLj_FVT9g/edit?usp=sharing)

<!-- eda -->
EDA
------------

- Please use the following link to open the EDA notebook -> [Jupyter Notebook EDA](EDA_preprocessing/proj4_preprocess_func.ipynb)

<!-- preprocessing -->
Preprocessing
------------

- Please use the following link to open the preprocessing notebook -> [Jupyter Notebook Preprocessing](EDA_preprocessing/proj4_eda_preprocessing.ipynb)


<!-- IMAGE AUGMENTATION -->
Image Augmentation
------------

- Image Augmentation is a python application 
- Code is broken down into three sections

1. Main Class ->[Main Class](Image_Augmentation/main.py)
2. Augmentation Class ->[Augmentation Class](Image_Augmentation/Aug_Image.py)
3. Tools ->[Tools](Image_Augmentation/tools.py)

Example of running the application:
    
    python main.py

- Application assumes that the training.csv file is located in the same folder 
- Application will create a file called train_aug.csv that will contain the augmented images

Image selection for the augmentation is based on the flags that are added during EDA/preprocessing.  
The selection is done in this notebook -> [Jupyter Notebook](Image_Augmentation/Image_Augmentation_Selection.ipynb)  

<!-- model -->
Model
------------
Final Model -> [Jupyter Notebook](models/Final_Model.ipynb)  

All the models that were used and tested:
- 1block_2FC -> [Jupyter Notebook](models/1block_2FC.ipynb)
- 2block_2FC -> [Jupyter Notebook](models/2block_2FC.ipynb)
- 3block_2FC -> [Jupyter Notebook](models/3block_2FC.ipynb)
- 4block_2FC -> [Jupyter Notebook](models/4block_2FC.ipynb)
- 5block_2FC -> [Jupyter Notebook](models/5block_2FC.ipynb)
- 6block_2FC -> [Jupyter Notebook](models/6block_2FC.ipynb)
- 6block_2FC_150_epochs -> [Jupyter Notebook](models/6block_2FC_150_epochs.ipynb)
- data_augmented_6block_2FC -> [Jupyter Notebook](models/data_augmented_6block_2FC.ipynb)
- eigenfaces_v3 -> [Jupyter Notebook](models/eigenfaces_v3.ipynb)
- facenet-face-matching -> [Jupyter Notebook](models/facenet-face-matching.ipynb)
