#!/usr/bin/env python
# coding: utf-8

# # To Do
# 1. Implement option for single prediction
# 1. Write doc strings for every function
# 1. Exception handling and check for invalid inputs.(most important! requirement of kernel_sizes and layer_activatons have to have the same amount of elements as the number of layers (num_cnn_layers)!

import tensorflow as tf
import numpy as np
import rasterio
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score    
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas
import pycm
from tqdm import tqdm
import sys


class WindturbineDetector():
    
    def __init__(self, S1_selection_windturbine_path=None, S1_selection_no_windturbine_path=None, 
                 S2_selection_windturbine_path=None, S2_selection_no_windturbine_path=None,
                 categories_windturbine_crops=[1], categories_no_windturbine_crops=[1], 
                 pixel="50p", image_bands=["B04", "B08", "VV"], S2_rescale_factor=2**14, S1_value_increment=5, S1_rescale_factor=20, 
                 rotation_range=0, zoom_range=0, width_shift_range=0, height_shift_range=0,
                 horizontal_flip=False, vertical_flip=False, fill_mode="constant", cval=0.0,
                 num_cnn_layers=2, filters=16, kernel_sizes=[5, 5], layer_activations=["relu", "relu"],
                 input_shape=[50, 50, 3], pool_size=2, strides=2, full_connection_units=128, 
                 full_connection_activation="relu", output_units=1, output_activation="sigmoid",
                 optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"], epochs=10, random_state=0,
                 test_size=0.3):
        
            """
            initialize all parameters for the data preparation
            
            Parameters for data import
            ----------
            categories_windturbine_crops: list, [1,2,3]
                Set one or more categories of selection for windturbine selection. 
                Default is [3]
            categories_no_windturbine_crops: list, [1,2]
                Set one or more categories of selection for random crop selection. 
                Default is [2]
            pixel: str, ("10p", "20p", "30p", "40p" or "50p")
                Set one pixel value for the image.
                Default is "30p"
            S1_selection_windturbines_path: pathlib.Path
                Set the path to Sentinel-1 selection_windturbine folder in pathlib.Path format following the folder convention.
            S1_selection_no_windturbines_paths: pathlib.Path
                Set the path to Sentinel-1 selection_no_windturbine folder in pathlib.Path format following the folder convention.
            S2_selection_windturbines_path: pathlib.Path
                Set the path to Sentinel-2 selection_windturbine folder in pathlib.Path format following the folder convention.
            S2_selection_no_windturbines_paths: pathlib.Path
                Set the path to Sentinel-2 selection_no_windturbine folder in pathlib.Path format following the folder convention.
            image_bands: list, ["B04", "B08", "VV"]
                Set the preferred image bands for the image.
                Default is ["B04", "B08", "VV"]
            
            Parameters for data preprocessing
            ----------
            S2_rescale_factor: int, >0
                Set a rescale factor for the image preprocessing in order to get values between 0 and 1.
                Default is 2**14
            S1_value_increment: int
                Increments all values by given number.
                Use this to get all values in a positive range.
                Default is 5.
            S1_rescale_factor: int, >0
                Set a rescale factor for the image preprocessing in order to get values between 0 and 1.
                Default is 20                
            rotation_range: int, 0 to 180
                Set a value to randomly rotate images in the range (degrees, 0 to 180)
                Default is 10
            zoom_range: float or [lower, upper] 
                Set a range for random zoom. If a float, [lower, upper] = [1-zoom_range, 1+zoom_range].
                Default is 0.1
            width_shift_range: float, 1-D array-like or int 
                Set a value in order to shift the image width wise.
                Default is 0.1
            height_shift_range: float, 1-D array-like or int 
                Set a value in order to shift the image height wise.
                Default is 0.1   
            horizontal_flip: Boolean, True or False
                Set a value to randomly flip inputs horizontally.
                Default is False
            vertical_flip: Boolean, True or False
                Set a value to randomly flip inputs vertically.
                Default is False
            fill_mode: str, ("constant", "nearest", "reflect" or "wrap"). 
                Set a mode for the fillmode. Points outside the boundaries of the input are filled according 
                to the given mode: 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
                                   'nearest': aaaaaaaa|abcd|dddddddd
                                   'reflect': abcddcba|abcd|dcbaabcd
                                   'wrap': abcdabcd|abcd|abcdabcd
                Default is "constant"
            cval: Float or Int 
                Value used for points outside the boundaries when fill_mode = "constant".
                Default is 0.0
            
            Parameters for building the convolutional neural network CNN
            ----------
            num_cnn_layers: int, >0
                Set the number of layers for the CNN.
                If changed number of kernel_sizes list has to changed accordingly.
                Default is 2
            filters: int, (16, 32, 64 ...)
                Set the number of filters used for the first layer (this mount is doubled each layer). 
                Default is 16
            kernel_sizes: list of int, [kernel size layer 1, kernel size layer 2, ...]
                Set the kernel size for each layer, first element corresponds to first layer etc..
                The amount of kernel sizes has to be equal to the number of cnn layers.
                Default is [5, 5]
            layer_activations: str, ("relu", "tanh", "sigmoid"... see more on keras.activations)
                Set the activaiton functions for each layer, first element corresponds to first layer etc..
                The amount of activation functions has to be equal to the number of cnn layers.
                Default is ["relu", "relu"]
            input_shape: list, [rows, cols, channels]
                Set the shape of the input image.
                Default is [30, 30, 4]
            pool_size: int, >0
                Set the pool size of max pooling.
                Default is 2
            strides: int, >0
                Set the strides of max pooling.
                Default is 2
            full_connection_units: int, >0
                Set the dimensionality of the full connection outer space.
                Default is 128
            full_connection_activation: str, ("relu", "tanh", "sigmoid"... see more on keras.activations)
                Set the activation function of the full connection (ANN) layer.
                Default is "relu"
            output_units: int, >0
                Set the dimensionality of the output outer space.
                Default is 1
            output_activation: str, ("relu", "tanh", "sigmoid"... see more on keras.activations)
                Set the activation function of the output layer.
                Default is "sigmoid"
                
            Parameters for training the convolutional neural network CNN
            ----------
            optimizer: str, ("adam"... see more on keras.optimizers)
                Set the optimizer function of the cnn compiling stage.
                Default is "adam"
            loss: str, ("binary_crossentropy"... see more on keras.losses)
                Set the loss function of the cnn compiling stage.
                Default is "binary_crossentropy"
            metrics: list of str, (["accuracy"] ... see more on keras.metrics)
                Set the metrics of the cnn compiling stage.
                Default is ["accuracy"]
            epochs: int, >0
                Set the number of epochs to train the model.
                Default is 10
            random_state: int, >0
                Sets the random state for splitting the data in training and test datasets.
            test_size: int, 0 to 1
                Default is 0.2
            """

            if isinstance(S1_selection_windturbine_path, type(None)) or \
               isinstance(S1_selection_no_windturbine_path, type(None)) or \
               isinstance(S2_selection_windturbine_path, type(None)) or \
               isinstance(S2_selection_no_windturbine_path, type(None)):
                print("Please provide Sentinel-1 and Sentinel-2 paths!")
                exit()

            self.categories_windturbine_crops = categories_windturbine_crops
            self.categories_no_windturbine_crops = categories_no_windturbine_crops
            self.pixel = pixel
            self.pixel_num = int(pixel.replace("p", ""))
            self.S1_selection_windturbine_path = S1_selection_windturbine_path
            self.S1_selection_no_windturbine_path = S1_selection_no_windturbine_path
            self.S2_selection_windturbine_path = S2_selection_windturbine_path
            self.S2_selection_no_windturbine_path = S2_selection_no_windturbine_path
            self.image_bands = image_bands
            self.S2_rescale_factor = S2_rescale_factor
            self.S1_value_increment = S1_value_increment
            self.S1_rescale_factor = S1_rescale_factor
            self.rotation_range = rotation_range
            self.zoom_range = zoom_range
            self.width_shift_range = width_shift_range
            self.height_shift_range = height_shift_range
            self.horizontal_flip = horizontal_flip
            self.vertical_flip = vertical_flip
            self.fill_mode = fill_mode
            self.cval = cval
            self.num_cnn_layers = num_cnn_layers
            self.filters = filters
            self.kernel_sizes = kernel_sizes
            self.layer_activations = layer_activations
            self.input_shape = input_shape
            self.pool_size = pool_size
            self.strides = strides
            self.full_connection_units = full_connection_units
            self.full_connection_activation = full_connection_activation
            self.output_units = output_units
            self.output_activation = output_activation
            self.optimizer = optimizer
            self.loss = loss
            self.metrics = metrics
            self.epochs = epochs
            self.random_state = random_state
            self.test_size = test_size
            
            self.cnn = None
            self.indices = []
            self.indices_train = []
            self.indices_test = []
            self.X_train = None
            self.X_test = None
            self.y_train = None
            self.y_test = None
            self.y_pred = None
            self.confusion_matrix = None
            self.accuracy_score = None
            self.pycm = None
            

    def get_images_from_path(self, windturbines, categories, S1_path, S2_path):
        """Expects a pathlib path and windturbine paramter (0 = no windturbine, 1 = windturbine)
        Returns the independent variable four dimensional numpy array with every image bands, categories 
        and pixel shape selected by the user of every crop inside the folders. Also this function returns
        the dependent variable vector (windturbine: Yes/No) corresponding to the independent variable array
        (images).
        
        Parameters
        ----------
        windtubines: int, (1 or 0)
            Set the parameter to either 1 for data with windturbines and 0 without windturbines.
            There is no default!
        categories: list, [1,2,3] or [1,2]
            Set one or more categories of the selection. 
            There is no default!
        S1_path: pathlib.Path
            Set the path to Sentinel-1 image folder in pathlib.Path format following the folder convention.
        S2_path: pathlib.Path
            Set the path to Sentinel-2 image folder in pathlib.Path format following the folder convention.
        
        Returns
        ----------
        X_images: list, 4D array
            Returns a 4D list with images of every folder inside the given path
        y_images: list, 1D array
            Returns a 1D list with 1s (windturbines) or 0s (no windturbines) corresponding to given input
        """
        
        X_images = []
        y_images = []

        # loop through every category inside the selected windturbine crop folder
        for S1_category in S1_path.glob("*"):
            # only select categories and pixel shape selected by the user
            if S1_category.name.count("_") == 3:
                if int(S1_category.name.split("_")[1]) in categories and S1_category.name.split("_")[3] == self.pixel:

                    try:
                        cat = int(S1_category.name.split("_")[1])
                        px = S1_category.name.split("_")[3]
                        S2_category = list(S2_path.glob(f"*_{str(cat)}_*_{str(px)}"))[0]
                    except:
                        continue

                    S1_crop_directories = list(S1_category.glob("*"))

                    for S1_crop in tqdm(S1_crop_directories, desc=f"Scanning crops [{S1_category.name}]:"):
                        if S1_crop.is_dir() and S1_crop.name != "0_combined-preview":

                            if S1_crop.name.count("_") == 1:
                                
                                lon, lat = S1_crop.name.split("_")

                                try:
                                    S2_crop = list(S2_category.glob(f"*_{lon}*_{lat}*"))[0]
                                except:
                                    print(f"Could not find Sentinel-2 crop for Sentinel-1 crop {S1_crop.name}.")
                                    print(f"Used glob: *_{lon}*_{lat}*_*")
                                    print(f"For this s2 crop dir: {S2_category.name}")                                    
                                    continue                                

                            else:

                                lon, lat, pixel, shift = S1_crop.name.split("_")

                                try:
                                    S2_crop = list(S2_category.glob(f"*_{lon}*_{lat}*_*_{pixel}_{shift}"))[0]
                                except:
                                    print(f"Could not find Sentinel-2 crop for Sentinel-1 crop {S1_crop.name}.")
                                    print(f"Used glob: *_{lon}*_{lat}*_*_{pixel}_{shift}")
                                    print(f"For this s2 crop dir: {S2_category.name}")
                                    sys.exit()

                            S1_image_path = S1_crop  

                            S2_image_path = S2_crop / "sensordata" / "R10m"
                            if not S2_image_path.exists():
                                S2_image_path = S2_crop   
                                                        
                            image_list = np.array([])

                            for band in self.image_bands:

                                if band.startswith("V"):

                                    try:
                                        if band == "VV":
                                            image_file = list(S1_image_path.glob("*VV.tif"))[0]
                                        elif band == "VH":
                                            image_file = list(S1_image_path.glob("*VH.tif"))[0]
                                    except:
                                        continue

                                    with rasterio.open(str(image_file.absolute())) as f:
                                        values = f.read(indexes=1) + self.S1_value_increment

                                    values = values * ( 1/self.S1_rescale_factor )  

                                    if image_list.size == 0:
                                        image_list = values
                                    else:
                                        image_list = np.dstack((image_list, values))                                    

                                if band.startswith("B"):

                                    try:
                                        element = list(S2_image_path.glob(f"*_*_{band}_10m.jp2"))[0]
                                    except:
                                        continue

                                    with rasterio.open(str(element)) as f:
                                        values = f.read(indexes=1)

                                    values = values * ( 1/self.S2_rescale_factor )

                                    if image_list.size == 0:
                                        image_list = values
                                    else:
                                        image_list = np.dstack((image_list, values))

                            if image_list.shape == (self.pixel_num, self.pixel_num, len(self.image_bands)):
                                X_images.append(image_list)
                                y_images.append(windturbines)
                                self.indices.append(S2_crop.name.split("_")[0])
                            else:
                                print(f"WARNING: Crop omitted due to wrong shape size S1:{S1_crop.name} S2:{S2_crop.name}")
                                print(f"Shape size: {image_list.shape} instead of {(self.pixel_num, self.pixel_num, len(self.image_bands))}")

                print("Done.")
                print(f"Image-Array contains {len(X_images)} images.")
                print(f"Meta-List contains {len(y_images)} elements [ windturbines:{y_images.count(1)} non-windturbines:{y_images.count(0)} ].")

        
        return X_images, y_images 
    
    
    def create_wt_identification_data(self):
        """Takes in path lists for windturbine and no windturbine image crops, appends every image to an array
        and simultaniously adds a factorial variable to another list which indicates if the image contains a windturbine

        Parameters
        ----------
        selection_windturbines_path: pathlib.Path, pathlib.Path("")
            Set a list of paths to selection_windturbine folders in pathlib.Path format following the folder convention.
            Default is ""
        selection_no_windturbines_paths: pathlib.Path, pathlib.Path("")
            Set a list of paths to selection__no_windturbine folders in pathlib.Path format following the folder convention.
            Default is ""

        Returns
        ----------
        X: list, 4D array
            Returns a 4D list with images of every folder inside the given paths
        y: list, 1D array
            Returns a 1D list with 1s (windturbines) and 0s (no windturbines)
        """
        
        # initialize the independent and dependent variable
        X = []
        y = []
    
        X_images, y_images = self.get_images_from_path(windturbines=1, categories=self.categories_windturbine_crops,
                                                       S1_path=self.S1_selection_windturbine_path, S2_path=self.S2_selection_windturbine_path)
        X.extend(X_images)
        y.extend(y_images)

        X_images, y_images = self.get_images_from_path(windturbines=0, categories=self.categories_no_windturbine_crops,
                                                       S1_path=self.S1_selection_no_windturbine_path, S2_path=self.S2_selection_no_windturbine_path)
        X.extend(X_images)
        y.extend(y_images)
        
        X = np.array(X)
    
        return X, y
        
        
    def preprocess_data(self, X, y):
        
        train_datagen = ImageDataGenerator(rotation_range=self.rotation_range,  # randomly rotate images in the range (degrees, 0 to 180)
                                           zoom_range=self.zoom_range, # Randomly zoom image 
                                           width_shift_range=self.width_shift_range,  # randomly shift images horizontally (fraction of total width)
                                           height_shift_range=self.height_shift_range,  # randomly shift images vertically (fraction of total height)
                                           horizontal_flip=self.horizontal_flip,  # randomly flip images
                                           vertical_flip=self.vertical_flip,  # randomly flip images
                                           fill_mode=self.fill_mode, cval=self.cval)
        dataset = train_datagen.flow(x=X, y=y)
        
        return dataset
    
    
    def build_CNN(self):
        
        # Initialize CNN
        cnn = tf.keras.models.Sequential()
        
        # Step 1: Convolution and pooling of layers
        for i in range(1, self.num_cnn_layers+1):
            if i == 1:
                # First layer of the cnn
                cnn.add(tf.keras.layers.Conv2D(filters=self.filters*i, kernel_size=self.kernel_sizes[0], activation=self.layer_activations[0], input_shape=self.input_shape))
                cnn.add(tf.keras.layers.MaxPool2D(pool_size=self.pool_size, strides=self.strides))
            else:
                # Creation of more layers depending on the num_cnn_layers variable
                cnn.add(tf.keras.layers.Conv2D(filters=self.filters*i, kernel_size=self.kernel_sizes[i-1], activation=self.layer_activations[i-1]))
                cnn.add(tf.keras.layers.MaxPool2D(pool_size=self.pool_size, strides=self.strides))
        
        # Step 2: Flattening
        cnn.add(tf.keras.layers.Flatten())
        
        # Step 3: Full connection
        cnn.add(tf.keras.layers.Dense(units=self.full_connection_units, activation=self.full_connection_activation))
        
        # Step 4: Output Layer
        cnn.add(tf.keras.layers.Dense(units=self.output_units, activation=self.output_activation))
        
        return cnn
    
    
    def train_CNN(self, cnn, training_set, test_set):
        
        # Step 1: compiling the CNN
        cnn.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        
        # Step 2: Training the CNN on the Training set and evaluating it on the Test set
        cnn.fit(x=training_set, validation_data=test_set, epochs=self.epochs)
        
        return cnn
    
    
    def create_confusion_matrix(self):
        
        self.y_pred = self.cnn.predict(self.X_test)
        self.y_pred = self.y_pred.astype(int)
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        ac = accuracy_score(self.y_test, self.y_pred)
        
        return cm, ac
    
    
    def predict_single_observation(self):
        pass
    
    
    def detect_windturbines_with_CNN(self):

        # 1. Import the data:
        print("\nImporting data:")
        print("-----------------\n")
        X, y = self.create_wt_identification_data()

        # 2. Split data into training and test data:
        print("\nSplitting data:")
        print("-----------------\n")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.test_size, random_state = self.random_state)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # 2.1. Randomize and split the indices with the same random_state in order to keep the indices of the crops
        X_train, X_test, indices_train, indices_test = train_test_split(X, self.indices, test_size = self.test_size, random_state = self.random_state)
        self.indices_train = indices_train
        self.indices_test = indices_test

        # 3. Preprocess the data:
        print("\nPreprocessing data:")
        print("-----------------\n")
        training_set = self.preprocess_data(X_train, y_train)
        test_set = self.preprocess_data(X_test, y_test)
        
        # 4. Build the CNN:
        print("\nBuilding the CNN:")
        print("-----------------\n")
        cnn = self.build_CNN()
        
        # 5. Compile, train and evaluate the CNN:
        print("\nCompiling, training and evaluating the CNN:")
        print("-----------------\n")
        cnn = self.train_CNN(cnn, training_set, test_set)
        self.cnn = cnn
        
        # 6. Create confusion matrix and accuracy score:
        print("\nCreate confusion matrix and calculate accuracy score:")
        print("-----------------\n")
        cm, ac = self.create_confusion_matrix()
        self.confusion_matrix = cm
        self.accuracy_score = ac
        
        self.pycm = pycm.ConfusionMatrix(self.y_test, self.y_pred)
        
        print("\nDone! CNN object available through .cnn")


# In[37]:


#test = WindturbineDetector(selection_windturbine_paths=[Path("/data/projects/windturbine-identification-sentinel/croppedTiles/us-uswtdb_selection_windturbines")], 
#                             selection_no_windturbine_paths=[Path("/data/projects/windturbine-identification-sentinel/croppedTiles/selection_no-windturbines")], 
#                             pixel="50p", rotation_range=0, zoom_range=0, width_shift_range=0, height_shift_range=0,
#                             num_cnn_layers=2, filters=32, kernel_sizes=[5, 5], layer_activations=["relu", "relu"],
#                             input_shape=[50, 50, 4], random_state=0)


# In[38]:


#test.detect_windturbines_with_CNN()


# In[39]:


#print(test.confusion_matrix)
#test.pycm.print_matrix()
#accuracy_score(test.y_test, test.y_pred)


# In[ ]:


## Predict if a windturbine is on a located on a specific image
#index = 0
#image_to_predict = np.expand_dims(test.X_test[index], axis = 0)
#result = test.cnn.predict(image_to_predict)

## find image ID
#image_index = test.indices_test[index]

#if result[0][0] == 1:
#    print(f"The image with index {image_index} has a windturbine")
#else:
#    print(f"The image with index {image_index} has NO windturbine")

