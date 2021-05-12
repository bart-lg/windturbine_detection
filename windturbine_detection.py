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
import csv
from tqdm import tqdm
import time
from itertools import zip_longest


def reduce_coordinate_digits(lon, lat):
    coordinateDecimalsForComparison = 4
    lon = lon[ : ( lon.find(".") + coordinateDecimalsForComparison + 1 ) ]
    lat = lat[ : ( lat.find(".") + coordinateDecimalsForComparison + 1 ) ]
    return lon, lat

class WindturbineDetector():
    
    def __init__(self, selection_windturbine_paths=[""], selection_no_windturbine_paths=[""], s1_crops_path1=None,
                 s1_crops_path2=None, s1_only=False, s1_dim_reduced=False, categories_windturbine_crops=[3], 
                 categories_no_windturbine_crops=[2], pixel="30p", image_bands=["B02", "B03", "B04", "B08"], 
                 coordinate_filter_csv=None, no_windturbine_limit=None,
                 rescale_factor=2**14, s1_scale_shift=50, s1_rescale_factor=100,
                 rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1,
                 horizontal_flip=False, vertical_flip=False, fill_mode="constant", cval=0.0,
                 num_cnn_layers=2, filters=16, kernel_sizes=[5, 5], layer_activations=["relu", "relu"],
                 input_shape=[30, 30, 4], pool_size=2, strides=2, full_connection_units=128, 
                 full_connection_activation="relu", output_units=1, output_activation="sigmoid",
                 optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"], epochs=10, random_state=0,
                 test_size=0.2, export_path="", export_name="", csv_confusion_matrix=""):
        
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
            s1_crops_path: str
                Path for Sentinel-1 crops
                Use this if you want to combine Sentinel-1 and Sentinel-2 crops.
                If you want to use Sentinel-1 images only use the selection_windturbines_paths and selection_no_windturbines_paths
                and set s1_only to true.
            s1_only: boolean
                Set this to true if only Sentinel-1 images should be used.
                In this case set selection_windturbines_paths and selection_no_windturbines_paths to the Sentinel-1 paths and this flag to True.
                Default is False
            s1_dim_reduced: boolean
                Set this flag to True if the S1 images got reduced in dimensions.
                The script expects <lon>_<lat> as format for crop directories and the files
                s1_dim_reduced_VV.tif and/or s1_dim_reduced_VH.tif within the crop directories if this flag is set to True.
                Default is False 
            pixel: str, ("10p", "20p", "30p", "40p" or "50p")
                Set one pixel value for the image.
                Default is "30p"
            selection_windturbines_paths: pathlib.Path, pathlib.Path("")
                Set a list of paths to selection_windturbine folders in pathlib.Path format following the folder convention.
                Default is ""
            selection_no_windturbines_paths: pathlib.Path, pathlib.Path("")
                Set a list of paths to selection__no_windturbine folders in pathlib.Path format following the folder convention.
                Default is ""
            image_bands: list, ["B02", "B03", "B04", "B08"]
                Set the preferred image bands for the image.
                Default is ["B02", "B03", "B04", "B08"]
            coordinate_filter_csv: str
                Path to csv file containing coordinates in lat/lon format with "lat" and "lon" headers.
                Only coordinates contained in csv are used for training and testing.
                For the comparison of the coordinates the coordinate values are reduced to a precision of 4 digits after comma.
                Default is None
            no_windturbine_limit: int
            	Limits the number of used non-windturbine images.
            	Default is None (use all images)
            
            Parameters for data preprocessing
            ----------
            rescale_factor: int, >0
                Set a rescale factor for the image preprocessing of Sentinel-2 crops in order to get values between 0 and 1.
                Default is 2**14
            s1_scale_shift: int
                Shifts the Sentinel-1 values before rescaling.
                Default is 50
            s1_rescale_factor: int, >0
                Set a rescale factor for the image preprocessing of Sentinel-1 crops in order to get values between 0 and 1.
                Default is 100
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

            Parameters for export
            ----------
            export_path: str
                Path were the results should be exported to.
            export_name: str
                Descriptive name which is used for the confusion matrix csv and the image ids csv.
            csv_confusion_matrix: str

            """

            self.categories_windturbine_crops = categories_windturbine_crops
            self.categories_no_windturbine_crops = categories_no_windturbine_crops
            self.s1_crops_path1 = s1_crops_path1
            self.s1_crops_path2 = s1_crops_path2
            self.s1_only = s1_only
            self.s1_dim_reduced = s1_dim_reduced
            self.pixel = pixel
            self.pixel_num = int(pixel.replace("p", ""))
            self.selection_windturbine_paths = selection_windturbine_paths
            self.selection_no_windturbine_paths = selection_no_windturbine_paths
            self.image_bands = image_bands
            if isinstance(coordinate_filter_csv, type(None)):
                self.coordinate_filter_csv = None
            else:
                self.coordinate_filter_csv = Path(coordinate_filter_csv)
            self.no_windturbine_limit = no_windturbine_limit
            self.rescale_factor = rescale_factor
            self.s1_scale_shift = s1_scale_shift
            self.s1_rescale_factor = s1_rescale_factor
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
            self.export_path = Path(export_path)
            self.export_name = export_name
            self.csv_confusion_matrix = csv_confusion_matrix
            
            self.cnn = None
            self.indices = []
            self.indices_train = []
            self.indices_test = []
            self.X = None
            self.y = None
            self.X_train = None
            self.X_test = None
            self.y_train = None
            self.y_test = None
            self.y_pred = None
            self.confusion_matrix = None
            self.accuracy_score = None
            self.pycm = None
            

    def get_images_from_path(self, windturbines, categories, path=""):
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
        path: pathlib.Path, pathlib.Path("")
            Set a list of paths to sentinal image folders in pathlib.Path format following the folder convention.
            Default is ""
        
        Returns
        ----------
        X_images: list, 4D array
            Returns a 4D list with images of every folder inside the given path
        y_images: list, 1D array
            Returns a 1D list with 1s (windturbines) or 0s (no windturbines) corresponding to given input
        """
        
        X_images = []
        y_images = []

        s1_subfolder1 = None
        if not isinstance(self.s1_crops_path1, type(None)) and self.s1_crops_path1.is_dir():
            s1_subfolder1_list = list(self.s1_crops_path1.glob(f"{self.pixel}*"))
            if s1_subfolder1_list != None and s1_subfolder1_list[0].is_dir():
                s1_subfolder1 = s1_subfolder1_list[0]

        s1_subfolder2 = None
        if not isinstance(self.s1_crops_path1, type(None)) and self.s1_crops_path2.is_dir():
            s1_subfolder2_list = list(self.s1_crops_path2.glob(f"{self.pixel}*"))
            if s1_subfolder2_list != None and s1_subfolder2_list[0].is_dir():
                s1_subfolder2 = s1_subfolder2_list[0]    

        coordinates = None
        if not isinstance(self.coordinate_filter_csv, type(None)):
            if self.coordinate_filter_csv.exists():
                col_list = ["lon", "lat"]
                coordinates = pandas.read_csv(self.coordinate_filter_csv.absolute(), usecols=col_list, dtype=str)

        # loop through every category inside the selected windturbine crop folder
        for category in path.glob("*"):
            # only select categories and pixel shape selected by the user
            if category.name.count("_") == 3:
                if int(category.name.split("_")[1]) in categories and category.name.split("_")[3] == self.pixel:
                    crop_directories = list(category.glob("*"))
                    for crop in tqdm(crop_directories, desc="Scanning crops"):
                        if crop.is_dir() and not crop.name.startswith("0_"):

                            # check coordinates (windturbines only)
                            if windturbines == 1 and isinstance(coordinates, pandas.core.frame.DataFrame) and len(coordinates) > 0:
                                include = False
                                crop_components = crop.name.split("_")
                                if len(crop_components) > 2:
                                    crop_lon = str(crop_components[1])
                                    crop_lat = str(crop_components[2])
                                else:
                                    crop_lon = str(crop_components[0])
                                    crop_lat = str(crop_components[1])
                                crop_lon, crop_lat = reduce_coordinate_digits(crop_lon, crop_lat)
                                for i in range(len(coordinates)):
                                    lon = str(coordinates["lon"][i])
                                    lat = str(coordinates["lat"][i]) 
                                    lon, lat = reduce_coordinate_digits(lon, lat)
                                    if ( lon in crop_lon or crop_lon in lon ) and ( lat in crop_lat or crop_lat in lat ):
                                        include = True
                                        break
                            else:
                                include = True

                        	# limit non-windturbine images
                        	if windturbines == 0 and not isinstance(self.no_windturbine_limit, type(None)) and \
                        	   len(X_images) > self.no_windturbine_limit:
                        		include = False

                            if include == True and ( "VV" in self.image_bands or "VH" in self.image_bands ):

                                if not self.s1_only:
                                    s1_crop1 = self.getCropDirByCoordinates(crop.name.split("_")[1], 
                                            crop.name.split("_")[2], s1_subfolder1)
                                    s1_crop2 = self.getCropDirByCoordinates(crop.name.split("_")[1], 
                                            crop.name.split("_")[2], s1_subfolder2)
                                    if s1_crop1 == None and s1_crop2 == None:
                                        include = False                                

                            # if include != True:
                                # print(f"Not including: {crop.name.split('_')[0]}")

                            if include == True:

                                image_list = np.array([])

                                for band in self.image_bands:

                                    # add Sentinel-2 band
                                    if band in ("B02", "B03", "B04", "B08"):

                                        image_path = crop / "sensordata" / "R10m"
                                        if not image_path.exists():
                                            image_path = crop

                                        filenames = list(image_path.glob(f"*_*_{band}_10m.jp2"))
                                        if len(filenames) > 0:
                                            file = filenames[0]
                                            with rasterio.open(str(file)) as f:
                                                data = f.read(indexes=1)
                                                # rescale
                                                data = data / self.rescale_factor
                                                if image_list.size == 0:
                                                    image_list = data
                                                else:
                                                    image_list = np.dstack((image_list, data)) 
                                        else:
                                            print(f"SENTINEL-2: No files found for crop {crop.name}")                                           


                                    # add Sentinel-1 band
                                    # VH is band 1, VV is band 2
                                    if ( self.s1_only or s1_subfolder1 != None or s1_subfolder2 != None) and band in ("VH", "VV"):

                                        if self.s1_only:
                                            image_path = crop
                                        else:
                                            image_path = self.getCropDirByCoordinates(crop.name.split("_")[1], 
                                                crop.name.split("_")[2], s1_subfolder1)

                                            if image_path == None:
                                                image_path = self.getCropDirByCoordinates(crop.name.split("_")[1], 
                                                crop.name.split("_")[2], s1_subfolder2)

                                        if image_path != None:

                                            if self.s1_dim_reduced:
                                                if band == "VH":
                                                    file = image_path / "s1_dim_reduced_VH.tif"
                                                    if not file.exists():
                                                        file = image_path / "s1_shifted_VH.tif"
                                                if band == "VV":
                                                    file = image_path / "s1_dim_reduced_VV.tif"
                                                    if not file.exists():
                                                        file = image_path / "s1_shifted_VV.tif"                                                    
                                            else:
                                                file = image_path / "sensordata" / "s1_cropped.tif"

                                            if file.exists():
                                                with rasterio.open(str(file)) as f:
                                                    if self.s1_dim_reduced:
                                                        data = f.read(indexes=1)
                                                    else:
                                                        if band == "VH":
                                                            data = f.read(indexes=1)
                                                        if band == "VV":
                                                            data = f.read(indexes=2)
                                                    # rescale
                                                    data = data + self.s1_scale_shift
                                                    data = data / self.s1_rescale_factor
                                                    if image_list.size == 0:
                                                        image_list = data
                                                    else:
                                                        image_list = np.dstack((image_list, data))                                          
                                            else:
                                                print(f"ERROR: Could not find s1 image for {crop.name}")

                                if image_list.shape == (self.pixel_num, self.pixel_num, len(self.image_bands)):
                                    X_images.append(image_list)
                                    y_images.append(windturbines)
                                    self.indices.append(crop.name.split("_")[0])
                                else:
                                    print(f"WARNING: Crop omitted due to wrong shape size {crop.name}")

                    print("Done.")
                    print(f"Image-Array contains {len(X_images)} images.")
                    print(f"Meta-List contains {len(y_images)} elements [ windturbines:{y_images.count(1)} non-windturbines:{y_images.count(0)} ].")
        
        return X_images, y_images 


    def getCropDirByCoordinates(self, lon, lat, path):

        result = None
        
        if self.s1_dim_reduced:
            folders = list(path.glob(f"{lon}_{lat}"))
        else:
            folders = list(path.glob(f"*_{lon}_{lat}_*"))

        for folder in folders:
            if folder.is_dir():
                result = folder
                break

        return result

    
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
        
        for path in self.selection_windturbine_paths:
            print("Get images for windturbine paths...")
            X_images, y_images = self.get_images_from_path(windturbines=1, categories=self.categories_windturbine_crops,
                                                           path=path)
            X.extend(X_images)
            y.extend(y_images)

        for path in self.selection_no_windturbine_paths:
            print("Get images for non-windturbine paths...")
            X_images, y_images = self.get_images_from_path(windturbines=0, categories=self.categories_no_windturbine_crops,
                                                           path=path)
            X.extend(X_images)
            y.extend(y_images)
        
        X = np.array(X)
        print(f"Total shape of X_images: {X.shape}")
        print(f"Total shape of y_images: {np.array(y).shape}")
    
        return X, y
        
        
    def preprocess_data(self, X, y):
        
        train_datagen = ImageDataGenerator(rescale = None,
                                           rotation_range=self.rotation_range,  # randomly rotate images in the range (degrees, 0 to 180)
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


    def import_data(self):
        print("\nImporting data:")
        print("-----------------\n")
        if isinstance(self.X, type(None)) and isinstance(self.y, type(None)):
            self.X, self.y = self.create_wt_identification_data()
        else:
            print("Data already loaded!\n")        
    
    
    def detect_windturbines_with_CNN(self, random_state=None):

        if not isinstance(random_state, type(None)):
            self.random_state = random_state

        # 1. Import the data:
        self.import_data()

        # 2. Split data into training and test data:
        print("\nSplitting data:")
        print("-----------------\n")
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = self.test_size, random_state = self.random_state)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # 2.1. Randomize and split the indices with the same random_state in order to keep the indices of the crops
        X_train, X_test, indices_train, indices_test = train_test_split(self.X, self.indices, test_size = self.test_size, random_state = self.random_state)
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

        # 7. Export results
        print("\nExport results to csv files:")
        print("-----------------\n")
        self.exportResults()

        print("\nDone!")


    def exportResults(self):

        csv_cm_path = self.export_path / self.csv_confusion_matrix

        # create csv if non-existent

        if not csv_cm_path.is_file():
            with open(str(csv_cm_path), 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=",", quotechar="'")
                csvwriter.writerow([
                    'ID', 
                    'name', 
                    'accuracy', 
                    'Predict0_Actual0', 
                    'Predict1_Actual0', 
                    'Predict0_Actual1', 
                    'Predict1_Actual1'
                ])
            new_id = 1
        else:
            # determine new index
            with open(str(csv_cm_path), 'r') as csvfile:
                data_reversed = list(reversed(list(csv.reader(csvfile))))
                new_id = data_reversed[0][0]

        # append confusion matrix and accuracy to csv

        with open(str(csv_cm_path), 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=",", quotechar="'")
            csvwriter.writerow([
                new_id,
                self.export_name,
                accuracy_score(self.y_test, self.y_pred),
                self.pycm.TN,
                self.pycm.FP,
                self.pycm.FN,
                self.pycm.TP
            ])


        # write image ids to new csv (in filename use new index)

        # prepare data
        lists = [self.indices_train, self.y_train, 
                 self.indices_test, self.y_test, self.y_pred]
        # create columns and fill empty cells with ''
        export_data = zip_longest(*lists, fillvalue = '')

        filename = f"{new_id}_{self.export_name}_ids.csv"
        with open(str(self.export_path / filename), 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=",", quotechar="'")
            csvwriter.writerow([
                'indices_train',
                'y_train',
                'indices_test',
                'y_test',
                'y_pred'
            ])
            csvwriter.writerows(export_data)
        
        
        


#test = WindturbineDetector(selection_windturbine_paths=[Path("/data/projects/windturbine-identification-sentinel/croppedTiles/us-uswtdb_selection_windturbines")], 
#                             selection_no_windturbine_paths=[Path("/data/projects/windturbine-identification-sentinel/croppedTiles/selection_no-windturbines")], 
#                             pixel="50p", rotation_range=0, zoom_range=0, width_shift_range=0, height_shift_range=0,
#                             num_cnn_layers=2, filters=32, kernel_sizes=[5, 5], layer_activations=["relu", "relu"],
#                             input_shape=[50, 50, 4], random_state=0)


#test.detect_windturbines_with_CNN()


#print(test.confusion_matrix)
#test.pycm.print_matrix()
#accuracy_score(test.y_test, test.y_pred)


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

