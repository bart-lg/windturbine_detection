import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('../')
sys.path.append(os.getcwd())

from windturbine_detection_S1S2combined import WindturbineDetector
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score 

# Sentinel-1 and Sentinel-2 combined and randomly shifted by up to 50% with an image size of 50x50pixel

runs = 1

for i in range(0, runs):
    test = WindturbineDetector(S1_selection_windturbine_path=Path("/data/projects/windturbine-identification-sentinel/analyses/21_machine_learning/S1_wt-crops"), 
                             S1_selection_no_windturbine_path=Path("/data/projects/windturbine-identification-sentinel/analyses/21_machine_learning/S1_random-crops"), 
                             S2_selection_windturbine_path=Path("/data/projects/windturbine-identification-sentinel/analyses/21_machine_learning/S2_wt-crops"),
                             S2_selection_no_windturbine_path=Path("/data/projects/windturbine-identification-sentinel/analyses/21_machine_learning/S2_random-crops"),
                             categories_windturbine_crops=[3], categories_no_windturbine_crops=[1],
                             pixel="50p", image_bands=["B04", "B08", "VV"], S2_rescale_factor=2**14, S1_value_increment=5, S1_rescale_factor=20, 
                             rotation_range=0, zoom_range=0, width_shift_range=0, height_shift_range=0,
                             num_cnn_layers=2, filters=32, kernel_sizes=[5, 5], layer_activations=["relu", "relu"],
                             input_shape=[50, 50, 3], random_state=1, test_size=0.3)
    test.detect_windturbines_with_CNN()
    print(test.confusion_matrix)
    test.pycm.print_matrix()
    print(accuracy_score(test.y_test, test.y_pred)) 
