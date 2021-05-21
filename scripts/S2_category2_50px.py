import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('../')
sys.path.append(os.getcwd())

from windturbine_detection_S2only import WindturbineDetector
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score 

# Sentinel-2 non-shifted 50x50px

runs = 1

# test = WindturbineDetector(selection_windturbine_paths=[Path("/data/projects/windturbine-identification-sentinel/analyses/21_machine_learning/S2_wt-crops")], 
#                              selection_no_windturbine_paths=[Path("/data/projects/windturbine-identification-sentinel/analyses/21_machine_learning/S2_random-crops")], 
#                              categories_windturbine_crops=[2], categories_no_windturbine_crops=[1],
#                              pixel="50p", image_bands=["B02", "B03", "B04", "B08"], 
#                              rotation_range=0, zoom_range=0, width_shift_range=0, height_shift_range=0,
#                              num_cnn_layers=2, filters=32, kernel_sizes=[5, 5], layer_activations=["relu", "relu"],
#                              input_shape=[50, 50, 4], random_state=1, test_size=0.3, 
#                              export_path="/data/projects/windturbine-identification-sentinel/analyses/21_machine_learning/results", 
#                              export_name="s2-cat2-50p", csv_confusion_matrix="s2-cat2-50p.csv")
# test.import_data()

for i in range(0, runs):

    # test.detect_windturbines_with_CNN(random_state=i)
    test = WindturbineDetector(selection_windturbine_paths=[Path("/data/projects/windturbine-identification-sentinel/analyses/21_machine_learning/S2_wt-crops")], 
                             selection_no_windturbine_paths=[Path("/data/projects/windturbine-identification-sentinel/analyses/21_machine_learning/S2_random-crops")], 
                             categories_windturbine_crops=[2], categories_no_windturbine_crops=[1],
                             pixel="50p", image_bands=["B08", "B03", "B04"], 
                             rotation_range=0, zoom_range=0, width_shift_range=0, height_shift_range=0,
                             num_cnn_layers=2, filters=32, kernel_sizes=[5, 5], layer_activations=["relu", "relu"],
                             input_shape=[50, 50, 3], random_state=1, test_size=0.3)
    test.detect_windturbines_with_CNN()
    print(test.confusion_matrix)
    test.pycm.print_matrix()
    print(accuracy_score(test.y_test, test.y_pred)) 
