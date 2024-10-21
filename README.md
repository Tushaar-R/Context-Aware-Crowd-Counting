# Context-Aware-Crowd-Counting
This is an implementation of the CVPR 2019 paper "Context-Aware Crowd Counting".  
# Installation  
1.Install pytorch version 1.0.0 or later.  
2.Install python version 3.6 or later.  
3.Install visdom  
``` pip install visdom ```  
4.Install tqdm  
``` pip install tqdm```  
5.Clone this repository  ```git clone https://github.com/Tushaar-R/Context-Aware-Crowd-Counting.git ```  
We'll call the directory that you cloned Context-Aware_Crowd_Counting-pytorch as ROOT.  
 # Data Setup
 1.Download ShanghaiTech A Dataset from [ShanghaiTech A](https://www.kaggle.com/datasets/tushaar1ranganathan/shanghaitech-zip)  Use the folder labelled part_A
 2.Download ShanghaiTech B Dataset from [ShanghaiTech B](https://www.kaggle.com/datasets/tushaar1ranganathan/shanghai-tech-partb/data)  
 Note: In the ShanghaiTech A Dataset there is a dataset labelled part_B in it, however it's Ground truth maps have no been pre-processed in a suitable format to be taken as input for the code.  
 3.Unzip/ Extract the Datasets before use.  
 # Training
 1.In "implement.py" make sure to replace all paths used for the dataset in the code with the paths of the dataset in your server.  
 2.Training parameters such as epochs,learning rates, etc can be changed by modifying their values in main().  
 3.If you are in the cloned directory use command ``` python3 implement.py```, otherwise use the relative/ absolute path of "implement.py" and run it.  
 4. './checkpoints' is created to keep track of previous epoch weights, in the event that program stops execution.
# Testing
1. Test will run automatically in "implement.py".
2. You can modify the index of image which you want to display when calling def estimate_density_map(img_root, gt_dmap_root, index) using the index parameter.

