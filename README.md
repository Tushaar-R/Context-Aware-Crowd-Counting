# Context-Aware-Crowd-Counting
This is an implementation of the CVPR 2019 paper "Context-Aware Crowd Counting".  
# Installation  
1.Install pytorch version 1.0.0 or later.  
2.Install python version 3.6 or later.  
3.Install visdom  
``` pip install visdom ```  
4.Install tqdm  
``` pip install tqdm```  
5.Clone this repository  
```git clone https://github.com/Tushaar-R/Context-Aware-Crowd-Counting.git ```  
We'll call the directory where you cloned Context-Aware_Crowd_Counting-pytorch as ROOT.  
 # Data Setup
 1.Download ShanghaiTech A Dataset from [ShanghaiTech A](https://www.kaggle.com/datasets/tushaar1ranganathan/shanghaitech-zip). Use the folder labeled part_A.  
 2.Download ShanghaiTech B Dataset from [ShanghaiTech B](https://www.kaggle.com/datasets/tushaar1ranganathan/shanghai-tech-partb/data)  
 Note: In the ShanghaiTech A Dataset there is a folder labelled part_B in it, however it's Ground truth maps have not been pre-processed in a suitable format to be taken as input for the code.  
 3.Unzip/ Extract the Datasets before use.  
 # Training
 1.In "implement.py" make sure to replace all paths used for the dataset in the code with the paths of the dataset in your server. You may refer to the comments in the main() function.  
 2.Training parameters such as epochs,learning rates, etc can be changed by modifying their values in main().  
 3.If you are in the cloned directory use command ``` python3 implement.py```, otherwise use the relative/ absolute path of "implement.py" to run it.  
 4. './checkpoints' is created to keep track of previous epoch weights, in the event that program stops execution midway due to unforseen circumstance.
# Testing
1. Test will run automatically in "implement.py".
2. You can modify the index of image which you want to display when calling def estimate_density_map(img_root, gt_dmap_root, index) using the index parameter.
# Other Notes
Due to limitations in computation resources of Kaggle, only 360/1000 epochs could be trained for Part A and 125/1000 epochs could be trained for Part B. However results came very close to results of paper, considering that test mae error changed from 400->66 for Part A and 80->13 for Part B

