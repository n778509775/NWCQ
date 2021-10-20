# NWCQ
## Joint Deep Learning for Batch Effect Removal and Classification Based on Metabolomics

Codes and data for using our method, a novel end-to-end deep learning framework for improving diagnostic accuracy by batch-effect calibration in metabolomics data.

## Abstract
Metabolomics occupies an important position in both clinical diagnosis performance and basic researches for metabolic signatures and biomarkers. However, batch effects may occur if two technicians are responsible for different subsets of the experiments or if two different lots of reagents, chips or instruments are used. If not properly dealt with, batch effects can subsequently leads to serious concerns about the validity of the biological conclusions.<br />
To fully improve the accuracy of diagnosis for distinguishing between patients and healthy people better, it is necessary to remove the bias caused by the experimental environment. We present a novel end-to-end deep learning framework, which consists of three networks, namely calibrator network, reconstructor(s) and discriminator. We demonstrate that our algorithm outperforms existing methods for removing batch effects in public CyTOF and private MALDI MS datasets.

## Data
**Public CyTOF:** CyTOF is a mass cytometry technology that allows simultaneous measurements of multiple biomarkers in each cell of a specimen. We perform our experiments on a subset of the publicly available data used in Uri Shaham. Peripheral Blood Mononuclear Cells (PBMCs) were collected from two sclerosis patients that were thawed in two batches (on two different days) and incubated with or without some kind of ionomycin marks. All samples had dimension d = 25 and contained 1400-5000 cells each. 

**Private MALDI MS:** Another set of experiments were carried out based on the laser desorption/ionization mass spectrometry (LDI MS) detection results of serum samples. All the serum samples, including healthy controls and systemic lupus erythematosus (SLE) patients were collected according to standard procedures from RenJi hospital, School of Medicine, Shanghai Jiao Tong University. 

## Method
We propose a joint deep learning framework to calibrate batch effect first and then conduct sample classification (e.g., to derive disease diagnosis). Our framework consists of three major branches: (1) a calibrator to minimize the dissimilarity between different batches; (2) reconstructor(s) to guarantee that the sample data can be fully recovered after calibrating batch effect, which assures the fidelity of data processing in our framework; and (3) a discriminator to predict the labels of samples given their calibrated data. The loss function we optimized contains three components: the reconstruction loss between the input and output of the encoder-decoder backbone, the MMD-based loss between the codes of two batches, and classification loss between predictions and true labels. Our framework is shown in figure below.

![](illustration/network.png)

## Results
We apply the proposed method to two applications of CyTOF and MALDI MS, respectively, and demonstrate superior performance in achieving not only good batch effect removal but also satisfactory classification capability. 
Results for public CyTOF: 

![](illustration/CyTOF.png)

Results for private MALDI MS:

![](illustration/MALDI-MS.png)

## Dependencies
- Python 3.6.8<br />
- PyTorch 1.3.1<br />
- Sklearn 0.21.3<br />
- R 3.6.3<br />

## Files
The code we shared contains a total of three projects, namely...
### codes
***multi-reBatch.py***: An example of training label known datasets and computing accuracy on label unknown datasets after batch effect removal<br />
***classify.py***: An example of training label known datasets and computing accuracy on label unknown datasets before calibration<br />
***crossValidation-complex.py***: 10 fold cross validation to request the upper bound of cross-batch prediction<br />
***network.py***: Class definitions for the architecture of each network (Calibrator, Reconstructors, Discriminator)<br />
***function.py***: Data preprocessing, definition of MMD and some functions that implement visualization<br />

### data
The dataset is organized in the data folder:<br />
   ***"1.csv", "2.csv", "3.csv"...***: Every batches of the data in the corresponding project.<br />
   
**NOTE:** Each project is runing in the corresponding directory. The loss curve can be viewed in the corresponding plots folder.<br />

## Run our codes
1. Clone this git repository<br />   
   `git clone https://github.com/n778509775/NWCQ.git`  <br />    
   and install all the requirements listed above. Our operating environment is Ubuntu 16.04. You should **install all packages** required by the program as follows：   <br />   
   `sudo pip3 install -r requirement.txt`<br />   
   If only available for this user, please follow:<br />   
   `pip3 install -r requirement.txt --user`  <br /> 
    <br />
   For individual packages like ‘tkinter’ that cannot be successfully installed by ‘pip’, please try:<br />    
   `sudo apt-get install python3-tk`  <br /> 
    <br />
2. If you consider viewing classification results **before batch effect calibration**, you could run:   <br />    
   `python classify.py --data_folder your_data_path --dataset_file_list file_name_1 file_name_2 file_name_3...` <br />    
   For example, assuming you are currently in the MI directory, please execute：<br />   
   `python classify.py --dataset_file_list 2.csv 3.csv 1.csv` <br />
   <br />
3. If you consider viewing classification results **after batch effect calibration**, you could run:   <br />      
   `python multi-reBatch.py --data_folder your_data_path --dataset_file_list file_name_1 file_name_2 file_name_3...` <br />   
   For example, assuming you are currently in the CHD directory, please execute：<br />   
   `python multi-reBatch.py --dataset_file_list 3.csv 4.csv 1.csv 2.csv`<br />     
4. In order to obtain the upper bound of **cross-validation**, we could conduct 10 fold in-batch cross validation:   <br />    
    `python crossValidation-complex.py --data_folder your_data_path --train_file file_name`  <br />    
    For example, assuming you are currently in the MI directory, please execute：<br />   
    `python crossValidation-complex.py --train_file 1.csv` <br />
    <br />

## Citation
If you find this work useful for your research, please consider citing our article.
