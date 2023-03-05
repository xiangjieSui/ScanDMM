# ScanDMM

This is the repository of paper：  
```
@article{scandmm2023,
  title={ScanDMM: A Deep Markov Model of Scanpath Prediction for 360° Images},
  author={Xiangjie Sui and Yuming Fang and Hanwei Zhu and Shiqi Wang and Zhou Wang},
  journal = {IEEE Conference on Computer Vision and Pattern Recognition}, 
  year={2023}
}
```

# Implementation version:
Pytorch 1.8.1 & CUDA 10.1.  
Please referring to [requirements.txt](https://github.com/xiangjieSui/ScanDMM/blob/master/requirement.txt) for details.  
If your CUDA version is 10.1, you can also directly execute the following command to install the environment：  
```
conda create -n scandmm python==3.7  
conda activate scandmm
pip install -r requirements.txt
```

# Training  
1. To reproduce the training and validation dataset, please referring to [data_process.py](https://github.com/xiangjieSui/ScanDMM/blob/master/data_process.py). Alternatively, using the [ready-to-use data](https://github.com/xiangjieSui/ScanDMM/tree/master/Datasets).
2. Modify [training parameters](https://github.com/xiangjieSui/ScanDMM/blob/master/config.py) to satisfy your configuration.
3. Runing [train.py](https://github.com/xiangjieSui/ScanDMM/blob/master/train.py)
4. Check the training log and checkpoint in Log (created automatically) and [model](https://github.com/xiangjieSui/ScanDMM/tree/master/model) files, respectively.

# Test  
1. Prepare the test images and put them in a folder (e.g, [./demo/input](https://github.com/xiangjieSui/ScanDMM/tree/master/demo/input))  
2. Create an folder to store the results (e.g, [./demo/output](https://github.com/xiangjieSui/ScanDMM/tree/master/demo/output)) 
3. Excute:
```
python inference.py --model='./model/model_lr-0.0003_bs-64_epoch-435.pkl' --inDir='./demo/input' --outDir='./demo/output' --n_scanpaths=200 --length=20 if_plot==True
```
4. Check the results:  
``` 
sp_P48_5376x2688.png
```
![Snow](https://github.com/xiangjieSui/ScanDMM/blob/master/demo/output/sp_P48_5376x2688.png)  
```
scanpaths = np.load(P48_5376x2688.npy)
print(scanpaths.shape)
(200, 20, 2)
```

```
sp_P8_7500x3750.png
```
![Mu](https://github.com/xiangjieSui/ScanDMM/blob/master/demo/output/sp_P8_7500x3750.png)
```
scanpaths = np.load(P8_7500x3750.npy)
print(scanpaths.shape)
(200, 20, 2)
```

