# ScanDMM: A Deep Markov Model of Scanpath Prediction for 360° Images [CVPR2023]

[[Paper]]() [[openReview]](https://openreview.net/forum?id=Z5RSvPEbyK)  
[![Watch the video](https://img.youtube.com/vi/bEWBnG5GXsU/maxresdefault.jpg)](https://youtu.be/bEWBnG5GXsU)

# Implementation version
Pytorch 1.8.1 & CUDA 10.1.  
Please referring to [requirements.txt](https://github.com/xiangjieSui/ScanDMM/blob/master/requirement.txt) for details.  
If your CUDA version is 10.1, you can directly execute the following command to install the environment：  
```
conda create -n scandmm python==3.7  
conda activate scandmm
pip install -r requirements.txt
```

# Training  
1. To reproduce the training and validation dataset, please referring to [data_process.py](https://github.com/xiangjieSui/ScanDMM/blob/master/data_process.py). Alternatively, using the [ready-to-use data](https://github.com/xiangjieSui/ScanDMM/tree/master/Datasets).
2. Execute:  
```
python train.py --seed=1234 --dataset='./Datasets/Sitzmann.pkl' --lr=0.0003 --bs=64 --epochs=500 --save_root='./model/'
```
3. Check the training log and checkpoints in Log (created automatically) and [./model](https://github.com/xiangjieSui/ScanDMM/tree/master/model) files, respectively.

# Test  
1. Prepare the test images and put them in a folder (e.g, [./demo/input](https://github.com/xiangjieSui/ScanDMM/tree/master/demo/input))  
2. Create an folder to store the results (e.g, [./demo/output](https://github.com/xiangjieSui/ScanDMM/tree/master/demo/output)) 
3. A pre-trained weights (e.g, ['./model/model_lr-0.0003_bs-64_epoch-435.pkl'](https://github.com/xiangjieSui/ScanDMM/tree/master/model))  
4. Execute:
```
python inference.py --model='./model/model_lr-0.0003_bs-64_epoch-435.pkl' --inDir='./demo/input' --outDir='./demo/output' --n_scanpaths=200 --length=20 if_plot==True
```
5. Check the results:  
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

# Bibtex
```
@article{scandmm2023,
  title={ScanDMM: A Deep Markov Model of Scanpath Prediction for 360° Images},
  author={Xiangjie Sui and Yuming Fang and Hanwei Zhu and Shiqi Wang and Zhou Wang},
  journal = {IEEE Conference on Computer Vision and Pattern Recognition}, 
  year={2023}
}
```

# Acknowledgment
The author would like to thank Daniel Martin for publishing [ScanGAN](https://github.com/DaniMS-ZGZ/ScanGAN360) model and visualization functions. And thank Bingham Eli et al. for the implementation of [Pyro](https://github.com/pyro-ppl/pyro).
