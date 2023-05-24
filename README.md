# ScanDMM: A Deep Markov Model of Scanpath Prediction for 360° Images [CVPR2023]

[:sparkles:Paper](https://ece.uwaterloo.ca/~z70wang/publications/CVPR23_scanPath360Image.pdf)&ensp; &ensp;
[:sparkles:Poster](https://cvpr2023.thecvf.com/media/PosterPDFs/CVPR%202023/21446.png?t=1684911665.5451589)&ensp; &ensp;
[:sparkles:Presentation (YouTube)](https://www.youtube.com/watch?v=noXCcFvXY2k)&ensp; &ensp;
[:sparkles:Slide](https://cvpr2023.thecvf.com/media/cvpr-2023/Slides/21446.pdf)&ensp; &ensp;
[:sparkles:OpenReview](https://openreview.net/forum?id=Z5RSvPEbyK) &ensp; &ensp;


https://user-images.githubusercontent.com/65707367/223019204-6948e71f-1f30-4659-9498-353ef74ed1c9.mp4


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
2. Create a folder to store the results (e.g, [./demo/output](https://github.com/xiangjieSui/ScanDMM/tree/master/demo/output)) 
3. A pre-trained weights (e.g, ['./model/model_lr-0.0003_bs-64_epoch-435.pkl'](https://github.com/xiangjieSui/ScanDMM/tree/master/model))  
4. Execute:
```
python inference.py --model='./model/model_lr-0.0003_bs-64_epoch-435.pkl' --inDir='./demo/input' --outDir='./demo/output' --n_scanpaths=200 --length=20 --if_plot=True
```  
* Modify *n_scanpaths* and *length* to change the number and length of the produced scanpaths. Please referring to [inference.py](https://github.com/xiangjieSui/ScanDMM/blob/master/inference.py) for more details about the produced scanpaths.  
5. Check the results:  
``` 
sp_P48_5376x2688.png
```
![Snow](https://github.com/xiangjieSui/ScanDMM/blob/master/demo/output/sp_P48_5376x2688.png)  
```
scanpaths = np.load(P48_5376x2688.npy)
print(scanpaths.shape)
(200, 20, 2)
# (n_scanpaths, length, (y, x)). (y, x) are normalized coordinates in the range [0, 1] (y/x = 0 indicate the top/left edge).
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
@InProceedings{scandmm2023,
  title={ScanDMM: A Deep Markov Model of Scanpath Prediction for 360° Images},
  author={Xiangjie Sui and Yuming Fang and Hanwei Zhu and Shiqi Wang and Zhou Wang},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition}, 
  year={2023}
}
```

# Acknowledgment
The author would like to thank [Kede Ma](https://kedema.org/) for his inspiration, Daniel Martin et al. for publishing [ScanGAN](https://github.com/DaniMS-ZGZ/ScanGAN360) model and visualization functions, and Bingham Eli et al. for the implementation of [Pyro](https://github.com/pyro-ppl/pyro). We sincerely appreciate for their contributions.
