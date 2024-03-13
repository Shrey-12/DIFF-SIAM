# MAD-DDPM
MAD-DDPM is a one-class learning model that uses a reconstruction-based measure to determine whether the input images are bona fide or face morphs. At the core of the technique is a two-branch reconstruction procedure that uses denoising diffusion probabilistic models (DDPMs) learned over only bona-fide samples as the basis for the
detection tasks. The first branch models the distribution on bona-fide samples directly in the pixel-space (for low-level artifact detection), while the second captures the distribution of higher-level features extracted with a pretrained CNN.
      

![MAD-DDPM](MAD_DDPM.png)

## 0. Setting up Tensorflow GPU
```
wget <mini-conda-link>
conda install -y pip
conda create -n <myenv> python=3.8 pip
conda activate <myenv>
conda install cudnn cudatoolkit -c conda-forge
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/Lib1
```


## 1. Install the dependencies
The model is implemented using PyTorch. The full list of used libraries can be found in requirements.txt.
```
pip install torch=1.9.0+cu111 torchvision=0.10.0+cu111 -f https://download.pytorch.org/wh1/cu111/torch_stable.html
pip install -r requirements.txt
```

## 2. Prepare the data
The datasets you are using for training or testing should be placed in the directory called data. Datasets should have the following directory structure:
```
CASIA-WebFace
├── images
│   ├── bonafide
│   │   ├── raw
│   │   │   └── bonafide_img_1.png
│   │   │   └── bonafide_img_2.png
│   │   │   └── ...
├── features_scale_1
│   ├── bonafide
│   │   ├── raw
│   │   │   └── bonafide_img_1.pt
│   │   │   └── bonafide_img_2.pt
│   │   │   └── ...
├── features_scale_2
│   ...

FRLL
├── images
│   ├── bonafide
│   │   ├── raw
│   │   │   └── bonafide_img_1.png
│   │   │   └── bonafide_img_2.png
│   │   │   └── ...
│   ├── morphs
│   │   ├── morphing_method_1
│   │   │   └── morph_img_1.png
│   │   │   └── morph_img_2.png
│   │   │   └── ...
│   │   ├── morphing_method_2
│   │   ...
├── features_scale_1
│   ├── bonafide
│   │   ├── raw
│   │   │   └── bonafide_img_1.pt
│   │   │   └── bonafide_img_2.pt
│   │   │   └── ...
│   ├── morphs
│   │   ├── morphing_method_1
│   │   │   └── morph_img_1.pt
│   │   │   └── morph_img_2.pt
│   │   │   └── ...
│   │   ├── morphing_method_2
│   │   ...
├── features_scale_2
│   ...
```
![image](https://github.com/Shrey-12/MAD-DDPM/assets/98189346/a20ca74e-ce22-4fdc-8927-71a06f59396d) <br>
RetinaFace.ipynb contains a demonstration of its working.
Retina Face(https://github.com/serengil/retinaface)

Images are expected to have one of the following image extensions: '.jpg'. Their corresponding pre-extracted feature maps should be saved with the same name in '.pt' format (PyTorch tensors). features_scale_1 is the root directory of features extracted from cropped face images of size 224x224 pixels (tensor size is 1024x14x14), while features_scale_2 contains features of images of size 448x448 pixels (tensor size is 4x1024x14x14). 
### Model expects (1x980x32x32) = (1x1024x14x14) from 224x224 + (1x4x1024x14x14) from 448x448

![image](https://github.com/Shrey-12/MAD-DDPM/assets/98189346/2989a362-c64b-4700-a397-9af913af5d79)

MAD-DDPM is trained and tested on preprocessed datasets, where faces were first detected with RetinaFace, then cropped out with a margin of 5% of the detected bounding box height. Corresponding feature maps are extracted with a pretrained WideResNet. For more details please refer to the paper.

## 3. Training
To train the **image branch** MAD-DDPM on your dataset, run the following:
```
python train.py --train-set ./data/CASIA-WebFace/ --config configs/casia_webface.json --branch 1
```
To train with evaluation on a testing set after each training epoch run:
```
python train.py --train-set ./data/CASIA-WebFace/ --config configs/casia_webface.json --branch 1 --test-every 1 --test-set ./data/FRGC/
```
run process in background
```
CUDA_VISIBLE_DEVICES=2 nohup python train.py --train-set ./data/CASIA-WebFace/ --config configs/casia_webface.json --branch 1 > results/train1.log 2>&1 &
```
To train the **branch for features**, set the value of the argument branch to 2:
```
python train.py --train-set ./data/CASIA-WebFace/ --config configs/casia_webface.json --branch 2
```
Checkpoints are by default exported to the directory named output. 

## 4. Evaluation
To test a pretrained MAD-DDPM model run the following:
```
python test.py --test-set ./data/FRGC/ --image_branch_checkpoint output/experiment/checkpoints/model_image.pth --features_branch_checkpoint output/experiment/checkpoints/model_features.pth --config configs/casia_webface.json
```
## 5. Citing MAD-DDPM
If you find this code useful or you want to refer to the paper, please cite using the following BibTeX:
```
@INPROCEEDINGS{ivanovska2023mad_ddpm,
  author={Ivanovska, Marija and Štruc, Vitomir},
  booktitle={2023 11th International Workshop on Biometrics and Forensics (IWBF)}, 
  title={Face Morphing Attack Detection with Denoising Diffusion Probabilistic Models}, 
  year={2023},
  doi={10.1109/IWBF57495.2023.10156877}}}
```

## Acknowledgements
This code is largely based on [k-diffusion](https://github.com/crowsonkb/k-diffusion).

  
## References
[^1]: M. Ivanovska, V. Struc, Face Morphing Attack Detection with Denoising Diffusion Probabilistic Models, International Workshop on Biometrics and Forensics (IWBF), 2023
