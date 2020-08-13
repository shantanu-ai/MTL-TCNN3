## Description
Repository of Deep Multitask Texture Classifier(MTL-TCNN) - created as a part of Independent Research under Prof (Dr.) Dapeng Oliver Wu, ECE, UF, Florida, USA.

# Introduction:
This project uses the paper: <b>"Using filter banks in Convolutional Neural Networks for texture classification"</b>  [[arXiv]](https://arxiv.org/pdf/1601.02919.pdf) as a baseline model. <br/>
V. Andrearczyk & Paul F. Whelan

The implementation of TCNN3 as a single task classifier can be found at the following [location](https://github.com/Shantanu48114860/TCNN3)

## Abstract
Texture bestows important characteristics of many types of images in computer vision and classifying textures is one of the most challenging problems in pattern recognition which draws the attention of computer vision researchers over many decades. Recently with the popularity of deep learning algorithms, particularly Convolution Neural Network (CNN), researchers can extract features that helped them to improve the performance of tasks like object detection and recogni- tion significantly over previous handcrafted features. In texture classification, the CNN layers can be used as filter banks for feature extraction whose complexity will increase with the depth of the network. In this study, we introduce a novel multitask texture classifier(MTL-TCNN) where we used multitask learning instead of pretraining sharing feature representation between two common tasks; one task being identifying the objects from Imagenet dataset using Alexnet and second, being classifying the textures using TCNN3. For evaluation, we used two standard benchmark datasets (KTH-Tips and DTD) for texture classifi- cation. Our experiments demonstrated enhanced performance classifying textures over TCNN3.

## Contributors
[Shantanu Ghosh](https://www.linkedin.com/in/shantanu-ghosh-b369783a/)

[Dapeng Oliver Wu](http://www.wu.ece.ufl.edu/)

## Dependencies
[python 3.7.7](https://www.python.org/downloads/release/python-374/)

[pytorch 1.3.1](https://pytorch.org/get-started/previous-versions/)

## Dataset 
ImageNet:
The ImageNet dataset files can be accessed from the [location](https://uflorida-my.sharepoint.com/personal/shantanughosh_ufl_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fshantanughosh%5Fufl%5Fedu%2FDocuments%2FTexture%5FDataset%2FDataset%2FImageNet).
One needs to download the files and place them in /Dataset/ImageNet folder.

DTD:
The DTD dataset files can be accessed from the [location](https://uflorida-my.sharepoint.com/personal/shantanughosh_ufl_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fshantanughosh%5Fufl%5Fedu%2FDocuments%2FTexture%5FDataset%2FDataset%2FTexture%2FDTD).
One needs to download the files and place them in /Dataset/Texture/DTD folder.

Kth:
The DTD dataset files can be accessed from the [location](https://uflorida-my.sharepoint.com/personal/shantanughosh_ufl_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fshantanughosh%5Fufl%5Fedu%2FDocuments%2FTexture%5FDataset%2FDataset%2FTexture%2Fkth).
One needs to download the files and place them in /Dataset/Texture/kth folder.

## How to run
To reproduce the experiments mentioned in the report, first download the dataset as described above and then, type the following
command: 

<b>python3 main_texture_classifier.py</b>

# Hyperparameters:
Epochs(DTD): 400<br/>
Epochs(kth): 400<br/>
Learning rate: 0.0001<br/>
Batch size: 32<br/>
Weight Decay: 0.0005<br/>

## Report
The report of this research is kept at the following [location](https://github.com/Shantanu48114860/MTL-TCNN3/blob/master/Report/Texture_Classification.pdf).


## Contact
beingshantanu2406@gmail.com <br/>
shantanu.ghosh@ufl.edu

## License & copyright
© Shantanu Ghosh, University of Florida

Licensed under the [MIT License](LICENSE)
