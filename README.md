**Domain Adaptation for Nucleus Identification

This code is used to identify individual nuclei with generative adversarial learning for cross-site histopathological image data, which are acquired from different institutions. The code is implemented with PyTorch (version 0.4.1, https://pytorch.org/) on a Ubuntu Linux machine. 

The method described in the manuscript include two stages: (1) adversarial image translation and (2) deep regression model. This repository contains the source codes for Stage 2, i.e., deep regression model that identifies individual nuclei in translated image data. The image translation in Stage 1 is achieved by using the codes provided by CycleGAN, https://junyanz.github.io/CycleGAN/, and CyCADA, https://github.com/jhoffman/cycada_release.

For Stage 2, the usage of the source codes is shown as follows:

For training: ./train_fcn_cell_class.sh 

For testing: ./eval_fcn_cell_class.sh

<br />
References:

[1] Zhang et al. Generative Adversarial Domain Adaptation for Nucleus Quantification in Images of Tissue Immunohistochemically Stained for Ki-67, JCO Clinical Cancer Informatics, 2020.<br/><br/><br/>


**Domain Adaptation for Nucleus Detection

This code is used to detect individual nuclei with generative adversarial learning for different microscopy modality image data, which are acquired from different microscopes. The code is implemented with PyTorch (version 0.4.1, https://pytorch.org/) on a Ubuntu Linux machine. 

The method described in the manuscript include three stages: (1) adversarial image translation, (2) deep regression model, and (3) fine-tuning with pseudo-labels. This repository contains the source codes for Stages 2 and 3, i.e., train a deep regression model with translated source image data (Stage 2), and use the model to estimate nucleus positions in real target training images and fine-tunes the model with estimated nucleus positions. The image translation in Stage 1 is achieved by using the codes provided by CycleGAN, https://junyanz.github.io/CycleGAN/, and CyCADA, https://github.com/jhoffman/cycada_release.

For Stage 2 and 3 (traing a deep regression model, estimate nucleus positions and fine-tunes the model), the usage of the source codes is shown as follows:

For training: ./train_detection.sh 

For testing: ./eval_detection.sh

<br />
For pseudo-label generation based on the estimated nucleus positions on real target training images, please refer to the paper: Efficient and robust cell detection: A structured regression approach, Medical Image Analysis, 2018. <br /> 

<br /> 
References:

[2] Xing et al. Adversarial Domain Adaptation and Pseudo-Labeling for Cross-Modality Microscopy Image Quantification. MICCAI 2019.
