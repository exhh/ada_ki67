# Domain Adaptation for Nucleus Identification

This code is used to identify individual nuclei with generative adversarial learning for cross-site histopathological image data, which are acquired from different institutions. The code is implemented with PyTorch (version 0.4.1, https://pytorch.org/) on a Ubuntu Linux machine. This project is supported by the Informatics Technology for Cancer Research (ITCR) program from NIH.


The method described in the manuscript include two stages: (1) adversarial image translation and (2) deep regression model. This repository contains the source codes for Stage 2, i.e., deep regression model that identifies individual nuclei in translated image data. The image translation in Stage 1 is achieved by using the codes provided by CycleGAN, https://junyanz.github.io/CycleGAN/, and CyCADA, https://github.com/jhoffman/cycada_release.

For Stage 2, the usage of the source codes is shown as follows:

For training: ./train_fcn_cell_class.sh
For testing: ./eval_fcn_cell_class.sh

References:
[1] Zhang et al. Generative Adversarial Domain Adaptation for Nucleus Quantification in Images of Tissue Immunohistochemically Stained for Ki-67, JCO Clinical Cancer Informatics, 2020.
