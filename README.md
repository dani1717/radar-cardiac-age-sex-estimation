# Gender and Age Estimation from Cardiac Signals Captured via Radar using Data Augmentation and Deep Learning: A Privacy Concern

## Overview

This repository contains key components and implementation details related to the paper:

*Gender and Age Estimation from Cardiac Signals Captured via Radar using Data Augmentation and Deep Learning: A Privacy Concern*  
*Daniel Foronda-Pascual, Carmen Camara, Pedro Peris-Lopez*  
*Frontiers in Digital Health*  
*2025*

It provides implementations of the following modules used in the study:

- MODWT cardiac signal extraction via MATLAB Wavelet Toolbox functions (`modwt` and `modwtmra`) called through the Python-MATLAB engine.
- Conditional Wasserstein GAN (cWGAN) for data augmentation of radar signals.
- CNN model architecture for age and sex prediction from radar scalogram data.
- Dataset splitting and sampling functions that ensure balanced train/test sets by age group, and sex.


## Repository Contents

- `modwt_cardiac_extraction.py` — MODWT signal processing functions.
- `cWGAN.py` — Conditional Wasserstein GAN implementation for data augmentation.
- `CNN.py` — CNN_5layers_AgeSex model architecture.
- `split_function.py` — Dataset splitting and sampling utilities.


## Requirements

- Python 3.11
- PyTorch 2.3.0 with CUDA 12.2 (for GPU acceleration)
- MATLAB R2024a or later (for MODWT functions via engine)
- Other Python dependencies listed in `requirements.txt`

## Reproducibility

This repository aims to facilitate full reproducibility of the experiments described in the paper. Please consult the paper for detailed explanations of the experimental setup and dataset.

## License

This code is released under the Apache License 2.0.

## Citation

If you use this repository for your research, please cite the paper.

## Contact
For questions or issues, please contact Daniel Foronda-Pascual (daniel.foronda@uc3m.es).
