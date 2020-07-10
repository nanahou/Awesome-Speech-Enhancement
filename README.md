[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) 
[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning/graphs/commit-activity)
[![Commit Activity](https://img.shields.io/github/commit-activity/m/HuaizhengZhang/Awesome-System-for-Machine-Learning.svg?color=red)](https://github.com/nanahou/Awesome-Speech-Enhancement/graphs/commit-activity)
[![Last Commit](https://img.shields.io/github/last-commit/HuaizhengZhang/Awesome-System-for-Machine-Learning.svg)](https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning/commits/master)
[![Contribution](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/wq2012/awesome-diarization/blob/master/CONTRIBUTING.md)
[![GitHub license](https://img.shields.io/github/license/HuaizhengZhang/Awesome-System-for-Machine-Learning.svg?color=blue)](https://github.com/nanahou/Awesome-Speech-Enhancement/blob/master/LICENSE)


# Awesome Speech Enhancement
## Table of contents
* [Overview](#Overview)
* [Publications](#Publications)
  * [Survey](*Survey)
  * [Feature augmentation](#Feature-augmentation)
  * [Network design](#Network-design)
      * [Filter design](#Filter-design)
      * [Fusion techniques](#Fusion-techniques)
      * [Attention](#Attention)
      * [U-net](#U-Net)
      * [GAN](#GAN)
  * [Phase reconstruction](#Phase-reconstruction)
  * [Learning strategy](#Learning-strategy)
    * [Loss function](#Loss-function)
    * [Multi-task learning](#Multi-task-learning)
    * [Curriculum learning](#Curriculum-learning)
  * [Other improvements](#Other-improvements)
* [Datasets](#Datasets)
* [Tools](#Tools)
  * [Framework](#Framework)
  * [Evaluation](#Evaluation)
  * [Audio feature extraction](#Audio-feature-extraction)
  * [Audio data augmentation](#Audio-data-augmentation)
* [SOTA results](#SOTA-results)
* [Learning materials](#Learning-materials)
  * [Book or thesis](#book-or-thesis)
  * [Video](#Video)
  * [Slides](#Slides)

## Overview

This is a curated list of awesome Speech Enhancement tutorials, papers, libraries, datasets, tools, scripts and results. The purpose of this repo is to organize the world’s resources for speech enhancement, and make them universally accessible and useful.

To add items to this page, simply send a pull request.

## Publications
### Coming soon...
#### Survey
* A literature survey on single channel speech enhancement, 2020 [[paper]](http://www.ijstr.org/final-print/mar2020/A-Literature-Survey-On-Single-Channel-Speech-Enhancement-Techniques.pdf)
* Research Advances and Perspectives on the Cocktail Party Problem and Related Auditory Models, Bo Xu, 2019 [[paper (Chinese)]](http://www.aas.net.cn/article/zdhxb/2019/2/234)
* Deep Learning for Environmentally Robust Speech Recognition: An Overview of Recent Developments, Zixing Zhang, 2017 [[paper]](https://arxiv.org/pdf/1705.10874.pdf)
* Supervised speech separation based on deep learning: An Overview, 2017 [[paper]](https://arxiv.org/pdf/1708.07524.pdf)
* A review on speech enhancement techniques, 2015 [[paper]](https://ieeexplore.ieee.org/document/7087096)
* Nonlinear speech enhancement: an overview, 2007 [[paper]](https://www.researchgate.net/publication/225400856_Nonlinear_Speech_Enhancement_An_Overview)
#### Feature augmentation
* Speech enhancement using self-adaptation and multi-head attention, ICASSP 2020 [[paper]](https://arxiv.org/pdf/2002.05873.pdf)
* PAN: phoneme-aware network for monaural speech enhancement, ICASSP 2020 [[paper]](https://ieeexplore.ieee.org/document/9054334)
* Noise tokens: learning neural noise templates for environment-aware speech enhancement [[paper]](https://arxiv.org/pdf/2004.04001.pdf)
* Speaker-aware deep denoising autoencoder with embedded speaker identity for speech enhancement, Interspeech 2019 [[paper]](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2108.pdf)
#### Network design
##### Filter design
  * Efficient trainable front-ends for neural speech enhancement, ICASSP 2020 [[paper]](https://arxiv.org/pdf/2002.09286.pdf)
##### Fusion techniques
  * Masking and inpainting: a two-stage speech enhancement approach for low snr and non-stationary noise, ICASSP 2020 [[paper]](https://ieeexplore.ieee.org/document/9053188)
  * A composite dnn architecture for speech enhancement, ICASSP 2020 [[paper]](https://ieeexplore.ieee.org/document/9053821)
  * Multi-domain processing via hybrid denoising networks for speech enhancement, 2018 [[paper]](https://arxiv.org/pdf/1812.08914.pdf)
##### Attention
  * Speech enhancement using self-adaptation and multi-head attention, ICASSP 2020 [[paper]](https://arxiv.org/pdf/2002.05873.pdf)
  * Channel-attention dense u-net for multichannel speech enhancement, ICASSP 2020 [[paper]](https://arxiv.org/pdf/2001.11542.pdf)
  * T-GSA: transformer with gaussian-weighted self-attention for speech enhancement, ICASSP 2020 [[paper]](https://arxiv.org/pdf/1910.06762.pdf)
##### U-net
  * Phase-aware speech enhancement with deep complex u-net, ICLR 2019 [[paper]](https://openreview.net/pdf?id=SkeRTsAcYm) [[code]](https://github.com/sweetcocoa/DeepComplexUNetPyTorch)
##### GAN
  * PAGAN: a phase-adapted generative adversarial networks for speech enhancement, ICASSP 2020 [[paper](https://ieeexplore.ieee.org/document/9054256) 
  * Time-frequency masking-based speech enhancement using generative adversarial network, ICASSP 2018 [[paper]](http://150.162.46.34:8080/icassp2018/ICASSP18_USB/pdfs/0005039.pdf)
  * SEGAN: speech enhancement generative adversarial network, Interspeech 2017 [[paper]](https://arxiv.org/pdf/1703.09452.pdf) 
#### Phase reconstruction
* Phase reconstruction based on recurrent phase unwrapping with deep neural networks, ICASSP 2020 [[paper]](https://arxiv.org/pdf/2002.05832.pdf)
* PAGAN: a phase-adapted generative adversarial networks for speech enhancement, ICASSP 2020 [[paper](https://ieeexplore.ieee.org/document/9054256)
* Invertible dnn-based nonlinear time-frequency transform for speech enhancement, ICASSP 2020 [[paper]](https://arxiv.org/pdf/1911.10764.pdf)
* Phase-aware speech enhancement with deep complex u-net, ICLR 2019 [[paper]](https://openreview.net/pdf?id=SkeRTsAcYm) [[code]](https://github.com/sweetcocoa/DeepComplexUNetPyTorch)
* PHASEN: A Phase-and-Harmonics-Aware Speech Enhancement Network, AAAI 2020 [[paper]](https://aaai.org/Papers/AAAI/2020GB/AAAI-YinD.3057.pdf)
#### Learning strategy
##### Loss function
  * Speech denoising with deep feature losses, Interspeech 2019 [[paper]](https://arxiv.org/pdf/1806.10522.pdf)
  * End-to-end multi-task denoising for joint sdr and pesq optimization, Arxiv 2019 [[paper]](https://arxiv.org/pdf/1901.09146.pdf)
##### Multi-task learning
##### Curriculum learning
#### Other improvements
* Improving robustness of deep learning based monaural speech enhancement against processing artifacts, ICASSP 2020 [[paper]](https://ieeexplore.ieee.org/document/9054145)


## Tools
#### Framework

| Link | Language | Description |
| ---- | -------- | ----------- |
| [SETK](https://github.com/funcwj/setk) | Python & C++ | SETK: Speech Enhancement Tools integrated with Kaldi. |
| [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis) ![GitHub stars](https://img.shields.io/github/stars/tyiannak/pyAudioAnalysis?style=social) | Python | Python Audio Analysis Library: Feature Extraction, Classification, Segmentation and Applications. |
| [Beamformer](https://github.com/funcwj/setk/tree/master/doc/adaptive_beamformer) | Python | Implementation of the mask-based adaptive beamformer (MVDR, GEVD, MCWF). |
| [Time-frequency Mask](https://github.com/funcwj/setk/tree/master/doc/tf_mask) | Python | Computation of the time-frequency mask (PSM, IRM, IBM, IAM, ...) as the neural network training labels. |
| [SSL](https://github.com/funcwj/setk/tree/master/doc/ssl) | Python | Implementation of Sound Source Localization. |
| [Data format](https://github.com/funcwj/setk/tree/master/doc/format_transform) | Python | Format tranform between Kaldi, Numpy and Matlab. |


#### Evaluation

| Link | Language | Description |
| ---- | -------- | ----------- |
| [PESQ etc.](tools) | Matlab | Evaluation for PESQ, CSIG, CBAK, COVL, STOI |
| [SNR, LSD](https://github.com/nanahou/metric) | Python | Evaluation for signal-to-noise-ratio and log-spectral-distortion. |
| [SDR](https://github.com/nanahou/metric) | Matlab | Evaluation for signal-to-distortion-ratio. |

#### Audio feature extraction

| Link  | Language | Description |
| ----  | -------- | ----------- |
| [LPS](https://github.com/nanahou/LPS_extraction) | Python | Extract log-power-spectrum/magnitude spectrum/log-magnitude spectrum/Cepstral mean and variance normalization. |
| [MFCC](https://github.com/jameslyons/python_speech_features) ![GitHub stars](https://img.shields.io/github/stars/jameslyons/python_speech_features?style=social) | Python | This library provides common speech features for ASR including MFCCs and filterbank energies. |
| [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis) ![GitHub stars](https://img.shields.io/github/stars/tyiannak/pyAudioAnalysis?style=social) | Python | Python Audio Analysis Library: Feature Extraction, Classification, Segmentation and Applications. |

#### Audio data augmentation

| Link  | Language | Description |
| ----  | -------- | ----------- |
| [Data simulation](https://github.com/funcwj/setk/tree/master/doc/data_simu) | Python | Add reverberation, noise or mix speaker. |
| [RIR simulation](https://github.com/funcwj/setk/tree/master/doc/rir) | Python | Generation of the room impluse response (RIR) using image method. |
| [pyroomacoustics](https://github.com/LCAV/pyroomacoustics) ![GitHub stars](https://img.shields.io/github/stars/LCAV/pyroomacoustics?style=social) | Python | Pyroomacoustics is a package for audio signal processing for indoor applications. |
| [gpuRIR](https://github.com/DavidDiazGuerra/gpuRIR) ![GitHub stars](https://img.shields.io/github/stars/DavidDiazGuerra/gpuRIR?style=social) | Python | Python library for Room Impulse Response (RIR) simulation with GPU acceleration |
| [rir_simulator_python](https://github.com/sunits/rir_simulator_python) ![GitHub stars](https://img.shields.io/github/stars/sunits/rir_simulator_python?style=social) | Python | Room impulse response simulator using python |
| [audiomentations](https://github.com/iver56/audiomentations) ![GitHub stars](https://img.shields.io/github/stars/iver56/audiomentations?style=social) | Python | A Python library for audio data augmentation, e.g. time stretch, pitch shift, add noise, add room reverberation |


## Datasets
#### Speech ehancement datasets (sorted by usage frequency in paper) 

| Name | Utterances | Speakers | Language | Pricing | Additional information |
| ---- | ---------- | -------- | -------- | ------- | ---------------------- |
| [Dataset by University of Edinburgh](https://datashare.is.ed.ac.uk/handle/10283/1942) (2016)| 35K+ | 86 | English | Free | Noisy speech database for training speech enhancement algorithms and TTS models. |
| [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1) (1993)| 6K+ | 630 | English | $250.00 | The TIMIT corpus of read speech is one of the earliest speaker recognition datasets. |
| [VCTK](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html) (2009) | 43K+ | 109 | English | Free | Most were selected from a newspaper plus the Rainbow Passage and an elicitation paragraph intended to identify the speaker's accent. |
| [WSJ0](https://catalog.ldc.upenn.edu/LDC93S6A) (1993) | -- | 149 | English | $1500 | The WSJ database was generated from a machine-readable corpus of Wall Street Journal news text. |
| [LibriSpeech](http://www.openslr.org/12) (2015) | 292K | 2K+ | English | Free | Large-scale (1000 hours) corpus of read English speech. |
| [CHiME series](https://chimechallenge.github.io/chime6/) (~2020) | -- | -- | English | Free | The database is published by CHiME Speech Separation and Recognition Challenge. | 

#### Augmentation noise sources (sorted by usage frequency in paper)

| Name | Noise types | Pricing | Additional information |
| ---- | ----------- | ------- | ---------------------- |
| [DEMAND](https://zenodo.org/record/1227121#.Xv2VsZP7RhE) (2013) | 18 | Free | Diverse Environments Multichannel Acoustic Noise Database provides a set of recordings that allow testing of algorithms using real-world noise in a variety of settings. |
| [115 Noise](http://staff.ustc.edu.cn/~jundu/The%20team/yongxu/demo/115noises.html) (2015)| 115 | Free | The noise bank for simulate noisy data with clean speech. For N1-N100 noises, they were collected by Guoning Hu and the other 15 home-made noise types by USTC.|
| [NoiseX-92](http://www.speech.cs.cmu.edu/comp.speech/Section1/Data/noisex.html) (1996)| 15 | Free | Database of recording of various noises available on 2 CDROMs. |
| [RIR_Noises](https://www.openslr.org/28/) (2017)| - | Free | A database of simulated and real room impulse responses, isotropic and point-source noises. The audio files in this data are all in 16k sampling rate and 16-bit precision.This data includes all the room impulse responses (RIRs) and noises we used in our paper "A Study on Data Augmentation of Reverberant Speech for Robust Speech Recognition" submitted to ICASSP 2017. It includes the real RIRs and isotropic noises from the RWCP sound scene database, the 2014 REVERB challenge database and the Aachen impulse response database (AIR); the simulated RIRs generated by ourselves and also the point-source noises that extracted from the MUSAN corpus. |
## SOTA results
#### STOA results in [dataset by University of Edinburgh](https://datashare.is.ed.ac.uk/handle/10283/1942). The following methods are all trained by "trainset_28spk" and tested by common testset. ("F" denotes frequency-domain and "T" is time-domain.)

| Methods | Publish | Domain | PESQ | CSIG | CBAK | COVL | SegSNR | STOI |
| ------- | ----- |------------ | ---- | ---- | ---- | ---- | ------ | ---- |
| [Noisy](https://arxiv.org/pdf/1703.09452.pdf) | -- | -- | 1.97 | 3.35 | 2.44 | 2.63 | 1.68 | 0.91 |
| [Wiener](https://arxiv.org/pdf/1703.09452.pdf) | -- | -- | 2.22 | 3.23 | 2.68 | 2.67 | 5.07 | -- |
| [SEGAN](https://arxiv.org/pdf/1703.09452.pdf) | INTERSPEECH 2017 | T | 2.16 | 3.48 | 2.94 | 2.80 | 7.73 | 0.93 |
| [CNN-GAN](http://www.apsipa.org/proceedings/2018/pdfs/0001246.pdf) | APSIPA 2018 | F | 2.34 | 3.55 | 2.95 | 2.92 | -- | 0.93 |
| [WaveUnet](https://arxiv.org/pdf/1811.11307.pdf) | arxiv 2018 | T| 2.40 | 3.52 | 3.24 | 2.96 | 9.97 | -- |
| [WaveNet](https://arxiv.org/pdf/1706.07162.pdf) | ICASSP 2018 | T | -- | 3.62 | 3.24 | 2.98 | -- | -- |
| [U-net](https://ejhumphrey.com/assets/pdf/jansson2017singing.pdf) | ISMIR 2017 | F | 2.48 | 3.65 | 3.21 | 3.05 | 9.34 | -- |
| [MSE-GAN](http://150.162.46.34:8080/icassp2018/ICASSP18_USB/pdfs/0005039.pdf) | ICASSP 2018 | F | 2.53 | 3.80 | 3.12 | 3.14 | -- | 0.93 |
| [DFL](https://arxiv.org/pdf/1806.10522.pdf) | INTERSPEECH 2019 | T | -- | 3.86 | 3.33 | 3.22 | -- | -- |
| [DFL reimplemented](https://openreview.net/pdf?id=SkeRTsAcYm) | ICLR 2019 | T | 2.51 | 3.79 | 3.27 | 3.14 | 9.86 |-- |
| [TasNet](https://arxiv.org/pdf/1809.07454.pdf) | TASLP 2019 | T | 2.57 | 3.80 | 3.29 | 3.18 | 9.65 | -- |
| [MDPhD](https://arxiv.org/pdf/1812.08914.pdf) | arxiv 2018 | T&F | 2.70 | 3.85 | 3.39 | 3.27 | 10.22 | -- |
| [Complex U-net](https://openreview.net/pdf?id=SkeRTsAcYm) | INTERSPEECH 2019 | F | 3.24 | 4.34 | 4.10 | 3.81 | 16.85 | -- |
| [Complex U-net reimplemented](https://arxiv.org/pdf/1901.09146.pdf) | arxiv 2019 | F | 2.87 | 4.12 | 3.47 | 3.51 | 9.96 | -- |
| [SDR-PRSQ](https://arxiv.org/pdf/1901.09146.pdf) | arxiv 2019 | F | 3.01 | 4.09 | 3.54 | 3.55 | 10.44 |
| [T-GSA](https://arxiv.org/pdf/1910.06762.pdf) | ICASSP 2020 | F | 3.06 | 4.18 | 3.59 | 3.62 | 10.78 | --|
| [RHRnet](https://arxiv.org/pdf/1904.07294.pdf) (using full dataset) | ICASSP 2020 | T | 3.20 | 4.37 | 4.02 | 3.82 | 14.71 | 0.98 |

## Learning materials
#### Book or thesis
  * Audio Source Separation and Speech Enhancement, Emmanuel Vincent, 2019 [[link]](https://github.com/gemengtju/Tutorial_Separation/tree/master/book)
  * A Study on WaveNet, GANs and General CNNRNN Architectures, 2019 [[link]](http://www.diva-portal.org/smash/get/diva2:1355369/FULLTEXT01.pdf)
  * Deep learning: method and applications, 2016 [[link]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/DeepLearning-NowPublishing-Vol7-SIG-039.pdf)
  * Deep learning by Ian Goodfellow and Yoshua Bengio and Aaron Courville, 2016 [[link]](https://www.deeplearningbook.org/)
  * Robust automatic speech recognition by Jinyu Li and Li Deng, 2015 [[link]](https://www.sciencedirect.com/book/9780128023983/robust-automatic-speech-recognition)

#### Video
  * CCF speech seminar 2020 [[link]](https://www.bilibili.com/video/BV1MV411k7iJ)
  * Real-time Single-channel Speech Enhancement with Recurrent Neural Networks by Microsoft Research, 2019 [[link]](https://www.youtube.com/watch?v=r6Ijqo5E3I4)
  * Deep learning in speech by Hongyi Li, 2019 [[link]](https://www.youtube.com/playlist?list=PLJV_el3uVTsOK_ZK5L0Iv_EQoL1JefRL4)
  * High-Accuracy Neural-Network Models for Speech Enhancement, 2017 [[link]](https://www.microsoft.com/en-us/research/video/high-accuracy-neural-network-models-speech-enhancement/)
  * DNN-Based Online Speech Enhancement Using Multitask Learning and Suppression Rule Estimation, 2015 [[link]](https://www.microsoft.com/en-us/research/video/dnn-based-online-speech-enhancement-using-multitask-learning-and-suppression-rule-estimation/)
  * Microphone array signal processing: beyond the beamformer,2011 [[link]](https://www.microsoft.com/en-us/research/video/microphone-array-signal-processing-beyond-the-beamformer/)

#### Slides
  * Deep learning in speech by Hongyi Li, 2019 [[link]](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML19.html)
  * Learning-based approach to speech enhancement and separation (INTERSPEECH tutorial, 2016) [[link]](https://github.com/nanahou/Awesome-Speech-Enhancement/blob/master/learning-materials/2016-interspeech-tutorial.pdf)
  * Deep learning for speech/language processing (INTERSPEECH tutorial by Li Deng, 2015) [[link]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/interspeech-tutorial-2015-lideng-sept6a.pdf)
  * Speech enhancement algorithms (Stanford University, 2013) [[link]](https://ccrma.stanford.edu/~njb/teaching/sstutorial/part1.pdf)
