[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning/graphs/commit-activity)
[![Commit Activity](https://img.shields.io/github/commit-activity/m/HuaizhengZhang/Awesome-System-for-Machine-Learning.svg?color=red)](https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning/graphs/commit-activity)
[![Last Commit](https://img.shields.io/github/last-commit/HuaizhengZhang/Awesome-System-for-Machine-Learning.svg)](https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning/commits/master)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) [![Contribution](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/wq2012/awesome-diarization/blob/master/CONTRIBUTING.md)
[![GitHub license](https://img.shields.io/github/license/HuaizhengZhang/Awesome-System-for-Machine-Learning.svg?color=blue)](https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning/blob/master/LICENSE)


# Awesome Speech Enhancement
## Table of contents

* [Overview](#Overview)
* [Learning materials](#Learning-materials)
  * [Book or thesis](#book-or-thesis)
  * [Video tutorials](#Video-tutorials)
  * [Slides](#Slides)
  * [Tech blogs](#Tech-blogs)
* [Publications](#Publications)
  * [Feature augmentation](#Feature-augmentation)
  * [Network design](#Network-design)
    * [Mask-based techniques](#Mask-based-techniques)
      * [Filter design](#Filter-design)
      * [Mask design](#Mask-design)
    * [Regression-based techniques](#Regression-based-techniques)
    * [Fusion techniques](#Fusion-techniques)
  * [Learning strategy](#Learning-strategy)
    * [Loss function](#Loss-function)
    * [Batch Normalization](#Batch-Normalization)
    * [Multi-task learning](#Multi-task-learning)
    * [Curriculum learning](#Curriculum-learning)
  * [Other improvements](#Other-improvements)
* [Datasets](#Datasets)
* [Tools](#Tools)
  * [Framework](#Framework)
  * [Evaluation](#Evaluation)
  * [Audio feature extraction](#Audio-feature-extraction)
  * [Audio data augmentation](#Audio-data-augmentation)
  * [Other sotware](#Other-software)
* [SOTA results](#SOTA-results)
* [Leaderboards](#Leaderboards)
* [Products](#Products)


## Overview

This is a curated list of awesome Speech Enhancement tutorials, papers, libraries, datasets, tools, scripts and results. The purpose of this repo is to organize the world’s resources for speech enhancement, and make them universally accessible and useful.

To add items to this page, simply send a pull request. ([contributing guide](CONTRIBUTING.md))

## Publications

### Single speech enhancement
### Uncategorized
* [A literature survey on single channel speech enhancement, 2020](http://www.ijstr.org/final-print/mar2020/A-Literature-Survey-On-Single-Channel-Speech-Enhancement-Techniques.pdf)
* [A review on speech enhancement techniques, 2015](https://ieeexplore.ieee.org/document/7087096)
* [Nonlinear speechenhancement: an overview, 2007](https://www.researchgate.net/publication/225400856_Nonlinear_Speech_Enhancement_An_Overview)
* [A Regression Approach to Speech Enhancement Based on Deep Neural Networks, TASLP 2013](http://staff.ustc.edu.cn/~jundu/The%20team/yongxu/demo/pdfs/YongXU_Taslp_2015.pdf)
* IRM-based-Speech-Enhancement-using-LSTM
[[Code]](https://github.com/haoxiangsnr/IRM-based-Speech-Enhancement-using-LSTM)
* nn-irm
[[Code]](https://github.com/zhaoforever/nn-irm)
* Speech Enhancement Using a Two-Stage Network for an Efficient Boosting Strategy
[[Code]](https://github.com/jtkim-kaist/Speech-enhancement)[[PDF]](https://ieeexplore.ieee.org/document/8668449)
* SETK: Speech Enhancement Tools integrated with Kaldi 
[[Code]](https://github.com/funcwj/setk)
* sednn:deep_learning_for_speech_enhancement_keras_python 
[[Code]](https://github.com/yongxuUSTC/sednn)
* Speech_Enhancement_DNN_NMF 
[[Code]](https://github.com/eesungkim/Speech_Enhancement_DNN_NMF)
* Deep-Learning-for-Speech-Enhancement 
[[Code]](https://github.com/miralv/Deep-Learning-for-Speech-Enhancement)
* gcc-nmf:Real-time GCC-NMF Blind Speech Separation and Enhancement 
[[Code]](https://github.com/seanwood/gcc-nmf)
* TensorFlow-speech-enhancement-Chinese [[Code]](https://github.com/linan2/TensorFlow-speech-enhancement-Chinese)
* DNN-Speech-enhancement-demo-tool [[Code]](https://github.com/yongxuUSTC/DNN-Speech-enhancement-demo-tool)
* CNN-for-single-channel-speech-enhancement [[Code]](https://github.com/dtx525942103/CNN-for-single-channel-speech-enhancement)
* rnn-speech-denoising [[Code]](https://github.com/amaas/rnn-speech-denoising)
* DNN-SpeechEnhancement [[Code]](https://github.com/hyli666/DNN-SpeechEnhancement)
* segan_pytorch [[Code]](https://github.com/santi-pdp/segan_pytorch)
* PHASEN[[Code]](https://github.com/huyanxin/phasen)
* TCNSE [[Code]](https://github.com/ykoyama58/tcnse)
* pb_chime5:Speech enhancement system for the CHiME-5 dinner party scenario [[Code]](https://github.com/fgnt/pb_chime5)


## tools
* AKtools:the open software toolbox for signal acquisition, processing, and inspection in acoustics [[SVN Code]](https://svn.ak.tu-berlin.de/svn/AKtools)(username: aktools; password: ak)
* MatlabToolbox [[Code]](https://github.com/IoSR-Surrey/MatlabToolbox)
* athena-signal [[Code]](https://github.com/athena-team/athena-signal）
* python_speech_features [[Code]](https://github.com/jameslyons/python_speech_features)
* speechFeatures:语音处理，声源定位中的一些基本特征 [[Code]](https://github.com/SusannaWull/speechFeatures)
* sap-voicebox [[Code]](https://github.com/ImperialCollegeLondon/sap-voicebox)
* Calculate-SNR-SDR [[Code]](https://github.com/JusperLee/Calculate-SNR-SDR)
* RIR-Generator [[Code]](https://github.com/ehabets/RIR-Generator)
* Python library for Room Impulse Response (RIR) simulation with GPU acceleration [[Code]](https://github.com/DavidDiazGuerra/gpuRIR)
* ROOMSIM:binaural image source simulation [[Code]](https://github.com/Wenzhe-Liu/ROOMSIM)
* binaural-image-source-model [[Code]](https://github.com/iCorv/binaural-image-source-model)

### Framework

| Link | Language | Description |
| ---- | -------- | ----------- |
| [SIDEKIT for diarization (s4d)](https://projets-lium.univ-lemans.fr/s4d/) | Python | An open source package extension of SIDEKIT for Speaker diarization. |
| [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis) ![GitHub stars](https://img.shields.io/github/stars/tyiannak/pyAudioAnalysis?style=social) | Python | Python Audio Analysis Library: Feature Extraction, Classification, Segmentation and Applications. |
| [AaltoASR](https://github.com/aalto-speech/speaker-diarization) ![GitHub stars](https://img.shields.io/github/stars/aalto-speech/speaker-diarization?style=social) | Python & Perl | Speaker diarization scripts, based on AaltoASR. |
| [LIUM SpkDiarization](https://projets-lium.univ-lemans.fr/spkdiarization/) | Java | LIUM_SpkDiarization is a software dedicated to speaker diarization (i.e. speaker segmentation and clustering). It is written in Java, and includes the most recent developments in the domain (as of 2013). |
| [kaldi-asr](https://github.com/kaldi-asr/kaldi/tree/master/egs/callhome_diarization) [![Build Status](https://travis-ci.com/kaldi-asr/kaldi.svg?branch=master)](https://travis-ci.com/kaldi-asr/kaldi) | Bash | Example scripts for speaker diarization on a portion of CALLHOME used in the 2000 NIST speaker recognition evaluation. |
| [Alize LIA_SpkSeg](https://alize.univ-avignon.fr/) | C++ | ALIZÉ is an opensource platform for speaker recognition. LIA_SpkSeg is the tools for speaker diarization. |
| [pyannote-audio](https://github.com/pyannote/pyannote-audio) ![GitHub stars](https://img.shields.io/github/stars/pyannote/pyannote-audio?style=social) | Python | Neural building blocks for speaker diarization: speech activity detection, speaker change detection, speaker embedding. |
| [pyBK](https://github.com/josepatino/pyBK) ![GitHub stars](https://img.shields.io/github/stars/josepatino/pyBK?style=social) | Python | Speaker diarization using binary key speaker modelling. Computationally light solution that does not require external training data. |
| [Speaker-Diarization](https://github.com/taylorlu/Speaker-Diarization) ![GitHub stars](https://img.shields.io/github/stars/taylorlu/Speaker-Diarization?style=social) | Python | Speaker diarization using uis-rnn and GhostVLAD. An easier way to support openset speakers. |
| [EEND](https://github.com/hitachi-speech/EEND) ![GitHub stars](https://img.shields.io/github/stars/hitachi-speech/EEND?style=social) | Python & Bash & Perl | End-to-End Neural Diarization. |
| [VBDiarization](https://github.com/Jamiroquai88/VBDiarization) ![GitHub stars](https://img.shields.io/github/stars/Jamiroquai88/VBDiarization?style=social) | Python | Speaker diarization based on Kaldi x-vectors using pretrained model trained in Kaldi ([kaldi-asr/kaldi](https://github.com/kaldi-asr/kaldi)) and converted to ONNX format ([onnx/onnx](https://github.com/onnx/onnx)) running in ONNXRuntime ([Microsoft/onnxruntime](https://github.com/Microsoft/onnxruntime)). |
| [RE-VERB](https://github.com/team-re-verb/RE-VERB) ![GitHub stars](https://img.shields.io/github/stars/team-re-verb/RE-VERB?style=social) | Python & JavaScript | RE: VERB is speaker diarization system, it allows the user to send/record audio of a conversation and receive timestamps of who spoke when. |

### Evaluation

| Link | Language | Description |
| ---- | -------- | ----------- |
| [pyannote-metrics](https://github.com/pyannote/pyannote-metrics) ![GitHub stars](https://img.shields.io/github/stars/pyannote/pyannote-metrics?style=social) [![Build Status](https://travis-ci.org/pyannote/pyannote-metrics.svg?branch=master)](https://travis-ci.org/pyannote/pyannote-metrics)  | Python| A toolkit for reproducible evaluation, diagnostic, and error analysis of speaker diarization systems. |
| [SimpleDER](https://github.com/wq2012/SimpleDER) ![GitHub stars](https://img.shields.io/github/stars/wq2012/SimpleDER?style=social) [![Build Status](https://travis-ci.org/wq2012/SimpleDER.svg?branch=master)](https://travis-ci.org/wq2012/SimpleDER) | Python | A lightweight library to compute Diarization Error Rate (DER). |
| NIST md-eval | Perl | (1) modified [md-eval.pl](http://www1.icsi.berkeley.edu/~knoxm/dia/) from [Mary Tai Knox](http://www1.icsi.berkeley.edu/~knoxm); (2) [md-eval-v21.pl](https://github.com/jitendrab/btp/blob/master/c_code/single_diag_gaussian_no_viterbi/md-eval-v21.pl) from [jitendra](https://github.com/jitendrab); (3) [md-eval-22.pl](https://github.com/nryant/dscore/blob/master/scorelib/md-eval-22.pl) from [nryant](https://github.com/nryant) |
| [dscore](https://github.com/nryant/dscore) ![GitHub stars](https://img.shields.io/github/stars/nryant/dscore?style=social) | Python & Perl | Diarization scoring tools. |
| [Sequence Match Accuracy](https://github.com/google/uis-rnn/blob/master/uisrnn/evals.py) | Python | Match the accuracy of two sequences with Hungarian algorithm. |

### Audio feature extraction

| Link  | Language | Description |
| ----  | -------- | ----------- |
| [LibROSA](https://github.com/librosa/librosa) ![GitHub stars](https://img.shields.io/github/stars/librosa/librosa?style=social) | Python | Python library for audio and music analysis. https://librosa.github.io/ |
| [python_speech_features](https://github.com/jameslyons/python_speech_features) ![GitHub stars](https://img.shields.io/github/stars/jameslyons/python_speech_features?style=social) | Python | This library provides common speech features for ASR including MFCCs and filterbank energies. https://python-speech-features.readthedocs.io/en/latest/ |
| [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis) ![GitHub stars](https://img.shields.io/github/stars/tyiannak/pyAudioAnalysis?style=social) | Python | Python Audio Analysis Library: Feature Extraction, Classification, Segmentation and Applications. |

### Audio data augmentation

| Link  | Language | Description |
| ----  | -------- | ----------- |
| [pyroomacoustics](https://github.com/LCAV/pyroomacoustics) ![GitHub stars](https://img.shields.io/github/stars/LCAV/pyroomacoustics?style=social) | Python | Pyroomacoustics is a package for audio signal processing for indoor applications. It was developed as a fast prototyping platform for beamforming algorithms in indoor scenarios. https://pyroomacoustics.readthedocs.io |
| [gpuRIR](https://github.com/DavidDiazGuerra/gpuRIR) ![GitHub stars](https://img.shields.io/github/stars/DavidDiazGuerra/gpuRIR?style=social) | Python | Python library for Room Impulse Response (RIR) simulation with GPU acceleration |
| [rir_simulator_python](https://github.com/sunits/rir_simulator_python) ![GitHub stars](https://img.shields.io/github/stars/sunits/rir_simulator_python?style=social) | Python | Room impulse response simulator using python |

### Other software

| Link | Language | Description |
| ---- | -------- | ----------- |
| [VB Diarization](https://github.com/wq2012/VB_diarization) ![GitHub stars](https://img.shields.io/github/stars/wq2012/VB_diarization?style=social) [![Build Status](https://travis-ci.org/wq2012/VB_diarization.svg?branch=master)](https://travis-ci.org/wq2012/VB_diarization) | Python | VB Diarization with Eigenvoice and HMM Priors. |

## Datasets
### Speech ehancement datasets

| Name | Utterances | Speakers | Language | Pricing | Additional information |
| ---- | ---------- | -------- | -------- | ------- | ---------------------- |
| [Dataset by University of Edinburgh](https://datashare.is.ed.ac.uk/handle/10283/1942) (2016)| 35K+ | 86 | English | Free | Noisy speech database for training speech enhancement algorithms and TTS models. |
| [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1) (1993)| 6K+ | 630 | English | $250.00 | The TIMIT corpus of read speech is one of the earliest speaker recognition datasets. |
| [VCTK](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html) (2009) | 43K+ | 109 | English | Free | Most were selected from a newspaper plus the Rainbow Passage and an elicitation paragraph intended to identify the speaker's accent. |
| [WSJ]() |
| [LibriSpeech](http://www.openslr.org/12) (2015) | 292K | 2K+ | English | Free | Large-scale (1000 hours) corpus of read English speech. |

### Augmentation noise sources

| Name | Utterances | Pricing | Additional information |
| ---- | ---------- | ------- | ---------------------- |
| [AudioSet](https://research.google.com/audioset/) | 2M | Free | A large-scale dataset of manually annotated audio events. |
| [MUSAN](https://www.openslr.org/17/) | N/A | Free | MUSAN is a corpus of music, speech, and noise recordings. |

## SOTA results
* STOA results in [dataset by University of Edinburgh](https://datashare.is.ed.ac.uk/handle/10283/1942). The following methods are all trained by "trainset_28spk" and tested by common testset.

| Methods | Feature type | PESQ | CSIG | CBAK | COVL | SegSNR | STOI |
| ------- | ------------ | ---- | ---- | ---- | ---- | ------ | ---- |
| Noisy | -- | 1.97 | 3.35 | 2.44 | 2.63 | 1.68 | 0.91 |
| Wiener | -- | 2.22 | 3.23 | 2.68 | 2.67 | 5.07 | -- |
| SEGAN | T | 2.16 | 3.48 | 2.94 | 2.80 | 7.73 | 0.93 |
| CNN-GAN | F | 2.34 | 3.55 | 2.95 | 2.92 | -- | 0.93 |
| WaveUnet | T| 2.40 | 3.52 | 3.24 | 2.96 | 9.97 | -- |
| WaveNet | T | -- | 3.62 | 3.24 | 2.98 | -- | -- |
| U-net | F | 2.48 | 3.65 | 3.21 | 3.05 | 9.34 | -- |
| MSE-GAN | F | 2.53 | 3.80 | 3.12 | 3.14 | -- | 0.93 |
| DFL | T | -- | 3.86 | 3.33 | 3.22 | -- | -- |
| DFL reimplemented by | T | 2.51 | 3.79 | 3.27 | 3.14 | 9.86 |-- |
| TasNet | T | 2.57 | 3.80 | 3.29 | 3.18 | 9.65 | -- |
| MDPhD | T&F | 2.70 | 3.85 | 3.39 | 3.27 | 10.22 | -- |
| Complex U-net | F | 3.24 | 4.34 | 4.10 | 3.81 | 16.85 | -- |
| Complex U-net reimplemented by | F | 2.87 | 4.12 | 3.47 | 3.51 | 9.96 | -- |
| SDR-PRSQ | F | 3.01 | 4.09 | 3.54 | 3.55 | 10.44 |
| RHRnet | T | 3.20 | 4.37 | 4.02 | 3.82 | 14.71 | 0.98 |



## Other learning materials
### Book or thesis
* [A Study on WaveNet, GANs and General CNNRNN Architectures, 2019](http://www.diva-portal.org/smash/get/diva2:1355369/FULLTEXT01.pdf)
* [Deep learning: method and applications, 2016](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/DeepLearning-NowPublishing-Vol7-SIG-039.pdf)
* [Deep learning by Ian Goodfellow and Yoshua Bengio and Aaron Courville, 2016](https://www.deeplearningbook.org/)
* [Robust automatic speech recognition by Jinyu Li and Li Deng, 2015](https://www.sciencedirect.com/book/9780128023983/robust-automatic-speech-recognition)

### Video tutorials
* [CCF speech seminar 2020](https://www.bilibili.com/video/BV1MV411k7iJ)
* [Real-time Single-channel Speech Enhancement with Recurrent Neural Networks by Microsoft Research, 2019](https://www.youtube.com/watch?v=r6Ijqo5E3I4)
* [Deep learning in speech by Hongyi Li, 2019](https://www.youtube.com/playlist?list=PLJV_el3uVTsOK_ZK5L0Iv_EQoL1JefRL4)
* [High-Accuracy Neural-Network Models for Speech Enhancement, 2017](https://www.microsoft.com/en-us/research/video/high-accuracy-neural-network-models-speech-enhancement/)
* [DNN-Based Online Speech Enhancement Using Multitask Learning and Suppression Rule Estimation, 2015](https://www.microsoft.com/en-us/research/video/dnn-based-online-speech-enhancement-using-multitask-learning-and-suppression-rule-estimation/)
* [Microphone array signal processing: beyond the beamformer,2011](https://www.microsoft.com/en-us/research/video/microphone-array-signal-processing-beyond-the-beamformer/)


### Slides
* [Deep learning in speech by Hongyi Li, 2019](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML19.html)
* [Learning-based approach to speech enhancement and separation (INTERSPEECH tutorial, 2016)](https://github.com/nanahou/Awesome-Speech-Enhancement/blob/master/learning-materials/2016-interspeech-tutorial.pdf)
* [Deep learning for speech/language processing (INTERSPEECH tutorial by Li Deng, 2015)](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/interspeech-tutorial-2015-lideng-sept6a.pdf)
* [Speech enhancement algorithms (Stanford University, 2013)](https://ccrma.stanford.edu/~njb/teaching/sstutorial/part1.pdf)

## Products

| Company | Product |
| ------- | ------- |
| Google  | [Google Cloud Speech-to-Text API](https://cloud.google.com/speech-to-text/docs/multiple-voices) |
| Amazon  | [Amazon Transcribe](https://aws.amazon.com/transcribe) |
| IBM     | [Watson Speech To Text API](https://www.ibm.com/watson/services/speech-to-text) |
| DeepAffects | [Speaker Diarization API](https://www.deepaffects.com/diarization-api) |
