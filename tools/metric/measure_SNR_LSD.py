#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/30/2019 3:05 PM
# @Author  : HOU NANA
# @Site    : http://github.com/nanahou
# @File    : measure_SNR_LSD.py
import torch
import numpy as np
import librosa
import os
from scripts.extract_LPS_CMVN import get_power_spec

def comp_SNR(x, y):
    """
       Compute SNR (signal to noise ratio)
       Arguments:
           x: vector (torch.Tensor), enhanced signal
           y: vector (torch.Tensor), reference signal(ground truth)
    """
    ref = torch.pow(y, 2)
    if len(x) == len(y):
        diff = torch.pow(x-y, 2)
    else:
        stop = min(len(x), len(y))
        diff = torch.pow(x[:stop] - y[:stop], 2)

    ratio = torch.sum(ref) / torch.sum(diff)
    value = 10*torch.log10(ratio)

    return value

def comp_LSD(x, y):
    """
       Compute LSD (log spectral distance)
       Arguments:
           x: vector (torch.Tensor), enhanced signal
           y: vector (torch.Tensor), reference signal(ground truth)
    """
    if len(x) == len(y):
        diff = torch.pow(x-y, 2)
    else:
        stop = min(len(x), len(y))
        diff = torch.pow(x[:stop] - y[:stop], 2)

    sum_freq = torch.sqrt(torch.sum(diff, dim=1) / diff.size(1))
    value = torch.sum(sum_freq, dim=0) / sum_freq.size(0)

    return value



def main():

    wav_16k = '/data/disk3/hounana/Valentini-Botinhao_16k/formatted_28spk/noisy8k_clean16k/raw_wav/clean_testset_wav_16k/'
    # extend_16k = '/data/disk3/hounana/Valentini-Botinhao_16k/formatted_28spk/noisy8k_clean16k/raw_wav/clean_testset_wav_re16k/'
    # wav_16k = '/data/disk3/hounana/Valentini-Botinhao_16k/formatted_28spk/noisy_re16k_clean16k/tt/clean/'
    extend_16k = '/home/hounana/pytorch/enhancement/conv-tasnet-noisyBWE/exp/conv_tasnet_BWE_regression/results/spk1/'
    wav_list = [x for x in os.listdir(wav_16k) if x.endswith(".wav")]

    sum_snr_enhan = 0.0
    sum_lsd_enhan = 0.0

    for item in wav_list:
        item_org16k = wav_16k + item
        item_extend = extend_16k + item
        # item_extend = extend_16k + item[:-4] + '.wav..pr.wav'

        # compute SNR
        org_sig, org_sr = librosa.load(item_org16k, None, mono=True, offset=0.0, dtype=np.float32)
        ext_sig, ext_sr = librosa.load(item_extend, None, mono=True, offset=0.0, dtype=np.float32)
        x = torch.from_numpy(ext_sig)
        y = torch.from_numpy(org_sig)
        value_snr = comp_SNR(x, y)
        sum_snr_enhan += value_snr

        # compute LSD
        fft_len_16k, frame_shift_16k = 512, 256

        # extract magnitude and power
        power_16k = get_power_spec(item_org16k, fft_len_16k, frame_shift_16k)
        power_16k = torch.from_numpy(power_16k.astype(np.float)).float()

        power_ext = get_power_spec(item_extend, fft_len_16k, frame_shift_16k)
        power_ext = torch.from_numpy(power_ext.astype(np.float)).float()

        log_16k = torch.log(power_16k)
        log_ext = torch.log(power_ext)

        value_lsd = comp_LSD(log_ext, log_16k)

        sum_lsd_enhan += value_lsd


    avg_snr_enhan = sum_snr_enhan / len(wav_list)
    avg_lsd_enhan = sum_lsd_enhan / len(wav_list)

    print('avg_snr_enhan %f, avg_lsd_enhan %f' % (avg_snr_enhan, avg_lsd_enhan))




if __name__ == '__main__':
    main()