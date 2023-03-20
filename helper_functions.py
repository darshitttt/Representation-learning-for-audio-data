#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchaudio
import math
import os
import matplotlib.pyplot as plt
import librosa
import numpy as np


# In[2]:


# Given a path, this func gives us the samples from the audio
# Also gives us an option for resampling our data
def _get_sample(path, resample=None):
    effects = [["remix","1"]]
    if resample:
        effects.extend([
           ["lowpass", f"{resample // 2}"],
           ["rate", f"{resample}"]
       ])
    return torchaudio.sox_effects.apply_effects_file(path, effects=effects)


# Calling the above method
def get_sample(path, resample=None):
    return _get_sample(path, resample=resample)


# Same method for getting samples from speech data
def get_speech_sample(path, resample=None):
    '''
    Args:
        path: path to the file
        resample: resampling rate, if applicable
        
    Returns:
        waveform: waveform in the form of a torch.Tensor, with the resampled sr
        sr: the resampled sr, or the default sr
    
    '''
    return _get_sample(path, resample=resample)


# Plotting waveforms of the audio data
def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    '''
    Description: Plots the passed on waveform
    Args: 
        waveform: torch.Tensor waveform
        sample_rate: sr
        title: Title for the plot, Default: "Waveform"
        xlim: xlim for the plot
        ylim: ylim for the plot
    '''
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=False)


# Printing a description for the audio file
def print_stats(waveform, sample_rate=None, src=None):
    '''
    Prints basic stats for a given waveform
    Args:
        waveform: wf in the form of torch.Tensor
        sample_rate: sr, Default=None
        src: source, Default=None
        
    Prints:
        SR, Dtype, Max, Min, Mean, StdDev, and the values from the wf
    '''
    if src:
        print("-"*10)
        print(f"Source: {src}")
        print("-"*10)
    if sample_rate:
        print(f"Sample Rate: {sample_rate}")
    print("Dtype:", waveform.dtype)
    print(f" - Max:     {waveform.max().item():6.3f}")
    print(f" - Min:     {waveform.min().item():6.3f}")
    print(f" - Mean:    {waveform.mean().item():6.3f}")
    print(f" - Std Dev: {waveform.std().item():6.3f}")
    print()
    print(waveform)
    print()



# Plotting the spectrogram of the audio files
def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
    '''
    PLots the spectrogram for the given waveform
    
    Args:
        waveform: wf to be plotted
        sample_rate: sr
        title: Title for the plot, Default: "Spectrogram"
        xlim: xlim for the plot, Default: None
    '''
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show(block=False)


# In[ ]:




