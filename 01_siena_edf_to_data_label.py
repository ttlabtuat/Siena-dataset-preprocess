# ----------------------------------------------
# Siena dataset: EDF to data label
# Author: Kazi Mahmudul HASSAN 
# Date: September 22, 2025
# If you use this code, please cite this paper: http://dx.doi.org/10.1561/116.20240032

# ----------------------------------------------
# sample command
# python 01_siena_edf_to_data_label.py --test_pt 1
# 
# ----------------------------------------------

import os
import csv
import mne
import h5py
import time
import random
import pprint
import argparse
import linecache

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

import pandas as pd

import os
from os import listdir
from os.path import isfile, join

import mne
from mne import make_fixed_length_epochs 

from tqdm import tqdm



if __name__=='__main__':
    

    # Parameter list -------[Start]------------------------------------

    parser = argparse.ArgumentParser(description='Siena Data generation')

    parser.add_argument('--SAMP_FREQ',  type=float,        default=512.0,   help='Sampling Frequency')
    parser.add_argument('--Down_FREQ',  type=float,        default=500.0,   help='Downsampling Frequency')
    parser.add_argument('--duration',   type=float,        default=10.0,    help='duration')
    parser.add_argument('--interval',   type=float,        default=10.0,    help='interval')
    parser.add_argument('--lpfilt',     type=float,        default=1.0,     help='Lowpass Filter in frequency')
    parser.add_argument('--hpfilt',     type=float,        default=60.00,   help='Highpass Filter in frequency')
    parser.add_argument('--prepross',   type=str,          default='y',     help='Preprocessing')
    parser.add_argument('--stdz',       type=str,          default='n',     help='Standardization')
    parser.add_argument('--test_pt',    type=int,          default=0,       help='Patient ID')

    args = parser.parse_args()
    stdz = args.stdz
    prepross  = args.prepross
    
    SAMP_FREQ = args.SAMP_FREQ
    Down_FREQ = args.Down_FREQ
    
    duration  = args.duration
    interval  = args.interval
    
    LowpassFilt  = args.lpfilt
    HighpassFilt = args.hpfilt

    test_pt  = args.test_pt 

    # Parameter list -------[End]------------------------------------
    

    # directory ------------------------------------------------
    home_dir = "/shared/public/datasets/siena-scalp-eeg/1.0.0" 
    ROOT_DIR = f"/shared/home/hassan/siena-dataset/siena-{duration}s-{interval}s-{LowpassFilt}-{HighpassFilt}hz-Fs{Down_FREQ}hz-monopolar" # change this directory

    h5_DIR = os.path.join(ROOT_DIR, "data-label")    


    if not os.path.exists(h5_DIR):
        os.makedirs(h5_DIR)


    # directory ------------------------------------------------
    df = pd.read_csv('siena_sz_info.csv') 
    # display(df)
    uq_edf = np.unique(df['edf_name'])
    
    # print('#---------[EDF list]--------------')
    # for i, edf in enumerate(uq_edf,1):
    #     print(edf'{i}:{edf}')
    
    # Patient IDs
    # P = ['PN00', 'PN01', 'PN03', 'PN05', 'PN06', 'PN07', 'PN09', 
    #      'PN10', 'PN11', 'PN12', 'PN13', 'PN14', 'PN16', 'PN17']
    
    # P = ['PN01']
    
    Channel_list = ['eeg fp1', 'eeg fp2', 'eeg f3', 'eeg f4', 'eeg c3', 'eeg c4', 'eeg p3', 'eeg p4', 'eeg o1', 'eeg o2', 
                    'eeg f7',   'eeg f8', 'eeg t3', 'eeg t4', 'eeg t5', 'eeg t6', 'eeg fz', 'eeg cz', 'eeg pz']

    New_Channel_list = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
                         'F7',  'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']


    rename_dict = dict(zip(Channel_list, New_Channel_list))
    print(rename_dict)

    print('\n\n# Channel rename list : \n')
    for key, val in rename_dict.items():
        print(f'{key}: {val}')


    # uq_edf = uq_edf[:3]
    ep_label_shape_flag = {}
    

    print('#---------[EDF list]--------------')


    uq_edf_pt = [e for e in uq_edf if str(test_pt).zfill(2)==e[2:4]]
    print("EDF_list : \n")
    print(*uq_edf_pt, sep="\n")

    for i, edf in enumerate(uq_edf_pt,1):
        
        PT_NUM = edf[2:4]
        
        print(f'{i}:{edf}')
        edf_name = edf[:-4]
        
        df_edf = df[df['edf_name']==edf]
        print(df_edf)
               
        edf_filepath = os.path.join(home_dir, edf[0:4], edf)
        raw=mne.io.read_raw_edf(edf_filepath, preload=True)
    
        
        # Data extraction ----------------[start] -----------------
        
        # Specific 19 Channels selection and re-arrange 
        raw.rename_channels({name: name.lower() for name in raw.info['ch_names']}) # renamming channels 
        
        all_ch_flag = [ch for ch in Channel_list  if not ch in raw.ch_names]
        
        if len(all_ch_flag)==0:
            raw = raw.pick_channels(Channel_list)
            raw.reorder_channels(Channel_list)
            raw.rename_channels(rename_dict)
            print(f'Monopolar Channels: {raw.ch_names}')
        else: 
            print(f'Missing Channels: {all_ch_flag}')
            continue
    
        print('\n\n')

        NotchFrq = 50.0    
        raw.notch_filter(freqs=NotchFrq, fir_window = 'hann')
    
        if prepross=='y':
            # LowpassFilt, HighpassFilt = 1., 30.
            # raw = raw.copy().filter(LowpassFilt, HighpassFilt, fir_window = 'hann')

            # Apply highpass filter (Lowcut frequency)
            raw = raw.copy().filter(l_freq=LowpassFilt, h_freq=None, fir_window='hann')

            # Apply lowpass filter (Highcut frequency)
            raw = raw.copy().filter(l_freq=None, h_freq=HighpassFilt, fir_window='hann')

            # downsampling to 500 Hz same as Juntendo
            raw.resample(sfreq=Down_FREQ, npad='auto')  
            print(f'raw: {raw}')
    
        
    
        if stdz =='y':
            print('\n# Standardization Process ---------------[Start]')
            # raw_data = raw.get_data(Channel_list)
            raw_data = raw.get_data()
            print('raw_prepros_data.shape: ', raw_data.shape)
            print('\n')
        
            Mean = np.mean(raw_data, axis=1, keepdims=True)
            STD  = np.std(raw_data, axis=1, keepdims=True)
            print('Mean.shape: ', Mean.shape)
            print('STD.shape: ', STD.shape)
        
            # raw_prepros_data_Norm = np.transpose((np.transpose(raw_prepros_data)-Mean)/STD)
            raw_prepros_data_Norm = (raw_data-Mean)/STD
            print('raw_Norm.shape: ', raw_prepros_data_Norm.shape)
            print('\n\n')
        
            raw._data =  raw_prepros_data_Norm
            print('# Standardization Process ---------------[END]\n')
    
        # -------------------
        
        epochs = make_fixed_length_epochs( raw, 
                                           duration=duration, 
                                           overlap=duration - interval, 
                                           verbose=False)
    
    
        len_ch = len(epochs.info['ch_names'])
        print(f"epochs.info['ch_names']: {len_ch}: {epochs.info['ch_names']}")
        # epoch_data = epochs.get_data(bipol_channel_list)
        epoch_data = epochs.get_data(New_Channel_list)
        



        # Label extraction ----------------[start] -----------------
        start_time = np.array(df_edf['sz_start'])*Down_FREQ
        end_time   = np.array(df_edf['sz_end'])*Down_FREQ
    
        print(f'sz_start: {start_time}')
        print(f'sz_end:   {end_time}')
        print('\n\n')

        length = raw.get_data().shape[1]
        print('length: ', length)
        criteria = np.zeros(length)

        # Labeling seizure time from second to samples  
        for start, end in zip(start_time, end_time):
            criteria[int(start):int(end)] = np.ones(int(end)-int(start))
    
        LABEL = criteria.copy()
        print('LABEL.shape: ', LABEL.shape)
    
        criteria = criteria.reshape(1,-1) # record label 
        info = mne.create_info(ch_names=['label'], sfreq=Down_FREQ)
    
        Label_raw = mne.io.RawArray(criteria, info)
        print(f'Label_raw: {Label_raw}')  

        # label to Label_epochs ------------------------------
        Label_epochs = make_fixed_length_epochs(Label_raw, 
                                                duration=duration, 
                                                overlap=duration - interval, 
                                                verbose=False)
        Label_data = Label_epochs.get_data()
        print('Label_data.shape: ', Label_data.shape)
        Label_data = np.squeeze(Label_data)
        # print('Label_data.shape[1]: ', Label_data.shape[1])

        Label_per = np.sum(Label_data == 1, axis=1)/Label_data.shape[1]

        # Label extraction ----------------[Stop] -----------------

        


        #--------------------------------------------------------------------------------
        save_h5_path  = os.path.join(h5_DIR,  PT_NUM +'_'+ edf_name+'.h5')


        print('\n\n# Matching Data Shape ----------------------')
        
        print('epoch_data.shape: ', epoch_data.shape)
        print('Label_data.shape: ', Label_data.shape)
        print('Label_per.shape: ',  Label_per.shape)

        if epoch_data.shape[0]!=Label_data.shape[0]:
            print('Alert!!!!  Shape mismatch!!!')
            ep_label_shape_flag[edf_name] = 1
        else: 
            ep_label_shape_flag[edf_name] = 0

        # Save into a single HDF5 file
        with h5py.File(save_h5_path, "w") as hf:
            hf.create_dataset("data", data=epoch_data)
            hf.create_dataset("label", data=Label_data)
            hf.create_dataset("labelper", data=Label_per)

            print(f"{save_h5_path} saved successfully!")


