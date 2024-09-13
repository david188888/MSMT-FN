import os
from audiomentations import *
import torchaudio
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import re

class GaussianAugmentation:
    def __init__(self):
        self.augment = Compose([
            AddGaussianNoise(min_amplitude=0.003, max_amplitude=0.01, p=1.0),
            AddGaussianSNR(min_snr_db=10.0, max_snr_db=40.0, p=1.0)
        ])
    def __call__(self, audio, sample_rate):
        return self.augment(samples=audio, sample_rate=sample_rate)

class TemporalPitchAugmentation:
    def __init__(self):
        self.augment = Compose([
            TimeMask(min_band_part=0.1, max_band_part=0.1, p=1.0),
            PitchShift(min_semitones=-1.2, max_semitones=1.2, p=1.0)
        ])
    def __call__(self, audio, sample_rate):
        return self.augment(samples=audio, sample_rate=sample_rate)

class ShiftNormalizeAugmentation:
    def __init__(self, min_shift=-0.2, max_shift=0.2):
        self.augment = Compose([
            Shift(min_shift=min_shift, max_shift=max_shift, p=1.0),
            Normalize(p=0.1)
        ])
    def __call__(self, audio, sample_rate):
        return self.augment(samples=audio, sample_rate=sample_rate)

class AudioAugmentation:
    def __init__(self, min_shift=-0.2, max_shift=0.2):
        self.shift_normalize_augmenter = ShiftNormalizeAugmentation(
            min_shift, max_shift)
    def __call__(self, audio, sample_rate):
        audio = self.shift_normalize_augmenter(audio, sample_rate)
        return audio

if __name__ == '__main__':
    csv_path = "/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/mosi/mosi_label.csv"
    audio_directory = "/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/mosi/wav"
    df = pd.read_csv(csv_path)
    
    df = df[df['mode']=='train'].reset_index()
    
    
    texts = list(df['text'])
    video_ids = list(df['video_id'])
    clip_ids = list(df['clip_id'])
    annotation = list(df['annotation'])
    targets = list(df['label'])
    targets_T = list(df['label_T'])
    targets_A = list(df['label_A'])
    targets_V = list(df['label_V'])
    mode = list(df['mode'])
    audio_file_paths = [os.path.join(audio_directory, str(video_id), f"{clip_id}.wav") for video_id, clip_id in zip(video_ids, clip_ids)]
    
    gaussian_augment = GaussianAugmentation()
    temporal_pitch_augment = TemporalPitchAugmentation()
    audio_augment = AudioAugmentation()
    
    # Create lists to store augmented data
    augmented_texts = []
    augmented_annotation = []
    augmented_audio_file_paths = []
    augmented_video_ids = []
    augmented_clip_ids = []
    augmented_labels = []
    augmented_label_T = []
    augmented_label_A = []
    augmented_label_V = []
    augmented_mode = []
    
    # Perform augmentations
    for i in range(len(audio_file_paths)):

        current_audio, _ = torchaudio.load(audio_file_paths[i])
        current_audio = current_audio.detach().numpy()
        
        label = targets[i]
        label_A = targets_A[i]
        label_T = targets_T[i]
        label_V = targets_V[i]
        mode_ = mode[i]
        annotations = annotation[i]
        video_id = video_ids[i]
        clip_id = clip_ids[i]
        audio_name = audio_file_paths[i]
        filename = re.search(r'[^/]+$', audio_name).group()
        match = re.match(r'(.*)\.wav', audio_name)
        if match:
            audio_name = match.group(1)
        text = texts[i]
        
        # Store the original data
        augmented_texts.append(text)
        augmented_labels.append(label)
        augmented_audio_file_paths.append(filename)
        augmented_video_ids.append(video_id)
        augmented_clip_ids.append(clip_id)
        augmented_label_A.append(label_A)
        augmented_label_T.append(label_T)
        augmented_label_V.append(label_V)
        augmented_mode.append(mode_)
        augmented_annotation.append(annotations)
        
        # Perform augmentations
        current_audio_1 = gaussian_augment(current_audio, sample_rate=16000)
        augment_type_1 = "gaussian"

        current_audio_2 = temporal_pitch_augment(current_audio, sample_rate=16000)
        augment_type_2 = "temporal"
        
        current_audio_3 = audio_augment(current_audio, 16000)
        augment_type_3 = "shiftnormalize"
        
        temp_path_1 = f"{audio_name}_{augment_type_1}.wav"
        temp_path_2 = f"{audio_name}_{augment_type_2}.wav"
        temp_path_3 = f"{audio_name}_{augment_type_3}.wav"
        
        torchaudio.save(f"{temp_path_1}", torch.tensor(current_audio_1), 16000)
        torchaudio.save(f"{temp_path_2}", torch.tensor(current_audio_2), 16000)
        torchaudio.save(f"{temp_path_3}", torch.tensor(current_audio_3), 16000)
        
        # Store the augmented data
        for j in range(1, 4):
            augmented_texts.append(text)
            augmented_labels.append(label)
            augmented_audio_file_paths.append(filename)
            augmented_video_ids.append(video_id)
            augmented_clip_ids.append(clip_id)
            augmented_label_A.append(label_A)
            augmented_label_T.append(label_T)
            augmented_label_V.append(label_V)
            augmented_mode.append(mode_)
            augmented_annotation.append(annotations)
    
    print('Augmentation finished')
    
    # 将增强后的数据保存到csv文件
    augmented_df = pd.DataFrame({
        'video_id': augmented_video_ids,
        'clip_id': augmented_clip_ids,
        'text': augmented_texts,
        'label': augmented_labels,
        'label_T': augmented_label_T,
        'label_A': augmented_label_A,
        'label_V': augmented_label_V,
        'mode': augmented_mode,
        'annotation': augmented_annotation
    })
    augmented_df.to_csv("/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/mosi/mosi_augment.csv", index=False)
