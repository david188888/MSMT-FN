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
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            AddGaussianSNR(min_snr_db=5.0, max_snr_db=40.0, p=0.5)
        ])

    def __call__(self, audio, sample_rate):
        return self.augment(samples=audio, sample_rate=sample_rate)


class TemporalPitchAugmentation:
    def __init__(self):
        self.augment = Compose([
            TimeMask(min_band_part=0.1, max_band_part=0.1, p=0.5),
            PitchShift(min_semitones=-2, max_semitones=2, p=0.5)
        ])

    def __call__(self, audio, sample_rate):
        return self.augment(samples=audio, sample_rate=sample_rate)


class ShiftNormalizeAugmentation:
    def __init__(self, min_shift=-0.2, max_shift=0.2):
        self.augment = Compose([
            Shift(min_shift=min_shift, max_shift=max_shift, p=0.5),
            Normalize(p=0.5)
        ])

    def __call__(self, audio, sample_rate):
        return self.augment(samples=audio, sample_rate=sample_rate)


class AudioAugmentation:
    def __init__(self, min_shift=-0.1, max_shift=0.1):
        self.shift_normalize_augmenter = ShiftNormalizeAugmentation(
            min_shift, max_shift)

    def __call__(self, audio, sample_rate=16000):
        audio = self.shift_normalize_augmenter(audio, sample_rate)
        return audio


if __name__ == '__main__':
    csv_path = "/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/Customer/trainlabel.csv"
    audio_directory = "/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/Customer/train"
    df = pd.read_csv(csv_path)
    
    encoder = LabelEncoder()
    targets = encoder.fit_transform(list(df['label']))
    texts = list(df['text'])
    audio_file_paths = [f"{audio_directory}/{audio_id}.wav" for audio_id in df['audio_id']]
    
    
    gaussian_augment = GaussianAugmentation()
    temporal_pitch_augment = TemporalPitchAugmentation()
    audio_augment = AudioAugmentation()

    # Create lists to store augmented data
    augmented_texts = []
    augmented_targets = []
    augmented_audio_file_paths = []

    # Perform exponential augmentation
    for i in range(len(audio_file_paths)):
        current_audio, _ = torchaudio.load(audio_file_paths[i])
        current_audio = current_audio.detach().numpy()
        print(current_audio)
        label = targets[i]
        num_augmentations = 2
        phone = audio_file_paths[i]
        # 除去尾缀wav
        match = re.match(r'(.*)\.wav', phone)
        if match:
            phone = match.group(1)
        text = texts[i]

        # Store the original data
        augmented_texts.append(text)
        augmented_targets.append(label)
        augmented_audio_file_paths.append(audio_file_paths[i])

        # Perform augmentations
        for _ in range(num_augmentations):
            labelx = np.random.rand(1)
            print(labelx)
            if labelx < 0.3:
                current_audio = gaussian_augment(
                    current_audio, sample_rate=16000)
                augment_type = "gaussian"
            elif labelx > 0.3 and labelx < 0.7:
                current_audio = temporal_pitch_augment(
                    current_audio, sample_rate=16000)
                augment_type = "temporal"
            else:
                current_audio = audio_augment(current_audio)
                augment_type = "shiftnormalize"
            

            # Convert to tensor and save to a temporary file
            temp_path = f"{phone}_{augment_type}.wav"
            dir = "/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/Customer/train_augmented"
            torchaudio.save(f"{dir}/{temp_path}", torch.tensor(current_audio), 16000)
            # torchaudio.save(temp_path, torch.tensor(current_audio), 16000)
            # Store the augmented data
            augmented_texts.append(text)
            augmented_targets.append(label)
            augmented_audio_file_paths.append(temp_path)


            
    # 将增强后的数据保存到csv文件
    augmented_df = pd.DataFrame({
        'audio_id': [i for i in augmented_audio_file_paths],
        'text': augmented_texts,
        'label': augmented_targets
    })

    augmented_df.to_csv("/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/Customer/train_augmented.csv", index=False)
    
    # 将增强后的数据