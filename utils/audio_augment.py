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
    csv_path = "/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/qa_new_data/dialog_train.csv"
    audio_directory = "/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/qa_new_data/train_no_silence"
    df = pd.read_csv(csv_path)
    
    encoder = LabelEncoder()
    targets = encoder.fit_transform(list(df['label']))
    texts = list(df['text'])
    phones = list(df['phone'])
    clips_id = list(df['Segment_number'])
    audio_file_paths = [f"{audio_directory}/{audio_id}" for audio_id in df['Audio_id']]
    
    
    gaussian_augment = GaussianAugmentation()
    temporal_pitch_augment = TemporalPitchAugmentation()
    audio_augment = AudioAugmentation()

    # Create lists to store augmented data
    augmented_texts = []
    augmented_targets = []
    augmented_audio_file_paths = []
    augmented_phones = []
    augmented_clips = []

    # Perform exponential augmentation
    for i in range(len(audio_file_paths)):
        current_audio, _ = torchaudio.load(audio_file_paths[i])
        current_audio = current_audio.detach().numpy()
        
        label = targets[i]
        phone = phones[i]

        clip_id = clips_id[i]
        audio_name = audio_file_paths[i]
        filename = re.search(r'[^/]+$', audio_name).group()
        match = re.match(r'(.*)\.wav', audio_name)
        if match:
            audio_name = match.group(1)
        text = texts[i]



        # Store the original data
        augmented_texts.append(text)
        augmented_targets.append(label)
        augmented_audio_file_paths.append(filename)
        augmented_phones.append(phone)
        augmented_clips.append(clip_id)

        # Perform augmentations
        if label != 3:
                current_audio_1 = gaussian_augment(
                    current_audio, sample_rate=8000)
                augment_type_1 = "gaussian"

                current_audio_2 = temporal_pitch_augment(
                    current_audio, sample_rate=8000)
                augment_type_2 = "temporal"
                
                current_audio_3 = audio_augment(current_audio, 8000)
                augment_type_3 = "shiftnormalize"
                
                temp_path_1 = f"{audio_name}_{augment_type_1}.wav"
                temp_path_2 = f"{audio_name}_{augment_type_2}.wav"
                temp_path_3 = f"{audio_name}_{augment_type_3}.wav"
                
                # dir = "/home/lhy-scnu/mmml/verify/verify_augmented"
                torchaudio.save(f"{temp_path_1}", torch.tensor(current_audio_1), 8000)
                torchaudio.save(f"{temp_path_2}", torch.tensor(current_audio_2), 8000)
                torchaudio.save(f"{temp_path_3}", torch.tensor(current_audio_3), 8000)

                # Store the augmented data
                for i in range(1,4):
                    augmented_texts.append(text)
                    augmented_targets.append(label)
                    augmented_phones.append(phone)
                    augmented_clips.append(clip_id)
                    temp_path = eval(f"temp_path_{i}")
                    augmented_audio_file_paths.append(temp_path)
                    
        else:
            current_audio = gaussian_augment(current_audio,8000)
            temp_path = f"{audio_name}_gaussian.wav"
            augment_type = "gaussian"
            torchaudio.save(temp_path, torch.tensor(current_audio), 8000)
            
            augmented_texts.append(text)
            augmented_phones.append(phone)
            augmented_clips.append(clip_id)
            augmented_targets.append(label)
            augmented_audio_file_paths.append(temp_path)
        
            # Convert to tensor and save to a temporary file


    print('Augmentation finished')


    # 将增强后的数据保存到csv文件
    augmented_df = pd.DataFrame({
        'phone': augmented_phones,
        'audio_id': augmented_audio_file_paths,
        'segment_number': augmented_clips,
        'text': augmented_texts,
        'label': augmented_targets
    })
    augmented_df.to_csv("/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/qa_new_data/train_augment.csv", index=False)