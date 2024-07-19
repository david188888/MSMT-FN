import torch
from torch import nn
import transformers
import torchaudio
# import audiomentations
from audiomentations import *
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import soundfile as sf


# class AudioAugmentation:
#     def __init__(self,min_shift=-0.5, max_shift=0.5):
#         self.augment = audiomentations.Compose([
#             audiomentations.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
#             audiomentations.AddGaussianSNR(min_snr_db=5.0, max_snr_db=40.0, p=0.5),
#             audiomentations.TimeMask(min_band_part=0.1, max_band_part=0.5, p=0.5),
#             audiomentations.PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
#             audiomentations.Shift(min_shift==-0.5, max_shift==0.5, p=0.5),
#             audiomentations.Normalize(p=0.5)
#         ])

#     def __call__(self, audio):
#         return self.augment(audio, sample_rate=16000)

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
        self.shift_normalize_augmenter = ShiftNormalizeAugmentation(min_shift, max_shift)

    def __call__(self, audio, sample_rate=16000):
        audio = self.shift_normalize_augmenter(audio, sample_rate)
        return audio

import pandas as pd
import numpy as np
import torch
import torchaudio
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor

class Dataset_audio_text(torch.utils.data.Dataset):
    def __init__(self, csv_path, audio_directory, augment=False, num_augmentations=2):
        df = pd.read_csv(csv_path)

        # store the label and text
        encoder = LabelEncoder()
        self.targets = encoder.fit_transform(list(df['label']))
        self.texts = list(df['text'])
        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

        # store the audio
        self.audio_file_paths = [f"{audio_directory}/{audio_id}.wav" for audio_id in df['audio_id']]
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True
        )

        self.augment = False
        self.num_augmentations = num_augmentations

        if self.augment:
            self.gaussian_augment = GaussianAugmentation()
            self.temporal_pitch_augment = TemporalPitchAugmentation()
            self.audio_augment = AudioAugmentation()

            # Create lists to store augmented data
            self.augmented_texts = []
            self.augmented_targets = []
            self.augmented_audio_file_paths = []

            # Perform exponential augmentation
            for i in range(len(self.audio_file_paths)):
                current_audio, _ = torchaudio.load(self.audio_file_paths[i])
                current_audio = current_audio.numpy()
                label = self.targets[i]
                
                text = self.texts[i]

                # Store the original data
                self.augmented_texts.append(text)
                self.augmented_targets.append(label)
                self.augmented_audio_file_paths.append(self.audio_file_paths[i])

                # Perform augmentations
                for _ in range(self.num_augmentations):
                    labelx = np.random.rand(1)
                    if labelx <0.3:
                        current_audio = self.gaussian_augment(current_audio, sample_rate=16000)
                    elif labelx >0.3 and labelx < 0.7:
                        current_audio = self.temporal_pitch_augment(current_audio, sample_rate=16000)
                    else:
                        current_audio = self.audio_augment(current_audio)

                    # Convert to tensor and save to a temporary file
                    temp_path = f"temp/aug_{i}_{_}.wav"
                    torchaudio.save(temp_path, torch.tensor(current_audio), 16000)

                    # Store the augmented data
                    self.augmented_texts.append(text)
                    self.augmented_targets.append(label)
                    self.augmented_audio_file_paths.append(temp_path)

        else:
            self.augmented_texts = self.texts
            self.augmented_targets = self.targets
            self.augmented_audio_file_paths = self.audio_file_paths

    def __getitem__(self, index):
        # extract text features
        text = str(self.augmented_texts[index])

        # tokenize text
        tokenized_text = self.tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True
        )
        # print(tokenized_text)

        # load audio
        sound, _ = torchaudio.load(self.augmented_audio_file_paths[index])
        
        # Convert tensor to numpy array
        sound = sound.numpy()

        if isinstance(sound, np.ndarray):
            sound = torch.tensor(sound)
        # Convert multi-channel audio to single-channel
        soundData = torch.mean(sound, dim=0, keepdim=False)

        # extract audio features
        features = self.feature_extractor(
            soundData, sampling_rate=16000, max_length=96000,
            return_attention_mask=True, truncation=True, padding="max_length"
        )
        audio_features = torch.tensor(np.array(features['input_values']), dtype=torch.float32).squeeze()
        audio_masks = torch.tensor(np.array(features['attention_mask']), dtype=torch.long).squeeze()

        return {
            "text_tokens": tokenized_text["input_ids"],
            "text_masks": tokenized_text["attention_mask"],
            "audio_inputs": audio_features,
            "audio_masks": audio_masks,
            "target": {
                "M": self.augmented_targets[index]
            }
        }

    def __len__(self):
        return len(self.augmented_targets)


def collate_fn_sims(batch):
    text_tokens = []
    text_masks = []
    audio_inputs = []
    audio_masks = []

    targets_M = []

    # organize batch
    for i in range(len(batch)):
        # text
        text_tokens.append(batch[i]['text_tokens'])
        text_masks.append(batch[i]['text_masks'])
        # audio
        audio_inputs.append(batch[i]['audio_inputs'])
        audio_masks.append(batch[i]['audio_masks'])

       # labels
        targets_M.append(batch[i]['target']['M'])

    return {
        # text
        "text_tokens": torch.tensor(text_tokens, dtype=torch.long),
        "text_masks": torch.tensor(text_masks, dtype=torch.long),
        # audio
        "audio_inputs": torch.stack(audio_inputs),
        "audio_masks": torch.stack(audio_masks),
        # labels
        "targets": 
            torch.tensor(targets_M, dtype=torch.float32),
    }


def data_loader(batch_size):
    # csv_path = 'data/labell.csv'
    # audio_file_path = "data/OnlyCustomer"
    # data = Dataset_audio_text(csv_path, audio_file_path)
    # train_data, test_data, val_data = torch.utils.data.random_split(
    #     data, [int(0.8*len(data)), int(0.1*len(data)), len(data)-int(0.8*len(data))-int(0.1*len(data))])
    train_label_path = '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/Origin/trainlabel.csv'
    test_label_path = '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/Origin/testlabel.csv'
    verify_label_path = '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/Origin/verifylabel.csv'
    
    train_file_path = '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/Origin/train'
    test_file_path = '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/Origin/test'
    verify_file_path = '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/Origin/verify'
    
    
    train_data = Dataset_audio_text(train_label_path, train_file_path, augment=True)
    test_data = Dataset_audio_text(test_label_path, test_file_path, augment=False)
    val_data = Dataset_audio_text(verify_label_path, verify_file_path, augment=False)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, collate_fn=collate_fn_sims, shuffle=True)
    test_loader = DataLoader(
        test_data, batch_size=batch_size, collate_fn=collate_fn_sims, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size,
                            collate_fn=collate_fn_sims, shuffle=False)
    return train_loader, test_loader, val_loader
