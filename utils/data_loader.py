import torch
from torch import nn
import transformers
import torchaudio
# import audiomentations
from audiomentations import *
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor
from torch.utils.data import DataLoader, BatchSampler
import random
import pandas as pd
import numpy as np
import soundfile as sf
import os


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


class DynamicBatchSampler(BatchSampler):
    def __init__(self,label_csv_path,shuffle=False):
        self.audio_directory = label_csv_path
        self.shuffle = shuffle
        
        df = pd.read_csv(label_csv_path)
        
        #创建一个从电话号码到所有对话段的索引的列表的映射
        self.phone_to_dialogue = {}
        phone_list = list(df['phone'])
        for idx, phone in enumerate(phone_list):
            if phone not in self.phone_to_dialogue:
                self.phone_to_dialogue[phone] = []
            self.phone_to_dialogue[phone].append(idx
            )
            
            
    def __iter__(self):
        dialogue_keys = list(self.phone_to_dialogue.keys())
        if self.shuffle:
            # 如果启用洗牌，随机打乱电话号码键列表
            random.shuffle(dialogue_keys)
        
        for key in dialogue_keys:
            segment_indices = self.phone_to_dialogue[key]
            print(f"Yielding batch for {key} with size {len(segment_indices)}")
            yield segment_indices
            
    def __len__(self):
        return len(self.phone_to_dialogue)


class Dataset_audio_text(torch.utils.data.Dataset):
    def __init__(self, csv_path, audio_directory):
        df = pd.read_csv(csv_path)

        # store the label and text
        encoder = LabelEncoder()
        self.targets = encoder.fit_transform(list(df['label']))
        self.texts = list(df['text'])
        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

        # store the audio
        self.audio_file_paths = [f"{audio_directory}/{audio_id}" for audio_id in df['audio_id']]
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True
        )
        

    def __getitem__(self, index):
        # extract text features
        text = str(self.texts[index])

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
        sound, _ = torchaudio.load(self.audio_file_paths[index])
        
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
                "M": self.targets[index]
            }
        }

    def __len__(self):
        return len(self.targets)


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


def data_loader():
    # csv_path = 'data/labell.csv'
    # audio_file_path = "data/OnlyCustomer"
    # data = Dataset_audio_text(csv_path, audio_file_path)
    # train_data, test_data, val_data = torch.utils.data.random_split(
    #     data, [int(0.8*len(data)), int(0.1*len(data)), len(data)-int(0.8*len(data))-int(0.1*len(data))])
    train_label_path = '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/qalabel.csv'
    test_label_path = '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/Origin/testlabel.csv'
    verify_label_path = '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/Origin/verifylabel.csv'
    
    train_file_path = '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/output_segments'
    test_file_path = '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/Origin/test'
    verify_file_path = '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/Origin/verify'
    
    

    
    
    train_data = Dataset_audio_text(train_label_path, train_file_path)
    test_data = Dataset_audio_text(test_label_path, test_file_path)
    val_data = Dataset_audio_text(verify_label_path, verify_file_path)
    
    batch_sampler_train = DynamicBatchSampler(train_label_path,shuffle= True)
    # batch_sampler_test = DynamicBatchSampler(test_label_path)
    # batch_sampler_verify = DynamicBatchSampler(verify_label_path)
    
    train_loader = DataLoader(
        train_data, batch_sampler=batch_sampler_train, collate_fn=collate_fn_sims)
    # test_loader = DataLoader(
    #     test_data, batch_sampler=batch_sampler_test, collate_fn=collate_fn_sims)
    # val_loader = DataLoader(val_data, batch_sampler=batch_sampler_verify,
    #                         collate_fn=collate_fn_sims)
    return train_loader