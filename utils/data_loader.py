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



# class DynamicBatchSampler(BatchSampler):
#     def __init__(self,label_csv_path,shuffle=False):
#         self.audio_directory = label_csv_path
#         self.shuffle = shuffle
        
#         df = pd.read_csv(label_csv_path)
        
#         #创建一个从电话号码到所有对话段的索引的列表的映射
#         self.phone_to_dialogue = {}
#         phone_list = list(df['phone'])
#         for idx, phone in enumerate(phone_list):
#             if phone not in self.phone_to_dialogue:
#                 self.phone_to_dialogue[phone] = []
#             self.phone_to_dialogue[phone].append(idx
#             )
            
            
#     def __iter__(self):
#         dialogue_keys = list(self.phone_to_dialogue.keys())
#         if self.shuffle:
#             # 如果启用洗牌，随机打乱电话号码键列表
#             random.shuffle(dialogue_keys)
        
#         for key in dialogue_keys:
#             segment_indices = self.phone_to_dialogue[key]
#             print(f"Yielding batch for {key} with size {len(segment_indices)}")
#             yield segment_indices
            
#     def __len__(self):
#         return len(self.phone_to_dialogue)


class Dataset_audio_text(torch.utils.data.Dataset):
    def __init__(self, csv_path, audio_directory):
        df = pd.read_csv(csv_path)

        # store the label and text
        encoder = LabelEncoder()
        self.targets = encoder.fit_transform(list(df['label']))
        self.phone_groups = df.groupby('phone')
        self.texts = df['text'].tolist()
        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

        # store the audio
        self.audio_file_paths = [f"{audio_directory}/{audio_id}" for audio_id in df['Audio_id']]
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True
        )
        

    def __getitem__(self, index):
        phone = list(self.phone_groups.indices.keys())[index]
        indeces = self.phone_groups.indices[phone]
        # print(f"the indeces is {indeces}")
        # extract text feature
        # tokenize text
        tokenized_text = [self.tokenizer(
            str(self.texts[i]),
            max_length=96,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True
        ) for i in indeces]
        text_tokens = torch.stack([torch.Tensor(tt["input_ids"]) for tt in tokenized_text])
        # print(f"the torch shape of text_tokens is {text_tokens.shape}")
        text_masks = torch.stack([torch.Tensor(tt["attention_mask"]) for tt in tokenized_text])
        
        audio_features = []
        audio_masks = []
        # load audio
        for i in indeces:
            sound, _ = torchaudio.load(self.audio_file_paths[i])
        
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
            audio_features.append(torch.tensor(np.array(features['input_values']), dtype=torch.float32).squeeze())
            audio_masks.append(torch.tensor(np.array(features['attention_mask']), dtype=torch.long).squeeze())
        audio_features = torch.stack(audio_features)
        audio_masks = torch.stack(audio_masks)
        target = self.targets[indeces[0]]
        
        

        return {
            "text_tokens": text_tokens,
            "text_masks": text_masks,
            "audio_inputs": audio_features,
            "audio_masks": audio_masks,
            "target": {
                "M": target
            }
        }

    def __len__(self):
        return len(self.phone_groups)


def collate_fn(batch):
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
        
    # print(f"size of text_tokens: {len(text_tokens), len(text_tokens[0]), len(text_tokens[0][0])}")
    # print(f"size of audio_inputs: {len(audio_inputs), len(audio_inputs[0]), len(audio_inputs[0][0])}")
    # print(f"targets_M: {targets_M}")
    
    text_tokens = [torch.tensor(tt, dtype=torch.long) for tt in text_tokens]
    text_masks = [torch.tensor(tm, dtype=torch.long) for tm in text_masks]
    audio_inputs = [torch.tensor(ai) for ai in audio_inputs]
    audio_masks = [torch.tensor(am) for am in audio_masks]

    return {
        # text
        "text_tokens": torch.nn.utils.rnn.pad_sequence(text_tokens, batch_first=True),
        "text_masks": torch.nn.utils.rnn.pad_sequence(text_masks, batch_first=True),
        # audio
        "audio_inputs": torch.nn.utils.rnn.pad_sequence(audio_inputs, batch_first=True),
        "audio_masks": torch.nn.utils.rnn.pad_sequence(audio_masks, batch_first=True),
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
    train_label_path = '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/qa_data/train/dialog_test_11.csv'
    test_label_path = '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/qa_data/test/dialog_test_11.csv'
    verify_label_path = '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/qa_data/verify/dialog_test_11.csv'
    
    train_file_path = '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/qa_data/train/dialog'
    test_file_path = '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/qa_data/test/dialog'
    verify_file_path = '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/qa_data/verify/dialog'
    
    train_data = Dataset_audio_text(train_label_path, train_file_path)
    test_data = Dataset_audio_text(test_label_path, test_file_path)
    val_data = Dataset_audio_text(verify_label_path, verify_file_path)
    
    
    train_loader = DataLoader(
        train_data, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(
        test_data,  batch_size=batch_size, collate_fn=collate_fn)
    val_loader = DataLoader(val_data,  batch_size=batch_size,
                            collate_fn=collate_fn)
    return train_loader, test_loader, val_loader