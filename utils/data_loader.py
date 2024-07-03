import torch
from torch import nn
import transformers
import torchaudio
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np


class Dataset_audio_text(torch.utils.data.Dataset):
    # Argument List
    #  csv_path: path to the csv file
    #  audio_directory: path to the audio files

    def __init__(self, csv_path, audio_directory):
        df = pd.read_csv(csv_path)

        # store the label and text
        self.targets = df['label']
        self.texts = df['text']
        self.tokenizer = AutoTokenizer.from_pretrained(
            "hfl/chinese-roberta-wwm-ext")

        # store the audio
        self.audio_file_paths = []
        for i in range(0, len(df)):
            file_name = str(df['audio_id'][i])+'.wav'
            file_path = audio_directory + "/" + file_name
            self.audio_file_paths.append(file_path)
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

        self.audio_id = df['audio_id']

    def __getitem__(self, index):
        # extract text features
        text = str(self.texts[index])

        # tokenize text
        tokenized_text = self.tokenizer(
            text,
            max_length=96,
            padding="max_length",     # Pad to the specified max_length.
            truncation=True,          # Truncate to the specified max_length.
            # Whether to insert [CLS], [SEP], <s>, etc.
            add_special_tokens=True,
            return_attention_mask=True
        )

        # load audio
        sound, _ = torchaudio.load(self.audio_file_paths[index])
        soundData = torch.mean(sound, dim=0, keepdim=False)

        # extract audio features
        features = self.feature_extractor(soundData, sampling_rate=16000, max_length=96000,
                                          return_attention_mask=True, truncation=True, padding="max_length")
        audio_features = torch.tensor(
            np.array(features['input_values']), dtype=torch.float32).squeeze()
        audio_masks = torch.tensor(
            np.array(features['attention_mask']), dtype=torch.long).squeeze()

        return {  # text
            "text_tokens": (tokenized_text["input_ids"]),
            "text_masks": (tokenized_text["attention_mask"]),
            # audio
            "audio_inputs": audio_features,
            "audio_masks": audio_masks,
            # labels
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
        "targets": {
            "M": torch.tensor(targets_M, dtype=torch.float32),
        }
    }


def data_loader(batch_size):
    csv_path = 'data/label.csv'
    audio_file_path = "data/audio"
    data = Dataset_audio_text(csv_path, audio_file_path)
    train_data, test_data, val_data = torch.utils.data.random_split(
        data, [int(0.8*len(data)), int(0.1*len(data)), int(0.1*len(data))])
    train_loader = DataLoader(
        train_data, batch_size=batch_size, collate_fn=collate_fn_sims, shuffle=True)
    test_loader = DataLoader(
        test_data, batch_size=batch_size, collate_fn=collate_fn_sims, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size,
                            collate_fn=collate_fn_sims, shuffle=False)
    return train_loader, test_loader, val_loader
