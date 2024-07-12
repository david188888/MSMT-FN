import torch
import torchaudio
import numpy as np
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor
import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.ch_model import rob_hub_cme

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = {
    "batch_size": 2,
    "learning_rate": 5e-6,
    "seed": 42,
    "model": "cme",
    "tasks": "MTA",
    "num_hidden_layers": 5,
    "dropout": 0.3,
    'cme_version': 'v3',

}


class ChConfig:
    def __init__(self, **entries):
        self.__dict__.update(entries)


config = ChConfig(**config)
model = rob_hub_cme(config).to(device)
model.load_state_dict(torch.load('checkpoint/loss.pth'))
for param in model.hubert_model.feature_extractor.parameters():
    param.requires_grad = False


model.eval()

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True
)
# tokenize text

audio_dir = 'test/Out'
label_path = 'test/output_7.10.csv'
label = pd.read_csv(label_path)
output_dict = {}

for root, dirs, files in os.walk(audio_dir):
    for file in files:
        if file.endswith(".wav"):
            audio_path = os.path.join(root, file)

            phone_number_old = file.split('.')[0]
        if 'process' in phone_number_old:
            continue
        else:
            phone_number = np.int64(phone_number_old)
            text = label[label['Phone'] ==
                         phone_number]['Transcription'].values[0]

            tokenized_text = tokenizer(
                text,
                max_length=96,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True
            )

            # load audio

            sound, _ = torchaudio.load(audio_path)

            # Convert tensor to numpy array
            sound = sound.numpy()

            if isinstance(sound, np.ndarray):
                sound = torch.tensor(sound)
                # Convert multi-channel audio to single-channel
            soundData = torch.mean(sound, dim=0, keepdim=False)

            # extract audio features
            features = feature_extractor(
                soundData, sampling_rate=16000, max_length=96000,
                return_attention_mask=True, truncation=True, padding="max_length"
            )
            audio_features = torch.tensor(
                np.array(features['input_values']), dtype=torch.float32).squeeze()
            audio_masks = torch.tensor(
                np.array(features['attention_mask']), dtype=torch.long).squeeze()

            input = {
                "text_tokens": tokenized_text["input_ids"],
                "text_masks": tokenized_text["attention_mask"],
                "audio_inputs": audio_features,
                "audio_masks": audio_masks
            }
            text_input_id = []
            text_mask_id = []
            audio_input = []
            audio_mask = []

            text_input_id.append(input['text_tokens'])
            text_mask_id.append(input['text_masks'])
            audio_input.append(input['audio_inputs'])
            audio_mask.append(input['audio_masks'])

            text_input_ids = torch.tensor(
                text_input_id, dtype=torch.long).to(device)
            text_mask_ids = torch.tensor(
                text_mask_id, dtype=torch.long).to(device)
            audio_input = torch.stack(audio_input).to(device)
            audio_mask = torch.stack(audio_mask).to(device)
            output = model(text_input_ids, text_mask_ids,
                           audio_input, audio_mask)

            with open('test/result.txt','a') as f:
                f.writelines(f"The result of {phone_number_old} is {output['M']}"+"\n")