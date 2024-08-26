import torch
from torch import nn
import torch.utils
import torch.utils.data
import torchaudio
# import audiomentations
from audiomentations import *
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor
from pathlib import Path
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
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


class QA_Dataset(torch.utils.data.Dataset):
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
            feature_size=1, sampling_rate=8000, padding_value=0.0, do_normalize=True, return_attention_mask=True
        )
        self.unique_phones = list(self.phone_groups.indices.keys())

    def __getitem__(self, index):
        phone = self.unique_phones[index]
        # print(f"the phone is {phone}")
        indeces = self.phone_groups.indices[phone]
        threshold = 10
        batches = []
        # print(f"the indeces is {indeces}")
        # extract text feature
        # tokenize text
        tokenized_text = [self.tokenizer(
            str(self.texts[i]),
            max_length=128,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True
        ) for i in indeces]
        text_tokens = torch.stack([torch.tensor(tt["input_ids"]) for tt in tokenized_text]).to(dtype=torch.long)
        # print(f"the torch shape of text_tokens is {text_tokens.shape}")
        text_masks = torch.stack([torch.tensor(tt["attention_mask"]) for tt in tokenized_text]).to(dtype=torch.long)
        
        audio_features = []
        audio_masks = []
        # load audio
        for i in indeces:
            sound, _ = torchaudio.load(self.audio_file_paths[i])
        
            if isinstance(sound, np.ndarray):
                sound = torch.tensor(sound)
        # Convert multi-channel audio to single-channel
            soundData = torch.mean(sound, dim=0, keepdim=False)

        # extract audio features
            features = self.feature_extractor(
                soundData, sampling_rate=8000, max_length=160000,
                return_attention_mask=True, truncation=True, padding="max_length"
            )
            audio_features.append(torch.tensor(np.array(features['input_values']), dtype=torch.float32).squeeze())
            audio_masks.append(torch.tensor(np.array(features['attention_mask']), dtype=torch.long).squeeze())
        audio_features = torch.stack(audio_features)
        audio_masks = torch.stack(audio_masks)
        target = torch.tensor(self.targets[indeces[0]],dtype=torch.float32)
        
        if len(indeces) <= threshold:
            batch = {
                "text_tokens": text_tokens,
                "text_masks": text_masks,
                "audio_inputs": audio_features,
                "audio_masks": audio_masks,
                "targets": target
            }
            batches.append(batch)
        else:
            first_batch = {
                "text_tokens": text_tokens[:threshold,:],
                "text_masks": text_masks[:threshold,:],
                "audio_inputs": audio_features[:threshold,:],
                "audio_masks": audio_masks[:threshold,:],
                "targets": target[:threshold]
            }
            
            second_batch = {
                "text_tokens": text_tokens[threshold:,:],
                "text_masks": text_masks[threshold:,:],
                "audio_inputs": audio_features[threshold:,:],
                "audio_masks": audio_masks[threshold:,:],
                "targets": target[threshold:]
            }
            batches.extend([first_batch, second_batch])
        return batches

    def __len__(self):
        return len(self.unique_phones)
    
    # def collate_fn(batch):
    #     # sample = batch[0]
    #     # organize batch
        
    #     text_tokens = []
    #     text_masks = []
    #     audio_inputs = []
    #     audio_masks = []
    #     targets = []
        
    #     for i in range(len(batch)):
    #         # text
    #         text_tokens.append(batch[i]['text_tokens'])
    #         text_masks.append(batch[i]['text_masks'])
    #         # audio
    #         audio_inputs.append(batch[i]['audio_inputs'])
    #         audio_masks.append(batch[i]['audio_masks'])

    #     # labels
    #         targets.append(batch[i]['targets'])

        
    #     text_tokens = torch.nn.utils.rnn.pad_sequence([tt.clone().detach().to(dtype=torch.long) for tt in text_tokens], batch_first=True)
    #     text_masks = torch.nn.utils.rnn.pad_sequence([tm.clone().detach().to(dtype=torch.long) for tm in text_masks],batch_first=True)
    #     audio_inputs = torch.nn.utils.rnn.pad_sequence([ai.clone().detach() for ai in audio_inputs], batch_first=True)
    #     audio_masks = torch.nn.utils.rnn.pad_sequence([am.clone().detach() for am in audio_masks],batch_first=True)
        
        
    #     # text_tokens = sample['text_tokens'].clone().detach().to(dtype=torch.long).squeeze()
    #     # text_masks = sample['text_masks'].clone().detach().to(dtype=torch.long).squeeze()
    #     # audio_inputs = sample['audio_inputs'].clone().detach().squeeze()
    #     # audio_masks = sample['audio_masks'].clone().detach().squeeze()

        


    #     return {
    #         # text
    #         "text_tokens": text_tokens,
    #         "text_masks": text_masks,
    #         # audio
    #         "audio_inputs": audio_inputs,
    #         "audio_masks": audio_masks,
    #         # labels
    #         "targets": 
    #             torch.tensor(targets, dtype=torch.float32),
    #     }
    

class MELDDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, audio_dir):
        """
        csv_file: 包含对话数据的CSV文件路径。
        audio_dir: 存放音频文件的目录路径。
        """
        df = pd.read_csv(csv_file)
        self.audio_dir = Path(audio_dir)
        encoder = LabelEncoder()
        self.target = encoder.fit_transform(list(df['Emotion']))
        self.text = df['Utterance']
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        self.dialog_group = df.groupby('Dialogue_ID')
        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
        self.audio_file_paths = []
        for index, row in df.iterrows():
            dialog_id = row['Dialogue_ID']
            utterance_id = row['Utterance_ID']
            
            file_name = f"dia{dialog_id}_utt{utterance_id}.wav"
            file_path = os.path.join(self.audio_dir, file_name)
            self.audio_file_paths.append(file_path)
        self.unqiue_dialog_ids = list(self.dialog_group.indices.keys())
        
    def __len__(self):
        return len(self.unqiue_dialog_ids)

    def __getitem__(self, idx):
        threshold = 20  # 定义阈值
        batches = []
        # 获取数据行
        dialog_id = self.unqiue_dialog_ids[idx]
        indeces = self.dialog_group.indices[dialog_id]

        target = torch.tensor([self.target[i] for i in indeces],dtype=torch.float)

        # 处理文本
        tokenized_text = [self.tokenizer(
                str(self.text[i]),            
                max_length = 60,                                
                padding = "max_length",     # Pad to the specified max_length. 
                truncation = True,          # Truncate to the specified max_length. 
                add_special_tokens = True,  # Whether to insert [CLS], [SEP], <s>, etc.   
                return_attention_mask = True            
            ) for i in indeces]  
        
        text_tokens = torch.stack([torch.tensor(tt['input_ids']) for tt in tokenized_text]).to(dtype=torch.long)
        text_masks = torch.stack([torch.tensor(tt['attention_mask']) for tt in tokenized_text]).to(dtype=torch.long)

        
        audio_features = []
        audio_masks = []
        
        for i in indeces:
            sound, _ = torchaudio.load(self.audio_file_paths[i])
            soundData = torch.mean(sound, dim=0, keepdim=False)
            features = self.feature_extractor(soundData, sampling_rate=16000, max_length=96000,return_attention_mask=True,truncation=True, padding="max_length")
            audio_features.append(torch.tensor(np.array(features['input_values']), dtype=torch.float32).squeeze())
            audio_masks.append(torch.tensor(np.array(features['attention_mask']), dtype=torch.long).squeeze())
        
        audio_features = torch.stack(audio_features)
        audio_masks = torch.stack(audio_masks)
        
        # print(f"audio_features shape: {audio_features.shape}")
        # print(f"audio_masks shape: {audio_masks.shape}")
        # print(f"text_tokens shape: {text_tokens.shape}")
        # print(f"text_masks shape: {text_masks.shape}")
        # print(f"Max value in text_inputs: {text_tokens.max()}")
        # print(f"Min value in text_inputs: {text_tokens.min()}")
        
        
        if len(indeces) <= threshold:
            batch = {
                "text_tokens": text_tokens,
                "text_masks": text_masks,
                "audio_inputs": audio_features,
                "audio_masks": audio_masks,
                "targets": target
            }
            batches.append(batch)
        else:
            first_batch = {
                "text_tokens": text_tokens[:threshold,:],
                "text_masks": text_masks[:threshold,:],
                "audio_inputs": audio_features[:threshold,:],
                "audio_masks": audio_masks[:threshold,:],
                "targets": target[:threshold]
            }
            
            second_batch = {
                "text_tokens": text_tokens[threshold:,:],
                "text_masks": text_masks[threshold:,:],
                "audio_inputs": audio_features[threshold:,:],
                "audio_masks": audio_masks[threshold:,:],
                "targets": target[threshold:]
            }
            batches.extend([first_batch, second_batch])
            
        return batches
            
            
            

        
    # def collate_fn(batch):
    #     sample = batch[0]
    #     text_tokens = sample['text_tokens'].clone().detach().to(dtype=torch.long).squeeze()
    #     text_masks = sample['text_masks'].clone().detach().to(dtype=torch.long).squeeze()
    #     audio_inputs = sample['audio_inputs'].clone().detach().squeeze()
    #     audio_masks = sample['audio_masks'].clone().detach().squeeze()
        
    #     return {
    #         "text_tokens": text_tokens,
    #         "text_masks": text_masks,
    #         "audio_inputs": audio_inputs,
    #         "audio_masks": audio_masks,
    #         "targets": torch.tensor(sample['targets'], dtype=torch.float32)
    #     }

class Dataset_sims(torch.utils.data.Dataset):
    # Argument List
    #  csv_path: path to the csv file
    #  audio_directory: path to the audio files
    #  mode: train, test, valid
    
    def __init__(self, csv_path, audio_directory, mode):       
        df = pd.read_csv(csv_path)
        df = df[df['mode']==mode].reset_index()
        
        # store labels
        encoder = LabelEncoder()
        self.targets = encoder.fit_transform(list(df['annotation']))
        
        # store texts
        self.texts = df['text']
        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        
        # store audio
        self.audio_file_paths = []
        
        self.video_id = df.groupby('video_id')
        self.unique_video_ids = list(self.video_id.indices.keys())
        for i in range(0,len(df)):
            clip_id = str(df['clip_id'][i])
            for j in range(4-len(clip_id)):
                clip_id = '0'+clip_id
            file_name = str(df['video_id'][i]) + '/' + clip_id + '.wav'
            file_path = audio_directory + "/" + file_name
            self.audio_file_paths.append(file_path)
      
        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)   
        
        
    def __getitem__(self, index):
        video_id = self.unique_video_ids[index]
        indeces = self.video_id.indices[video_id]
        threshold = 32
        batches = []
        target = torch.tensor([self.targets[i] for i in indeces],dtype=torch.float)
       # extract text features        
        tokenized_text = [self.tokenizer(
            str(self.texts[i]),            
            max_length = 48,                                
            padding = "max_length",     # Pad to the specified max_length. 
            truncation = True,          # Truncate to the specified max_length. 
            add_special_tokens = True,  # Whether to insert [CLS], [SEP], <s>, etc.   
            return_attention_mask = True            
        ) for i in indeces]               
        
        text_tokens = torch.stack([torch.tensor(tt['input_ids']) for tt in tokenized_text]).to(dtype=torch.long)
        text_masks = torch.stack([torch.tensor(tt['attention_mask']) for tt in tokenized_text]).to(dtype=torch.long)
                
                
        audio_features = []
        audio_masks = []
        for i in indeces:
        # extract audio features    
            sound,_ = torchaudio.load(self.audio_file_paths[i])
            soundData = torch.mean(sound, dim=0, keepdim=False)
            features = self.feature_extractor(soundData, sampling_rate=16000, max_length=96000,return_attention_mask=True,truncation=True, padding="max_length")
            audio_features.append(torch.tensor(np.array(features['input_values']), dtype=torch.float32).squeeze())
            audio_masks.append(torch.tensor(np.array(features['attention_mask']), dtype=torch.long).squeeze())
            
        audio_features = torch.stack(audio_features)
        audio_masks = torch.stack(audio_masks)

        
        if len(indeces) <= threshold:
            batch = {
                "text_tokens": text_tokens,
                "text_masks": text_masks,
                "audio_inputs": audio_features,
                "audio_masks": audio_masks,
                "targets": target
            }
            batches.append(batch)
        elif (len(indeces)>threshold and len(indeces)<=2*threshold):
            first_batch = {
                "text_tokens": text_tokens[:threshold,:],
                "text_masks": text_masks[:threshold,:],
                "audio_inputs": audio_features[:threshold,:],
                "audio_masks": audio_masks[:threshold,:],
                "targets": target[:threshold]
            }
            
            second_batch = {
                "text_tokens": text_tokens[threshold:,:],
                "text_masks": text_masks[threshold:,:],
                "audio_inputs": audio_features[threshold:,:],
                "audio_masks": audio_masks[threshold:,:],
                "targets": target[threshold:]
            }
            batches.extend([first_batch, second_batch])
            
        elif (len(indeces)>2*threshold and len(indeces)<=3*threshold):
            first_batch = {
                "text_tokens": text_tokens[:threshold,:],
                "text_masks": text_masks[:threshold,:],
                "audio_inputs": audio_features[:threshold,:],
                "audio_masks": audio_masks[:threshold,:],
                "targets": target[:threshold]
            }
            second_batch = {
                "text_tokens": text_tokens[threshold:2*threshold,:],
                "text_masks": text_masks[threshold:2*threshold,:],
                "audio_inputs": audio_features[threshold:2*threshold,:],
                "audio_masks": audio_masks[threshold:2*threshold,:],
                "targets": target[threshold:2*threshold]
            }
            third_batch = {
                "text_tokens": text_tokens[2*threshold:,:],
                "text_masks": text_masks[2*threshold:,:],
                "audio_inputs": audio_features[2*threshold:,:],
                "audio_masks": audio_masks[2*threshold:,:],
                "targets": target[2*threshold:]
            }
            batches.extend([first_batch, second_batch, third_batch])
            
        elif (len(indeces)>3*threshold and len(indeces)<=4*threshold):
            first_batch = {
                "text_tokens": text_tokens[:threshold,:],
                "text_masks": text_masks[:threshold,:],
                "audio_inputs": audio_features[:threshold,:],
                "audio_masks": audio_masks[:threshold,:],
                "targets": target[:threshold]
            }
            second_batch = {
                "text_tokens": text_tokens[threshold:2*threshold,:],
                "text_masks": text_masks[threshold:2*threshold,:],
                "audio_inputs": audio_features[threshold:2*threshold,:],
                "audio_masks": audio_masks[threshold:2*threshold,:],
                "targets": target[threshold:2*threshold]
            }
            third_batch = {
                "text_tokens": text_tokens[2*threshold:3*threshold,:],
                "text_masks": text_masks[2*threshold:3*threshold,:],
                "audio_inputs": audio_features[2*threshold:3*threshold,:],
                "audio_masks": audio_masks[2*threshold:3*threshold,:],
                "targets": target[2*threshold:3*threshold]
            }
            forth_batch = {
                "text_tokens": text_tokens[3*threshold:,:],
                "text_masks": text_masks[3*threshold:,:],
                "audio_inputs": audio_features[3*threshold:,:],
                "audio_masks": audio_masks[3*threshold:,:],
                "targets": target[3*threshold:]
            }
            batches.extend([first_batch, second_batch, third_batch, forth_batch])
            
        elif (len(indeces)>4*threshold and len(indeces)<=5*threshold):
            first_batch = {
                "text_tokens": text_tokens[:threshold,:],
                "text_masks": text_masks[:threshold,:],
                "audio_inputs": audio_features[:threshold,:],
                "audio_masks": audio_masks[:threshold,:],
                "targets": target[:threshold]
            }
            second_batch = {
                "text_tokens": text_tokens[threshold:2*threshold,:],
                "text_masks": text_masks[threshold:2*threshold,:],
                "audio_inputs": audio_features[threshold:2*threshold,:],
                "audio_masks": audio_masks[threshold:2*threshold,:],
                "targets": target[threshold:2*threshold]
            }
            third_batch = {
                "text_tokens": text_tokens[2*threshold:3*threshold,:],
                "text_masks": text_masks[2*threshold:3*threshold,:],
                "audio_inputs": audio_features[2*threshold:3*threshold,:],
                "audio_masks": audio_masks[2*threshold:3*threshold,:],
                "targets": target[2*threshold:3*threshold]
            }
            forth_batch = {
                "text_tokens": text_tokens[3*threshold:4*threshold,:],
                "text_masks": text_masks[3*threshold:4*threshold,:],
                "audio_inputs": audio_features[3*threshold:4*threshold,:],
                "audio_masks": audio_masks[3*threshold:4*threshold,:],
                "targets": target[3*threshold:4*threshold]
            }
            fivth_batch = {
                "text_tokens": text_tokens[4*threshold:,:],
                "text_masks": text_masks[4*threshold:,:],
                "audio_inputs": audio_features[4*threshold:,:],
                "audio_masks": audio_masks[4*threshold:,:],
                "targets": target[4*threshold:]
            }
            batches.extend([first_batch, second_batch, third_batch, forth_batch, fivth_batch])
            
        elif (len(indeces)>5*threshold and len(indeces)<=6*threshold):
            first_batch = {
                "text_tokens": text_tokens[:threshold,:],
                "text_masks": text_masks[:threshold,:],
                "audio_inputs": audio_features[:threshold,:],
                "audio_masks": audio_masks[:threshold,:],
                "targets": target[:threshold]
            }
            second_batch = {
                "text_tokens": text_tokens[threshold:2*threshold,:],
                "text_masks": text_masks[threshold:2*threshold,:],
                "audio_inputs": audio_features[threshold:2*threshold,:],
                "audio_masks": audio_masks[threshold:2*threshold,:],
                "targets": target[threshold:2*threshold]
            }
            third_batch = {
                "text_tokens": text_tokens[2*threshold:3*threshold,:],
                "text_masks": text_masks[2*threshold:3*threshold,:],
                "audio_inputs": audio_features[2*threshold:3*threshold,:],
                "audio_masks": audio_masks[2*threshold:3*threshold,:],
                "targets": target[2*threshold:3*threshold]
            }
            forth_batch = {
                "text_tokens": text_tokens[3*threshold:4*threshold,:],
                "text_masks": text_masks[3*threshold:4*threshold,:],
                "audio_inputs": audio_features[3*threshold:4*threshold,:],
                "audio_masks": audio_masks[3*threshold:4*threshold,:],
                "targets": target[3*threshold:4*threshold]
            }
            fivth_batch = {
                "text_tokens": text_tokens[4*threshold:5*threshold,:],
                "text_masks": text_masks[4*threshold:5*threshold,:],
                "audio_inputs": audio_features[4*threshold:5*threshold,:],
                "audio_masks": audio_masks[4*threshold:5*threshold,:],
                "targets": target[4*threshold:5*threshold]
            }
            sixth_batch = {
                "text_tokens": text_tokens[5*threshold:,:],
                "text_masks": text_masks[5*threshold:,:],
                "audio_inputs": audio_features[5*threshold:,:],
                "audio_masks": audio_masks[5*threshold:,:],
                "targets": target[5*threshold:]
            }
            batches.extend([first_batch, second_batch, third_batch, forth_batch, fivth_batch, sixth_batch])
            
        else:
            first_batch = {
                "text_tokens": text_tokens[:threshold,:],
                "text_masks": text_masks[:threshold,:],
                "audio_inputs": audio_features[:threshold,:],
                "audio_masks": audio_masks[:threshold,:],
                "targets": target[:threshold]
            }
            second_batch = {
                "text_tokens": text_tokens[threshold:2*threshold,:],
                "text_masks": text_masks[threshold:2*threshold,:],
                "audio_inputs": audio_features[threshold:2*threshold,:],
                "audio_masks": audio_masks[threshold:2*threshold,:],
                "targets": target[threshold:2*threshold]
            }
            third_batch = {
                "text_tokens": text_tokens[2*threshold:3*threshold,:],
                "text_masks": text_masks[2*threshold:3*threshold,:],
                "audio_inputs": audio_features[2*threshold:3*threshold,:],
                "audio_masks": audio_masks[2*threshold:3*threshold,:],
                "targets": target[2*threshold:3*threshold]
            }
            forth_batch = {
                "text_tokens": text_tokens[3*threshold:4*threshold,:],
                "text_masks": text_masks[3*threshold:4*threshold,:],
                "audio_inputs": audio_features[3*threshold:4*threshold,:],
                "audio_masks": audio_masks[3*threshold:4*threshold,:],
                "targets": target[3*threshold:4*threshold]
            }
            
            fivth_batch = {
                "text_tokens": text_tokens[4*threshold:5*threshold,:],
                "text_masks": text_masks[4*threshold:5*threshold,:],
                "audio_inputs": audio_features[4*threshold:5*threshold,:],
                "audio_masks": audio_masks[4*threshold:5*threshold,:],
                "targets": target[4*threshold:5*threshold]
            }
            sixth_batch = {
                "text_tokens": text_tokens[5*threshold:6*threshold,:],
                "text_masks": text_masks[5*threshold:6*threshold,:],
                "audio_inputs": audio_features[5*threshold:6*threshold,:],
                "audio_masks": audio_masks[5*threshold:6*threshold,:],
                "targets": target[5*threshold:6*threshold]
            }
            
            seventh_batch = {
                "text_tokens": text_tokens[5*threshold:,:],
                "text_masks": text_masks[5*threshold:,:],
                "audio_inputs": audio_features[5*threshold:,:],
                "audio_masks": audio_masks[5*threshold:,:],
                "targets": target[5*threshold:]
            }
            batches.extend([first_batch, second_batch,third_batch,forth_batch,fivth_batch,sixth_batch,seventh_batch])
        return batches
    
    def __len__(self):
        return len(self.unique_video_ids)



class Dataset_mosi(torch.utils.data.Dataset):
    # Argument List
    #  csv_path: path to the csv file
    #  audio_directory: path to the audio files
    #  mode: train, test, valid
    #  text_context_length
    #  audio_context_length
    
    def __init__(self, csv_path, audio_directory, mode):
        df = pd.read_csv(csv_path)
        invalid_files = ['3aIQUQgawaI/12.wav', '94ULum9MYX0/2.wav', 'mRnEJOLkhp8/24.wav', 'aE-X_QdDaqQ/3.wav', '94ULum9MYX0/11.wav', 'mRnEJOLkhp8/26.wav']
        for f in invalid_files:
            video_id = f.split('/')[0]
            clip_id = f.split('/')[1].split('.')[0]
            df = df[~((df['video_id']==video_id) & (df['clip_id']==int(clip_id)))]

        df = df[df['mode']==mode].sort_values(by=['video_id','clip_id']).reset_index()
        
        # store labels
        encoder = LabelEncoder()
        # df['annotation'] = df['annotation'].apply(lambda x: 'Negative' if x in ['Negative', 'Neutral'] else 'positive')
        # df = df[df['annotation'] != 'Neutral']
        # df = df.reset_index(drop=True)
        # self.targets = encoder.fit_transform(list(df['annotation']))
        # print("encoder classes: ", encoder.classes_)
        self.targets = df['label']
        
        # store texts
        df['text'] = df['text'].str[0]+df['text'].str[1::].apply(lambda x: x.lower())
        self.texts = df['text']
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")

        # store audio
        self.audio_file_paths = []
        ## loop through the csv entries
        for i in range(0,len(df)):
            file_name = str(df['video_id'][i])+'/'+str(df['clip_id'][i])+'.wav'
            file_path = audio_directory + "/" + file_name
            self.audio_file_paths.append(file_path)
        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

        self.video_id = df.groupby('video_id')
        self.unique_video_ids = list(self.video_id.indices.keys())
        
    def __getitem__(self, index):
        video_id = self.unique_video_ids[index]
        indeces = self.video_id.indices[video_id]       
        threshold = 20
        batches = []
        # tokenize text
        tokenized_text = [self.tokenizer(
                str(self.texts[i]),            
                max_length = 96,                                
                padding = "max_length",     # Pad to the specified max_length. 
                truncation = True,          # Truncate to the specified max_length. 
                add_special_tokens = True,  # Whether to insert [CLS], [SEP], <s>, etc.   
                return_attention_mask = True            
            ) for i in indeces]
        text_tokens = torch.stack([torch.tensor(tt['input_ids']) for tt in tokenized_text])
        text_masks = torch.stack([torch.tensor(tt['attention_mask']) for tt in tokenized_text])
        
        audio_features = []
        audio_masks = []
        for i in indeces:
            # load audio
            sound, _ = torchaudio.load(self.audio_file_paths[i])
            soundData = torch.mean(sound, dim=0, keepdim=False)
        # extract audio features
            features = self.feature_extractor(soundData, sampling_rate=16000, max_length=96000,return_attention_mask=True,truncation=True, padding="max_length")
            audio_features.append(torch.tensor(np.array(features['input_values']), dtype=torch.float32).squeeze())
            audio_masks.append(torch.tensor(np.array(features['attention_mask']), dtype=torch.long).squeeze())
        audio_features = torch.stack(audio_features)
        audio_masks = torch.stack(audio_masks)
        target = torch.tensor([self.targets[i] for i in indeces],dtype=torch.float)


        if len(indeces) <= threshold:
            batch = {
                "text_tokens": text_tokens,
                "text_masks": text_masks,
                "audio_inputs": audio_features,
                "audio_masks": audio_masks,
                "targets": target
            }
            batches.append(batch)
        else:
            if (len(indeces)>2*threshold and len(indeces)<=3*threshold):
                first_batch = {
                    "text_tokens": text_tokens[:threshold,:],
                    "text_masks": text_masks[:threshold,:],
                    "audio_inputs": audio_features[:threshold,:],
                    "audio_masks": audio_masks[:threshold,:],
                    "targets": target[:threshold]
                }
                second_batch = {
                    "text_tokens": text_tokens[threshold:2*threshold,:],
                    "text_masks": text_masks[threshold:2*threshold,:],
                    "audio_inputs": audio_features[threshold:2*threshold,:],
                    "audio_masks": audio_masks[threshold:2*threshold,:],
                    "targets": target[threshold:2*threshold]
                }
                third_batch = {
                    "text_tokens": text_tokens[2*threshold:,:],
                    "text_masks": text_masks[2*threshold:,:],
                    "audio_inputs": audio_features[2*threshold:,:],
                    "audio_masks": audio_masks[2*threshold:,:],
                    "targets": target[2*threshold:]
                }
                batches.extend([first_batch, second_batch, third_batch])
                
                
            elif (len(indeces)>3*threshold and len(indeces)<=4*threshold):
                first_batch = {
                    "text_tokens": text_tokens[:threshold,:],
                    "text_masks": text_masks[:threshold,:],
                    "audio_inputs": audio_features[:threshold,:],
                    "audio_masks": audio_masks[:threshold,:],
                    "targets": target[:threshold]
                } 
                
                second_batch = {
                    "text_tokens": text_tokens[threshold:2*threshold,:],
                    "text_masks": text_masks[threshold:2*threshold,:],
                    "audio_inputs": audio_features[threshold:2*threshold,:],
                    "audio_masks": audio_masks[threshold:2*threshold,:],
                    "targets": target[threshold:2*threshold]
                }
                
                third_batch = {
                    "text_tokens": text_tokens[2*threshold:3*threshold,:],
                    "text_masks": text_masks[2*threshold:3*threshold,:],
                    "audio_inputs": audio_features[2*threshold:3*threshold,:],
                    "audio_masks": audio_masks[2*threshold:3*threshold,:],
                    "targets": target[2*threshold:3*threshold]
                }
                
                forth_batch = {
                    "text_tokens": text_tokens[3*threshold:,:],
                    "text_masks": text_masks[3*threshold:,:],
                    "audio_inputs": audio_features[3*threshold:,:],
                    "audio_masks": audio_masks[3*threshold:,:],
                    "targets": target[3*threshold:]
                }
                
                batches.extend([first_batch, second_batch, third_batch, forth_batch])
            
            elif (len(indeces)>4*threshold):
                first_batch = {
                    "text_tokens": text_tokens[:threshold,:],
                    "text_masks": text_masks[:threshold,:],
                    "audio_inputs": audio_features[:threshold,:],
                    "audio_masks": audio_masks[:threshold,:],
                    "targets": target[:threshold]
                }
                second_batch = {
                    "text_tokens": text_tokens[threshold:2*threshold,:],
                    "text_masks": text_masks[threshold:2*threshold,:],
                    "audio_inputs": audio_features[threshold:2*threshold,:],
                    "audio_masks": audio_masks[threshold:2*threshold,:],
                    "targets": target[threshold:2*threshold]
                }
                third_batch = {
                    "text_tokens": text_tokens[2*threshold:3*threshold,:],
                    "text_masks": text_masks[2*threshold:3*threshold,:],
                    "audio_inputs": audio_features[2*threshold:3*threshold,:],
                    "audio_masks": audio_masks[2*threshold:3*threshold,:],
                    "targets": target[2*threshold:3*threshold]
                }
                forth_batch = {
                    "text_tokens": text_tokens[3*threshold:4*threshold,:],
                    "text_masks": text_masks[3*threshold:4*threshold,:],
                    "audio_inputs": audio_features[3*threshold:4*threshold,:],
                    "audio_masks": audio_masks[3*threshold:4*threshold,:],
                    "targets": target[3*threshold:4*threshold]
                }
                five_batch = {
                    "text_tokens": text_tokens[4*threshold:,:],
                    "text_masks": text_masks[4*threshold:,:],
                    "audio_inputs": audio_features[4*threshold:,:],
                    "audio_masks": audio_masks[4*threshold:,:],
                    "targets": target[4*threshold:]
                }
                batches.extend([first_batch, second_batch, third_batch, forth_batch,five_batch])
                
            else:
                first_batch = {
                    "text_tokens": text_tokens[:threshold,:],
                    "text_masks": text_masks[:threshold,:],
                    "audio_inputs": audio_features[:threshold,:],
                    "audio_masks": audio_masks[:threshold,:],
                    "targets": target[:threshold]
                }
                
                second_batch = {
                    "text_tokens": text_tokens[threshold:,:],
                    "text_masks": text_masks[threshold:,:],
                    "audio_inputs": audio_features[threshold:,:],
                    "audio_masks": audio_masks[threshold:,:],
                    "targets": target[threshold:]
                }
                batches.extend([first_batch, second_batch])
        return batches
    
    def __len__(self):
        return len(self.unique_video_ids)



def data_loader(batch_size):
    
    train_label_path = '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/MELD/data/train.csv'
    test_label_path = '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/MELD/data/test.csv'
    verify_label_path = '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/MELD/data/verify.csv'
    
    label_path = '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/mosei/moseilabel.csv'
    
    train_file_path = '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/MELD/data/train_splits_wav'
    test_file_path = '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/MELD/data/output_repeated_splits_test_wav'
    verify_file_path = '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/MELD/data/dev_splits_complete_wav'
    
    file_path = '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/mosei/wav'
    
    train_data = Dataset_mosi(label_path, file_path, mode='train')
    test_data = Dataset_mosi(label_path, file_path, mode='test')
    val_data = Dataset_mosi(label_path, file_path, mode='valid')
    
    # train_data = Dataset_sims(label_path, file_path, mode='train')
    # test_data = Dataset_sims(label_path, file_path, mode='test')
    # val_data = Dataset_sims(label_path, file_path, mode='valid')
    
    # train_data = MELDDataset(train_label_path, train_file_path)
    # test_data = MELDDataset(test_label_path, test_file_path)
    # val_data = MELDDataset(verify_label_path, verify_file_path)
    
    
    train_loader = DataLoader(
        train_data, batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(
        test_data,  batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_data,  batch_size=batch_size, shuffle=True)
    return train_loader, test_loader, val_loader