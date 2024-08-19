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
        text_tokens = torch.stack([torch.Tensor(tt["input_ids"]) for tt in tokenized_text]).to(dtype=torch.long)
        # print(f"the torch shape of text_tokens is {text_tokens.shape}")
        text_masks = torch.stack([torch.Tensor(tt["attention_mask"]) for tt in tokenized_text]).to(dtype=torch.long)
        
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
        target = self.targets[indeces[0]]
        
        return {
            "text_tokens": text_tokens,
            "text_masks": text_masks,
            "audio_inputs": audio_features,
            "audio_masks": audio_masks,
            "targets": target
        }

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
        self.audio_id = df['Dialogue_ID']
        encoder = LabelEncoder()
        self.target = encoder.fit_transform(list(df['label']))
        self.text = df['Utterance']
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        self.dialog_group = df.groupby('Dialogue_ID')
        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
        self.audio_file_paths = []
        for i in range(len(df)):
            self.audio_file_paths.append(self.audio_dir / f"{df['Dialogue_ID'][i]}_{df['Utterance_ID'][i]}.wav")
        self.unqiue_dialog_ids = list(self.dialog_group.indices.keys())
        
    def __len__(self):
        return len(self.unqiue_dialog_ids)

    def __getitem__(self, idx):
        # 获取数据行
        dialog_id = self.unqiue_dialog_ids[idx]
        indeces = self.dialog_group.indices[dialog_id]

        target = [self.target[i] for i in indeces]

        # 处理文本
        tokenized_text = [self.tokenizer(
                str(self.text[i]),            
                max_length = 96,                                
                padding = "max_length",     # Pad to the specified max_length. 
                truncation = True,          # Truncate to the specified max_length. 
                add_special_tokens = True,  # Whether to insert [CLS], [SEP], <s>, etc.   
                return_attention_mask = True            
            ) for i in indeces]  
        
        text_tokens = torch.stack([torch.Tensor(tt['input_ids']) for tt in tokenized_text])
        text_masks = torch.stack([torch.Tensor(tt['attention_mask']) for tt in tokenized_text])

        
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
        
        return {
            "text_tokens": text_tokens,
            "text_masks": text_masks,
            "audio_inputs": audio_features,
            "audio_masks": audio_masks,
            "targets": target
        }
        
    def collate_fn(batch):
        sample = batch[0]
        text_tokens = sample['text_tokens'].clone().detach().to(dtype=torch.long).squeeze()
        text_masks = sample['text_masks'].clone().detach().to(dtype=torch.long).squeeze()
        audio_inputs = sample['audio_inputs'].clone().detach().squeeze()
        audio_masks = sample['audio_masks'].clone().detach().squeeze()
        
        return {
            "text_tokens": text_tokens,
            "text_masks": text_masks,
            "audio_inputs": audio_inputs,
            "audio_masks": audio_masks,
            "targets": torch.tensor(sample['targets'], dtype=torch.float32)
        }

class Dataset_sims(torch.utils.data.Dataset):
    # Argument List
    #  csv_path: path to the csv file
    #  audio_directory: path to the audio files
    #  mode: train, test, valid
    
    def __init__(self, csv_path, audio_directory, mode):       
        df = pd.read_csv(csv_path)
        df = df[df['mode']==mode].reset_index()
        
        # store labels
        self.targets = df['annotation']
        
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
        target = [self.targets[i] for i in indeces]
       # extract text features        
        tokenized_text = [self.tokenizer(
            str(self.texts[i]),            
            max_length = 64,                                
            padding = "max_length",     # Pad to the specified max_length. 
            truncation = True,          # Truncate to the specified max_length. 
            add_special_tokens = True,  # Whether to insert [CLS], [SEP], <s>, etc.   
            return_attention_mask = True            
        ) for i in indeces]               
        
        text_tokens = torch.stack([torch.Tensor(tt['input_ids']) for tt in tokenized_text])
        text_masks = torch.stack([torch.Tensor(tt['attention_mask']) for tt in tokenized_text])
                
                
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

        
        return { # text
                "text_tokens": text_tokens,
                "text_masks": text_masks,
                 # audio
                "audio_inputs": audio_features,
                "audio_masks": audio_masks,
                 # labels
                "targets": target
                }
    
    def __len__(self):
        return len(self.unique_video_ids)
    
    # def collate_fn_sims(self, batch):   
        
    #     sample = batch[0]
    #     text_tokens = sample['text_tokens'].clone().detach().to(dtype=torch.long).squeeze()
    #     text_masks = sample['text_masks'].clone().detach().to(dtype=torch.long).squeeze()
    #     audio_inputs = sample['audio_inputs'].clone().detach().squeeze()
    #     audio_masks = sample['audio_masks'].clone().detach().squeeze()
    #     # organize batch
    #     # for i in range(len(batch)):
    #     #     # text
    #     #     text_tokens.append(batch[i]['text_tokens'])
    #     #     text_masks.append(batch[i]['text_masks'])
    #     #     #audio
    #     #     audio_inputs.append(batch[i]['audio_inputs'])
    #     #     audio_masks.append(batch[i]['audio_masks'])

    #     # # labels
    #     #     targets_M.append(batch[i]['target'])
            
    #     # text_tokens = torch.nn.utils.rnn.pad_sequence([tt.clone().detach().to(dtype=torch.long) for tt in text_tokens], batch_first=True)
    #     # text_masks = torch.nn.utils.rnn.pad_sequence([tm.clone().detach().to(dtype=torch.long) for tm in text_masks],batch_first=True)
    #     # audio_inputs = torch.nn.utils.rnn.pad_sequence([ai.clone().detach() for ai in audio_inputs], batch_first=True)
    #     # audio_masks = torch.nn.utils.rnn.pad_sequence([am.clone().detach() for am in audio_masks],batch_first=True)

    #     return {
    #             # text
    #             "text_tokens": text_tokens,
    #             "text_masks": text_masks,           
    #             # audio
    #             "audio_inputs": audio_inputs,
    #             "audio_masks": audio_masks,
    #             # labels
    #             "targets": {
    #                     torch.tensor(sample['target'], dtype=torch.float32),
    #                 }
    #             }   



class Dataset_mosi(torch.utils.data.Dataset):
    # Argument List
    #  csv_path: path to the csv file
    #  audio_directory: path to the audio files
    #  mode: train, test, valid
    #  text_context_length
    #  audio_context_length
    
    def __init__(self, csv_path, audio_directory, mode, text_context_length=2, audio_context_length=1):
        df = pd.read_csv(csv_path)
        invalid_files = ['3aIQUQgawaI/12.wav', '94ULum9MYX0/2.wav', 'mRnEJOLkhp8/24.wav', 'aE-X_QdDaqQ/3.wav', '94ULum9MYX0/11.wav', 'mRnEJOLkhp8/26.wav']
        for f in invalid_files:
            video_id = f.split('/')[0]
            clip_id = f.split('/')[1].split('.')[0]
            df = df[~((df['video_id']==video_id) & (df['clip_id']==int(clip_id)))]

        df = df[df['mode']==mode].sort_values(by=['video_id','clip_id']).reset_index()
        
        # store labels
        self.targets = df['annotation']
        
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

        # tokenize text
        tokenized_text = [self.tokenizer(
                str(self.texts[i]),            
                max_length = 96,                                
                padding = "max_length",     # Pad to the specified max_length. 
                truncation = True,          # Truncate to the specified max_length. 
                add_special_tokens = True,  # Whether to insert [CLS], [SEP], <s>, etc.   
                return_attention_mask = True            
            ) for i in indeces]
        text_tokens = torch.stack([torch.Tensor(tt['input_ids']) for tt in tokenized_text])
        text_masks = torch.stack([torch.Tensor(tt['attention_mask']) for tt in tokenized_text])
        
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
        target = [self.target[i] for i in indeces]


        return { # text
                "text_tokens": text_tokens,
                "text_masks": text_masks,
                # audio
                "audio_inputs": audio_features,
                "audio_masks": audio_masks,
                 # labels
                "targets": target,
                }
    
    def __len__(self):
        return len(self.unique_video_ids)


    def collate_fn(self,batch):
        sample = batch[0]
        text_tokens = sample['text_tokens'].clone().detach().to(dtype=torch.long).squeeze()
        text_masks = sample['text_masks'].clone().detach().to(dtype=torch.long).squeeze()
        audio_inputs = sample['audio_inputs'].clone().detach().squeeze()
        audio_masks = sample['audio_masks'].clone().detach().squeeze()
        
        return {
            "text_tokens": text_tokens,
            "text_masks": text_masks,
            "audio_inputs": audio_inputs,
            "audio_masks": audio_masks,
            "targets": torch.tensor(sample['targets'], dtype=torch.float32)
        }



def data_loader(batch_size):
    # csv_path = 'data/labell.csv'
    # audio_file_path = "data/OnlyCustomer"
    # data = QA_Dataset(csv_path, audio_file_path)
    # train_data, test_data, val_data = torch.utils.data.random_split(
    #     data, [int(0.8*len(data)), int(0.1*len(data)), len(data)-int(0.8*len(data))-int(0.1*len(data))])
    train_label_path = '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/qa_data/train/dialog_test_11.csv'
    test_label_path = '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/qa_data/test/dialog_test_11.csv'
    verify_label_path = '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/qa_data/verify/dialog_test_11.csv'
    
    train_file_path = '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/qa_data/train/dialog'
    test_file_path = '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/qa_data/test/dialog'
    verify_file_path = '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/qa_data/verify/dialog'
    
    train_data = QA_Dataset(train_label_path, train_file_path)
    test_data = QA_Dataset(test_label_path, test_file_path)
    val_data = QA_Dataset(verify_label_path, verify_file_path)
    
    
    train_loader = DataLoader(
        train_data, batch_size=batch_size)
    test_loader = DataLoader(
        test_data,  batch_size=batch_size)
    val_loader = DataLoader(val_data,  batch_size=batch_size)
    return train_loader, test_loader, val_loader