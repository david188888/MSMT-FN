import torch
from torch import nn
import torch.utils
import torch.utils.data
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
        

    def __getitem__(self, index):
        phone = list(self.phone_groups.indices.keys())[index]
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
        
        text_tokens = torch.nn.utils.rnn.pad_sequence([tt.clone().detach().to(dtype=torch.long) for tt in text_tokens], batch_first=True)
        text_masks = torch.nn.utils.rnn.pad_sequence([tm.clone().detach().to(dtype=torch.long) for tm in text_masks],batch_first=True)
        audio_inputs = torch.nn.utils.rnn.pad_sequence([ai.clone().detach() for ai in audio_inputs], batch_first=True)
        audio_masks = torch.nn.utils.rnn.pad_sequence([am.clone().detach() for am in audio_masks],batch_first=True)
        
        
        text_inputs = text_tokens.view(text_tokens.size(0)*text_tokens.size(1), text_tokens.size(2))
        text_masks = text_masks.view(text_masks.size(0)*text_masks.size(1), text_masks.size(2))
        audio_inputs = audio_inputs.view(audio_inputs.size(0)*audio_inputs.size(1), audio_inputs.size(2))
        audio_masks = audio_masks.view(audio_masks.size(0)*audio_masks.size(1), audio_masks.size(2))

        

        return {
            # text
            "text_tokens": text_inputs,
            "text_masks": text_masks,
            # audio
            "audio_inputs": audio_inputs,
            "audio_masks": audio_masks,
            # labels
            "targets": 
                torch.tensor(targets_M, dtype=torch.float32),
        }


class IEMOCAPDataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.roberta2, self.roberta3, self.roberta4, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open('data/iemocap_multimodal_features.pkl', 'rb'), encoding='latin1')
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]


class MELDDataset(torch.utils.data.Dataset):
    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.roberta2, self.roberta3, self.roberta4, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid, _ = pickle.load(open(path, 'rb'))

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)
        
        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
        self.teokenizer = AutoTokenizer.from_pretrained("roberta-large")
        
    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor(self.videoSpeakers[vid]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def return_labels(self):
        return_label = []
        for key in self.keys:
            return_label+=self.videoLabels[key]
        return return_label

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]

class Dataset_sims(torch.utils.data.Dataset):
    # Argument List
    #  csv_path: path to the csv file
    #  audio_directory: path to the audio files
    #  mode: train, test, valid
    
    def __init__(self, csv_path, audio_directory, mode):       
        df = pd.read_csv(csv_path)
        df = df[df['mode']==mode].reset_index()
        
        # store labels
        self.targets_M = df['label']
        
        # store texts
        self.texts = df['text']
        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        
        # store audio
        self.audio_file_paths = []

        for i in range(0,len(df)):
            clip_id = str(df['clip_id'][i])
            for j in range(4-len(clip_id)):
                clip_id = '0'+clip_id
            file_name = str(df['video_id'][i]) + '/' + clip_id + '.wav'
            file_path = audio_directory + "/" + file_name
            self.audio_file_paths.append(file_path)
      
        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)   
        
        
    def __getitem__(self, index):
       # extract text features
        text = str(self.texts[index])         
        tokenized_text = self.tokenizer(
            text,            
            max_length = 64,                                
            padding = "max_length",     # Pad to the specified max_length. 
            truncation = True,          # Truncate to the specified max_length. 
            add_special_tokens = True,  # Whether to insert [CLS], [SEP], <s>, etc.   
            return_attention_mask = True            
        )               
                
        # extract audio features    
        sound,_ = torchaudio.load(self.audio_file_paths[index])
        soundData = torch.mean(sound, dim=0, keepdim=False)
        features = self.feature_extractor(soundData, sampling_rate=16000, max_length=96000,return_attention_mask=True,truncation=True, padding="max_length")
        audio_features = torch.tensor(np.array(features['input_values']), dtype=torch.float32).squeeze()
        audio_masks = torch.tensor(np.array(features['attention_mask']), dtype=torch.long).squeeze()
            
        return { # text
                "text_tokens": tokenized_text["input_ids"],
                "text_masks": tokenized_text["attention_mask"],
                 # audio
                "audio_inputs": audio_features,
                "audio_masks": audio_masks,
                 # labels
                "target": {
                    "M": self.targets_M[index],
                }
                }
    
    def __len__(self):
        return len(self.targets_M)
    
    def collate_fn_sims(batch):   
        text_tokens = []  
        text_masks = []
        audio_inputs = []  
        audio_masks = []
        
        targets_M = []
        targets_T = []
        targets_A = []
    
        # organize batch
        for i in range(len(batch)):
            # text
            text_tokens.append(batch[i]['text_tokens'])
            text_masks.append(batch[i]['text_masks'])
            #audio
            audio_inputs.append(batch[i]['audio_inputs'])
            audio_masks.append(batch[i]['audio_masks'])

        # labels
            targets_M.append(batch[i]['target']['M'])
            targets_T.append(batch[i]['target']['T'])
            targets_A.append(batch[i]['target']['A'])        
        
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
                        "T": torch.tensor(targets_T, dtype=torch.float32),
                        "A": torch.tensor(targets_A, dtype=torch.float32)
                    }
                }   



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
        self.targets_M = df['annotation']
        
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

        # store context
        self.video_id = df['video_id']
        
    def __getitem__(self, index):
        # load text
        text = str(self.texts[index])             

        # tokenize text
        tokenized_text = self.tokenizer(
                text,            
                max_length = 96,                                
                padding = "max_length",     # Pad to the specified max_length. 
                truncation = True,          # Truncate to the specified max_length. 
                add_special_tokens = True,  # Whether to insert [CLS], [SEP], <s>, etc.   
                return_attention_mask = True            
            )  

        # load audio
        sound,_ = torchaudio.load(self.audio_file_paths[index])
        soundData = torch.mean(sound, dim=0, keepdim=False)


        # extract audio features
        features = self.feature_extractor(soundData, sampling_rate=16000, max_length=96000,return_attention_mask=True,truncation=True, padding="max_length")
        audio_features = torch.tensor(np.array(features['input_values']), dtype=torch.float32).squeeze()
        audio_masks = torch.tensor(np.array(features['attention_mask']), dtype=torch.long).squeeze()


        return { # text
                "text_tokens": torch.tensor(tokenized_text["input_ids"], dtype=torch.long),
                "text_masks": torch.tensor(tokenized_text["attention_mask"], dtype=torch.long),
                # audio
                "audio_inputs": audio_features,
                "audio_masks": audio_masks,
                 # labels
                "targets": torch.tensor(self.targets_M[index], dtype=torch.float),
                }
    
    def __len__(self):
        return len(self.targets_M)


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
        train_data, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(
        test_data,  batch_size=batch_size, collate_fn=collate_fn)
    val_loader = DataLoader(val_data,  batch_size=batch_size,
                            collate_fn=collate_fn)
    return train_loader, test_loader, val_loader