import torch
import torchaudio
import numpy as np
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor
import os
from torch.nn.functional import softmax
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.ch_model import rob_hub_cme
from torchinfo import summary
from torchviz import make_dot

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 配置类
class ChConfig:
    def __init__(self, **entries):
        self.__dict__.update(entries)

# 数据处理类
class DataProcessor:
    def __init__(self, tokenizer, feature_extractor):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

    def process_audios(self, audio_paths):
        audios = []
        for audio_path in audio_paths:
            sound, _ = torchaudio.load(audio_path)
            sound = torch.mean(sound, dim=0)
            audios.append(sound)
        return audios

    def process_texts(self, texts):
        tokenized_texts = [self.tokenizer(text, max_length=512, padding="max_length",
                                          truncation=True, add_special_tokens=True,
                                          return_attention_mask=True) for text in texts]
        return tokenized_texts

    def extract_features(self, sounds):
        features_list = [self.feature_extractor(sound, sampling_rate=16000, max_length=96000,
                                                return_attention_mask=True, truncation=True, padding="max_length") for sound in sounds]
        return features_list

# 模型管理器类
class ModelManager:
    def __init__(self, config, model_path):
        self.model = rob_hub_cme(config).to(device)
        # self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        for param in self.model.hubert_model.feature_extractor.parameters():
            param.requires_grad = False

    def predict(self, text_input_ids, text_mask_ids, audio_inputs, audio_masks):
        outputs = self.model(text_input_ids, text_mask_ids, audio_inputs, audio_masks)
        probabilities = softmax(outputs, dim=1)
        return probabilities.detach().cpu().numpy()

# 初始化
config = ChConfig(num_hidden_layers=5, dropout=0.3, cme_version='v3')
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
data_processor = DataProcessor(tokenizer, feature_extractor)
model_manager = ModelManager(config, '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/checkpoint/fusion_enhance_acc_3.pth')
labels = ['A', 'B', 'C', 'D', 'E']

# 批处理预测逻辑
def batch_predict(audio_paths, texts):
    # 处理文本和音频
    tokenized_texts = data_processor.process_texts(texts)
    sounds = data_processor.process_audios(audio_paths)
    features_list = data_processor.extract_features(sounds)
    
    
    # 转换成Tensor并堆叠
    text_input_ids = torch.tensor([item['input_ids'] for item in tokenized_texts],dtype=torch.long).to(device)
    text_mask_ids = torch.tensor([item['attention_mask'] for item in tokenized_texts],dtype=torch.long).to(device)
    audio_inputs = torch.stack([torch.tensor(item['input_values'], dtype=torch.float32).squeeze() for item in features_list]).to(device)
    audio_masks = torch.stack([torch.tensor(item['attention_mask'], dtype=torch.long).squeeze() for item in features_list]).to(device)

    
    outputs = model_manager.predict(text_input_ids, text_mask_ids, audio_inputs, audio_masks)
    predicted_labels = np.argmax(outputs, axis=1)
    return predicted_labels, text_input_ids, text_mask_ids, audio_inputs, audio_masks



# 示例批处理调用
audio_path1 = '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/test/Out/13320998597.wav'
audio_path2 = '/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/test/Out/13654582675.wav'
audio_paths = [audio_path1, audio_path2]  # 音频路径列表
texts = ["喂你好，打扰您，我们是口腔医院的，请问是否有蛀牙、缺牙或者牙齿不齐等口腔问题需要解决的？有。如果您牙齿有蛀牙断裂缺失等问题，可以来我院做个免费的检查，帮您了解自身口腔情况，可以帮您预约一下吗？好。啊噢您现在是在哪个地方？呢广州。广州哪里？呢我们广州的话广博是有30家院区的。我不是你们广州的，我是成都的", "嗯您好，打扰您了，我们广东韩非3月开年集美节有380万现金券免费领取，同时上门可以领取面膜补水，舒敏、中医调理、肩颈宝等项目免费体验。亲，您之前咨询的项目现在还有在做了解吗？没有了，谢谢，我不了解了，哈谢谢啊再见。整形美容对皮肤的问题以及面部五官轮廓可以有效快速改善，医美会比一般的护肤品更快更有效地看到效果，您可以来找我们院做个免费的visa皮肤检测，以及五官轮廓的面诊，详细了解您的变美需求，帮您预约一下权威的专家面诊可以吗？我这个我这个不是皮肤的问题，我这个是骨头的问题，我觉得要做的话就是做那个正颌手术。好的，我这边添加您的微信，把最近的活动发送给您，可以吗？"]





# 相应的文本列表
predicted_labels, text_input_ids, text_mask_ids, audio_inputs, audio_masks = batch_predict(audio_paths, texts)
result = pd.DataFrame({'audio_path': audio_paths, 'text': texts, 'label': [labels[i] for i in predicted_labels]})
# print(result)
summary(model_manager.model, input_data=(text_input_ids, text_mask_ids, audio_inputs, audio_masks))


