import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_loader import data_loader




train_loader= data_loader(batch_size = 8)

print(len(train_loader))

for i , batch in enumerate(train_loader):
    print(f"the shape of text_tokens: {batch['text_tokens'].shape}")
    print(f"the shape of text_masks: {batch['text_masks'].shape}")
    print(f"the shape of audio_inputs: {batch['audio_inputs'].shape}")
    print(f"the shape of audio_masks: {batch['audio_masks'].shape}")