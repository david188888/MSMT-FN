import torch
from torch import nn
from transformers import RobertaModel, HubertModel, AutoModel, Data2VecAudioModel
from utils.cross_attn_encoder import CMELayer, AttnConfig, GRU_context, GruConfig, Bottleneck, FCLayer
# from positional_encodings.torch_encodings import PositionalEncodingPermute1D, Summer
import torch.nn.functional as F
import gc
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
class rob_hub_cme(nn.Module):            
    def __init__(self, config):        
        super().__init__()

        # load text pre-trained model
        self.roberta_model = AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext")

        # load audio pre-trained model
        self.hubert_model = AutoModel.from_pretrained('TencentGameMate/chinese-hubert-base')
        
        # self.roberta_model = RobertaModel.from_pretrained('roberta-base')
        # self.data2vec_model = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base")
        
        # cls embedding layers
        self.text_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)
        self.audio_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)
        self.text_mixed_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768*2)
        self.audio_mixed_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768*2)
        
        self.version = config.version


        # CME layers
        Attn_config = AttnConfig(num_hidden_layers=config.num_hidden_layers,n_bottlenecks=config.n_bottlenecks,bottleneck_layers=config.bottleneck_layers)
        self.CME_layers = nn.ModuleList(
            [CMELayer(Attn_config) for _ in range(Attn_config.num_hidden_layers)]
        )

        self.Bottelenck_layer = nn.ModuleList(
            [Bottleneck(Attn_config) for _ in range(Attn_config.bottleneck_layers)]
        )
        if Attn_config.use_bottleneck:
            # self.bottleneck = nn.Parameter(torch.randn(
            #     1, Attn_config.n_bottlenecks, Attn_config.hidden_size) * 0.02)
            self.bottleneck = nn.Parameter(torch.empty(
                        1, Attn_config.n_bottlenecks, Attn_config.hidden_size))
            nn.init.xavier_normal_(self.bottleneck)
            self.bottleneck = self.bottleneck.to(dtype=torch.float32)

        
        GRU_config = GruConfig(hidden_size=config.hidden_size_gru, num_layers=config.num_layers_gru)
        self.GRU_layers = GRU_context(GRU_config)
        
        self.fc_layer = FCLayer(config)
        # multi-head attention
        # self.multi_head_attn = nn.MultiheadAttention(embed_dim=768, num_heads=12, dropout=config.dropout)
        
        
        
        self.inter_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768*2, 768),
            nn.ReLU(),                
            )

        
        
        # last linear output layer
        self.fused_output_layers = nn.Sequential(
                nn.Dropout(config.dropout),
                # nn.Linear(config.hidden_size_gru*2, 128),
                # nn.ReLU(),
                # nn.Linear(128, 2),
                
                nn.Linear(768,512),
                nn.ReLU(),
                nn.Linear(512, 5),
            )
        
        self.four_class_layer  = nn.Sequential(
                nn.Dropout(config.dropout),
                # nn.Linear(config.hidden_size_gru*2, 128),
                nn.Linear(768,512),
                nn.ReLU(),
                nn.Linear(512, 4),
            )
        
        self.three_class_layer  = nn.Sequential(
                nn.Dropout(config.dropout),
                # nn.Linear(config.hidden_size_gru*2, 128),
                nn.Linear(768,512),
                nn.ReLU(),
                nn.Linear(512, 3),
            )
        
        self.two_class_layer  = nn.Sequential(
                nn.Dropout(config.dropout),
                # nn.Linear(config.hidden_size_gru*2, 128),
                nn.Linear(768,512),
                nn.ReLU(),
                nn.Linear(512, 2),
            )
        
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=768*2, nhead=12, batch_first=True)
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2,enable_nested_tensor=False)
        encoder_layer = nn.TransformerEncoderLayer(d_model=768*2, nhead=12, batch_first=True)
        self.audio_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2,enable_nested_tensor=False)
        
    def prepend_cls(self, inputs, masks, layer_name):
        if layer_name == 'text':
            embedding_layer = self.text_cls_emb
        elif layer_name == 'audio':
            embedding_layer = self.audio_cls_emb
        elif layer_name == 'text_mixed':
            embedding_layer = self.text_mixed_cls_emb
        elif layer_name == 'audio_mixed':
            embedding_layer = self.audio_mixed_cls_emb
        index = torch.LongTensor([0]).to(device=inputs.device)
        cls_emb = embedding_layer(index)
        cls_emb = cls_emb.expand(inputs.size(0), 1, inputs.size(2))
        outputs = torch.cat((cls_emb, inputs), dim=1)
        
        cls_mask = torch.ones(inputs.size(0), 1).to(device=inputs.device)
        masks = torch.cat((cls_mask, masks), dim=1)
        return outputs, masks
    
    def forward(self, text_inputs, text_mask, audio_inputs, audio_mask,batch_size):
        
        # print(f"shape of text_inputs: {text_inputs.shape}")
        # print(f"shape of audio_inputs: {audio_inputs.shape}")
        
        dialog_len = text_inputs.size(0)//batch_size
        # text feature extraction
        raw_output = self.roberta_model(text_inputs, text_mask)

        T_hidden_states = raw_output.last_hidden_state
        T_features = raw_output['pooler_output']
                    
        # audio feature extraction
        audio_out = self.hubert_model(audio_inputs, audio_mask, output_attentions=True)
        # audio_out = self.data2vec_model(audio_inputs, audio_mask, output_attentions=True)
        A_hidden_states = audio_out.last_hidden_state
        # average over unmasked audio tokens
        A_features = []
        audio_mask_idx_new = []
        for batch in range(A_hidden_states.shape[0]):
                layer = 0
                while layer<12:
                    try:
                        padding_idx = sum(audio_out.attentions[layer][batch][0][0]!=0)
                        audio_mask_idx_new.append(padding_idx)
                        break
                    except:
                        layer += 1
                truncated_feature = torch.mean(A_hidden_states[batch][:padding_idx],0) #Shape is [768]
                A_features.append(truncated_feature)
        ## create new audio mask
        A_features = torch.stack(A_features,0).to(device)
        audio_mask_new = torch.zeros(A_hidden_states.shape[0], A_hidden_states.shape[1]).to(device)
        for batch in range(audio_mask_new.shape[0]):
                audio_mask_new[batch][:audio_mask_idx_new[batch]] = 1
        
        text_inputs, text_attn_mask = self.prepend_cls(T_hidden_states, text_mask, 'text') # add cls token
        audio_inputs, audio_attn_mask = self.prepend_cls(A_hidden_states, audio_mask_new, 'audio') # add cls token

        # pass through CME layers

        for layer_module in self.CME_layers:
            text_inputs, audio_inputs = layer_module(text_inputs, text_attn_mask,
                                                audio_inputs, audio_attn_mask)
            
            
            
        # different fusion strategies
        if self.version == 'v1':
            # fused features
            fused_features = torch.cat((text_inputs, audio_inputs), dim=1) # Shape is [batch_size,seq_a+seq_t ,768*2]
        elif self.version == 'v2':
            # concatenate original features with fused features
            text_concat_features = torch.cat((T_features.unsqueeze(1), text_inputs), dim=1) # Shape is [batch_size, 768*2]
            audio_concat_features = torch.cat((A_features.unsqueeze(1), audio_inputs), dim=1) # Shape is [batch_size, 768*2]
            fused_features = torch.cat((text_concat_features, audio_concat_features), dim=1) # Shape is [batch_size, 768*2] 
            
        elif self.version == 'v3':
            text_concat_features = torch.cat((T_hidden_states, text_inputs[:,1:,:]), dim=2) # Shape is [batch_size, text_length, 768*2]
            audio_concat_features = torch.cat((A_hidden_states, audio_inputs[:,1:,:]), dim=2) # Shape is [batch_size, audio_length, 768*2]
            text_concat_features, text_attn_mask = self.prepend_cls(text_concat_features, text_mask, 'text_mixed') # add cls token
            audio_concat_features, audio_attn_mask = self.prepend_cls(audio_concat_features, audio_mask_new, 'audio_mixed') # add cls token
            text_mixed_features = self.text_encoder(text_concat_features, src_key_padding_mask=(1-text_attn_mask).bool())

            audio_mixed_features = self.audio_encoder(audio_concat_features, src_key_padding_mask=(1-audio_attn_mask).bool())
            # fused features

            
            fused_hidden_states = torch.cat((text_mixed_features, audio_mixed_features), dim=1) # Shape is [batch_size, 768*4]
            fused_features = self.inter_output_layers(fused_hidden_states)
            
        else:
            fused_features = text_inputs
            
        expanded_bottleneck = torch.tile(self.bottleneck, (text_inputs.size(0), 1, 1))
        for layer_module in self.Bottelenck_layer:
            bottle = []
            fusion_output, fusion_bottleneck, lang_bottleneck = layer_module(fused_features, text_attn_mask, text_inputs, text_attn_mask, expanded_bottleneck)
            bottle.append(fusion_bottleneck)
            bottle.append(lang_bottleneck)
            new_bottleneck = torch.mean(torch.stack(bottle, dim=-1), dim=-1)
            expanded_bottleneck = new_bottleneck
            torch.cuda.empty_cache()
            

        fusion_output = self.fc_layer(fusion_output)
        fusion_output = fusion_output.view(batch_size, dialog_len, -1)
        
        # pass through GRU layers
        gru_output = self.GRU_layers(fusion_output).squeeze(0)
        # gru_output = gru_output.unsqueeze(1)
        # del fusion_output, text_attn_mask, audio_inputs, audio_attn_mask, text_inputs, expanded_bottleneck,text_outputs
        # torch.cuda.empty_cache()
        # gru_output = gru_output.unsqueeze(1)
        # output,_ = self.multi_head_attn(gru_output, gru_output, gru_output)
        # output = output.squeeze(1)
        
        # fused_output = self.fused_output_layers(gru_output)
    
        
        
        five_output = self.fused_output_layers(gru_output)
        four_output = self.four_class_layer(gru_output)
        three_output = self.three_class_layer(gru_output)
        two_output = self.two_class_layer(gru_output)
        
        
        gc.collect()
        # print(f"shape of fused_output: {fused_output.shape}")
        # return fused_output
        # return fused_output
        return five_output, four_output, three_output, two_output        



