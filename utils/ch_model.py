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

        # # load text pre-trained model
        # self.roberta_model = AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext")

        # # load audio pre-trained model
        # self.hubert_model = AutoModel.from_pretrained('TencentGameMate/chinese-hubert-base')
        
        self.roberta_model = RobertaModel.from_pretrained('roberta-base')
        self.data2vec_model = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base")
        
        # cls embedding layers
        self.text_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)
        self.audio_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)


        # CME layers
        Attn_config = AttnConfig(num_hidden_layers=config.num_hidden_layers,n_bottlenecks=config.n_bottlenecks,bottleneck_layers=config.bottleneck_layers)
        self.CME_layers = nn.ModuleList(
            [CMELayer(Attn_config) for _ in range(Attn_config.num_hidden_layers)]
        )

        self.Bottelenck_layer = nn.ModuleList(
            [Bottleneck(Attn_config) for _ in range(Attn_config.bottleneck_layers)]
        )
        if Attn_config.use_bottleneck:
            self.bottleneck = nn.Parameter(torch.randn(
                1, Attn_config.n_bottlenecks, Attn_config.hidden_size) * 0.02)
            self.bottleneck = self.bottleneck.to(dtype=torch.float32)

        
        GRU_config = GruConfig(hidden_size=config.hidden_size_gru, num_layers=config.num_layers_gru)
        self.GRU_layers = GRU_context(GRU_config)
        
        self.fc_layer = FCLayer(config)
        # multi-head attention
        # self.multi_head_attn = nn.MultiheadAttention(embed_dim=768, num_heads=12, dropout=config.dropout)

        
        
        # last linear output layer
        self.fused_output_layers = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size_gru*2, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
    def prepend_cls(self, inputs, masks, layer_name):
        if layer_name == 'text':
            embedding_layer = self.text_cls_emb
        elif layer_name == 'audio':
            embedding_layer = self.audio_cls_emb
        index = torch.LongTensor([0]).to(device=inputs.device)
        cls_emb = embedding_layer(index)
        cls_emb = cls_emb.expand(inputs.size(0), 1 ,inputs.size(2))
        outputs = torch.cat((cls_emb, inputs), dim=1)
        
        cls_mask = torch.ones(inputs.size(0), 1).to(device=inputs.device)
        masks = torch.cat((cls_mask, masks), dim=1)
        return outputs, masks
    
    def forward(self, text_inputs, text_mask, audio_inputs, audio_mask,batch_size):
        dialog_len = text_inputs.size(0)//batch_size
        # text feature extraction
        # t_output = torch.zeros(text_inputs.shape[0], text_inputs.shape[1], 768).to(device)
        # t_hidden = torch.zeros(text_inputs.shape[0], text_inputs.shape[1], text_inputs.shape[2], 768).to(device)
        raw_output = self.roberta_model(text_inputs, text_mask)

        T_hidden_states = raw_output.last_hidden_state
        # T_features = raw_output["pooler_output"]  # Shape is [batch_size, 768]
            # t_output[:,i,:] = T_features
            # t_hidden[:,i,:,:] = T_hidden_states
            # del T_hidden_states, T_features, raw_output
            # torch.cuda.empty_cache()

                    
        # audio feature extraction
        # audio_out = self.hubert_model(audio_inputs, audio_mask, output_attentions=True)
        audio_out = self.data2vec_model(audio_inputs, audio_mask, output_attentions=True)
        A_hidden_states = audio_out.last_hidden_state
        # average over unmasked audio tokens
        # A_features = []
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
            
            # truncated_feature = torch.mean(A_hidden_states[batch][:padding_idx],0) #Shape is [768]
        #     A_features.append(truncated_feature)
        # A_features = torch.stack(A_features,0).to(device)

        
        ## create new audio mask
        audio_mask_new = torch.zeros(A_hidden_states.shape[0], A_hidden_states.shape[1]).to(device)
        for batch in range(audio_mask_new.shape[0]):
                audio_mask_new[batch][:audio_mask_idx_new[batch]] = 1
        
        text_inputs, text_attn_mask = self.prepend_cls(T_hidden_states, text_mask, 'text') # add cls token
        audio_inputs, audio_attn_mask = self.prepend_cls(A_hidden_states, audio_mask_new, 'audio') # add cls token
        
        # del raw_output, T_hidden_states, A_hidden_states, audio_out, audio_mask_new
        # torch.cuda.empty_cache()

        # pass through CME layers

        # print(f"shape of text_inputs: {text_inputs.shape}")
        for layer_module in self.CME_layers:
            text_outputs = layer_module(text_inputs, text_attn_mask,
                                                audio_inputs, audio_attn_mask)
        expanded_bottleneck = torch.tile(self.bottleneck, (text_inputs.size(0), 1, 1))
        for layer_module in self.Bottelenck_layer:
            bottle = []
            fusion_output, fusion_bottleneck, lang_bottleneck = layer_module(text_outputs, text_attn_mask, text_inputs, text_attn_mask, expanded_bottleneck)
            bottle.append(fusion_bottleneck)
            bottle.append(lang_bottleneck)
            new_bottleneck = torch.mean(torch.stack(bottle, dim=-1), dim=-1)
            expanded_bottleneck = new_bottleneck
            # del new_bottleneck
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
        fused_output = self.fused_output_layers(gru_output)
        gc.collect()
        # print(f"shape of fused_output: {fused_output.shape}")
            
        return fused_output
        



