import torch
from torch import nn
from transformers import RobertaModel, HubertModel, AutoModel
from utils.cross_attn_encoder import CMELayer, BertConfig, GRU_context, GruConfig, Bottleneck
# from positional_encodings.torch_encodings import PositionalEncodingPermute1D, Summer
import torch.nn.functional as F
import gc


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
class rob_hub_cme(nn.Module):            
    def __init__(self, config):        
        super().__init__()

        # load text pre-trained model
        self.roberta_model = AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext")

        # load audio pre-trained model
        self.hubert_model = AutoModel.from_pretrained('TencentGameMate/chinese-hubert-base')
        
        # cls embedding layers
        self.text_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)
        self.audio_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)


        # CME layers
        Bert_config = BertConfig(num_hidden_layers=config.num_hidden_layers)
        self.CME_layers = nn.ModuleList(
            [CMELayer(Bert_config) for _ in range(Bert_config.num_hidden_layers)]
        )

        self.Bottelenck_layer = nn.ModuleList(
            [Bottleneck(Bert_config) for _ in range(Bert_config.bottleneck_layers)]
        )
        if Bert_config.use_bottleneck:
            self.bottleneck = nn.Parameter(torch.randn(
                1, Bert_config.n_bottlenecks, Bert_config.hidden_size) * 0.02)
            self.bottleneck = self.bottleneck.to(dtype=torch.float32)

        
        GRU_config = GruConfig(hidden_size=config.hidden_size_gru, num_layers=config.num_layers_gru)
        self.GRU_layers = GRU_context(GRU_config)
        
        # multi-head attention
        # self.multi_head_attn = BertSelfattLayer(Bert_config)

        
        
        # last linear output layer
        self.fused_output_layers = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Linear(512, 5),
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
        bottle = []
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
        audio_out = self.hubert_model(audio_inputs, audio_mask, output_attentions=True)
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
            del new_bottleneck
            torch.cuda.empty_cache()
        

        fusion_output = torch.mean(fusion_output.view(batch_size, dialog_len, fusion_output.size(1),fusion_output.size(2)),dim=2)
        # pass through GRU layers
        gru_output = self.GRU_layers(fusion_output)
        # gru_output = gru_output.unsqueeze(1)
        del fusion_output, text_attn_mask, audio_inputs, audio_attn_mask, T_hidden_states, A_hidden_states, audio_out, raw_output
        torch.cuda.empty_cache()
        
        fused_output = self.fused_output_layers(gru_output)
        
        gc.collect()
            
        
        return fused_output
        



