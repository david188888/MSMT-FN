import torch
from torch import nn
from transformers import RobertaModel, HubertModel, AutoModel
from utils.cross_attn_encoder import CMELayer, BertConfig, GRU_context, GruConfig
# from positional_encodings.torch_encodings import PositionalEncodingPermute1D, Summer
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
class rob_hub_cme(nn.Module):            
    def __init__(self, config):        
        super().__init__()

        # load text pre-trained model
        self.roberta_model = AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext")

        # load audio pre-trained model
        self.hubert_model = AutoModel.from_pretrained('TencentGameMate/chinese-hubert-base')
        
        # output layers for each single modality
        self.T_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768, 1)
           )           
        self.A_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768, 1)
          )
        
        # cls embedding layers
        self.text_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)
        self.audio_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)
        self.text_mixed_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768*2)
        self.audio_mixed_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768*2)

        # position encoding
        #self.pos_enc = Summer(PositionalEncoding1D(768))
        

        # CME layers
        Bert_config = BertConfig(num_hidden_layers=config.num_hidden_layers)
        self.CME_layers = nn.ModuleList(
            [CMELayer(Bert_config) for _ in range(Bert_config.num_hidden_layers)]
        )
        if Bert_config.use_bottleneck:
            self.bottleneck = nn.Parameter(torch.randn(
                1, Bert_config.n_bottlenecks, Bert_config.hidden_size) * 0.02)

        
        GRU_config = GruConfig(hidden_size=config.hidden_size, num_layers=config.num_layers)
        self.GRU_layers = nn.ModuleList(
            [GRU_context(GRU_config) for _ in range(GRU_config.num_layers)]
        )

        # fused method V2
        self.text_mixed_layer = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768*2, 768),
            nn.ReLU()
        )
        self.audio_mixed_layer = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768*2, 768),
            nn.ReLU()
        )
        
        # fusion method V3
        encoder_layer = nn.TransformerEncoderLayer(d_model=768*2, nhead=12, batch_first=True)
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2,enable_nested_tensor=False)
        encoder_layer = nn.TransformerEncoderLayer(d_model=768*2, nhead=12, batch_first=True)
        self.audio_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2,enable_nested_tensor=False)
        
        # last linear output layer
        self.fused_output_layers = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(768*4, 768),
                nn.ReLU(),
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Linear(512, 5)
            )
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
        cls_emb = cls_emb.expand(inputs.size(0), inputs.size(1), 1, inputs.size(3))
        outputs = torch.cat((cls_emb, inputs), dim=2)
        
        cls_mask = torch.ones(inputs.size(0), inputs.size(1),1).to(device=inputs.device)
        masks = torch.cat((cls_mask, masks), dim=2)
        return outputs, masks
    
    def forward(self, text_inputs, text_mask, audio_inputs, audio_mask):
        self.bottle = []
        # text feature extraction
        t_output = torch.zeros(text_inputs.shape[0], text_inputs.shape[1], 768).to(device)
        t_hidden = torch.zeros(text_inputs.shape[0], text_inputs.shape[1], text_inputs.shape[2], 768).to(device)
        for i in range(len(text_inputs[1])):
            segment = text_inputs[:,i,:].to(device)
            mask = text_mask[:,i,:].to(device)

            raw_output = self.roberta_model(segment, mask)

            T_hidden_states = raw_output.last_hidden_state
            T_features = raw_output["pooler_output"]  # Shape is [batch_size, 768]
            t_output[:,i,:] = T_features
            t_hidden[:,i,:,:] = T_hidden_states 
            
        # print(f"the shape of t_output: {t_output.shape}")
        # print(f"the shape of t_hidden: {t_hidden.shape}")
                    
        # audio feature extraction
        a_hidden = torch.zeros(audio_inputs.shape[0], audio_inputs.shape[1], 299, 768).to(device)
        a_attention = []
        for i in range(len(audio_inputs[1])):
            audio_segment = audio_inputs[:,i,:].to(device)
            audio_mask_segment = audio_mask[:,i,:].to(device)
            
            audio_out = self.hubert_model(audio_segment, audio_mask_segment, output_attentions=True)
            
            A_hidden_states = audio_out.last_hidden_state
            A_attention = audio_out.attentions
            
            
            a_hidden[:,i,:,:] = A_hidden_states
            a_attention.append(A_attention)
            
        # print(f"shape of a_features: {a_hidden.shape}")
        # average over unmasked audio tokens
        # A_features = []
        audio_mask_idx_new = []
        for batch in range(A_hidden_states.shape[0]):
            dialog_audio_mask_idx_new = []
            for dialog in range(A_hidden_states.shape[1]):
                layer = 0
                while layer<12:
                    try:
                        padding_idx = sum(a_attention[dialog][layer][batch][0][0]!=0)
                        dialog_audio_mask_idx_new.append(padding_idx)
                        break
                    except:
                        layer += 1
                # print(f"the dialog_audio_mask_idx_new: {dialog_audio_mask_idx_new}")
                        
            audio_mask_idx_new.append(dialog_audio_mask_idx_new)
            truncated_feature = torch.mean(A_hidden_states[batch][:padding_idx],0) #Shape is [768]
        #     A_features.append(truncated_feature)
        # A_features = torch.stack(A_features,0).to(device)

        
        ## create new audio mask
        audio_mask_new = torch.zeros(a_hidden.shape[0], a_hidden.shape[1], a_hidden.shape[2]).to(device)
        for batch in range(audio_mask_new.shape[0]):
            for dialog in range(audio_mask_new.shape[1]):
                audio_mask_new[batch][dialog][:audio_mask_idx_new[batch][dialog]] = 1
                
        # text output layer
        T_output = self.T_output_layers(T_features)                    # Shape is [batch_size, 2]
        
        # audio output layer
        # A_output = self.A_output_layers(A_features)                    # Shape is [batch_size, 2]
        
        # CME layers
        ## prepend cls tokens
        # print(f"shape of A_hidden_states: {A_hidden_states.shape}")
        text_inputs, text_attn_mask = self.prepend_cls(t_hidden, text_mask, 'text') # add cls token
        audio_inputs, audio_attn_mask = self.prepend_cls(a_hidden, audio_mask_new, 'audio') # add cls token
        batch_size = text_inputs.size(0)
        dialog_len = text_inputs.size(1)
        print(f"shape of audio_inputs: {audio_inputs.shape}")
        print(f"shape of audio_attn_mask: {audio_attn_mask.shape}")
        print(f"shape of text_inputs: {text_inputs.shape}")
        print(f"shape of text_attn_mask: {text_attn_mask.shape}")
        
        
        # change the shape of text_inputs and audio_inputs and text_attn_mask and audio_attn_mask
        text_inputs = text_inputs.view(text_inputs.shape[0]*text_inputs.shape[1], text_inputs.shape[2], text_inputs.shape[3])
        audio_inputs = audio_inputs.view(audio_inputs.shape[0]*audio_inputs.shape[1], audio_inputs.shape[2], audio_inputs.shape[3])
        text_attn_mask = text_attn_mask.view(text_attn_mask.shape[0]*text_attn_mask.shape[1], text_attn_mask.shape[2])
        audio_attn_mask = audio_attn_mask.view(audio_attn_mask.shape[0]*audio_attn_mask.shape[1], audio_attn_mask.shape[2])

        # position encoding
        # pos_enc_text = Summer(PositionalEncodingPermute1D(text_inputs.shape[1]))
        # text_inputs = pos_enc_text(text_inputs)
        # pos_enc_audio = Summer(PositionalEncodingPermute1D(audio_inputs.shape[1]))
        # audio_inputs = pos_enc_audio(audio_inputs)
        


        # pass through CME layers
        expanded_bottleneck = self.bottleneck.expand(text_inputs.size(0), -1, -1)
        for layer_module in self.CME_layers:
            text_inputs, audio_inputs, lang_bottleneck, audio_bottleneck = layer_module(text_inputs, text_attn_mask,
                                                audio_inputs, audio_attn_mask, expanded_bottleneck)
            self.bottle.append(lang_bottleneck)
            self.bottle.append(audio_bottleneck)
            stacked_bottle = torch.stack(self.bottle, dim=-1)
            new_bottleneck = torch.mean(stacked_bottle, dim=-1)
            self.bottleneck = nn.Parameter(new_bottleneck)
            
            
        text_inputs = text_inputs.view(batch_size, dialog_len, text_inputs.size(1), text_inputs.size(2))
        text_attn_mask = text_attn_mask.view(batch_size, dialog_len, text_attn_mask.size(1))
        audio_inputs = audio_inputs.view(batch_size, dialog_len, audio_inputs.size(1), audio_inputs.size(2))
        audio_attn_mask = audio_attn_mask.view(batch_size, dialog_len, audio_attn_mask.size(1))
        
        text_inputs = text_inputs.mean(dim=2)
        audio_inputs = audio_inputs.mean(dim=2)
        
            
            
        print(f"shape of text_inputs: {text_inputs.shape}")
        print(f"shape of audio_inputs: {audio_inputs.shape}")
        print(f"shape of text_attn_mask: {text_attn_mask.shape}")
        print(f"shape of audio_attn_mask: {audio_attn_mask.shape}")
        
        
        
        
        # pass through GRU layers
        for i, layer_module in enumerate(self.GRU_layers):
            # 判断是不是首次进入GRU，如果是初始化hidden
            print(f'layer_module: {i}')
            if i == 0:
                hidden = layer_module.init_hidden(text_inputs.size(0))
                
            print(f"shape of text_inputs: {text_inputs.shape}")
            print(f"shape of hidden: {hidden.shape}")
            gru_output = layer_module(text_inputs, hidden)
        
        
        print(f"shape of output: {gru_output.shape}")
            # concatenate original features with fused features
            
            
        # text_concat_features = torch.cat((T_hidden_states, text_inputs[:,1:,:]), dim=2) # Shape is [batch_size, text_length, 768*2]
        # audio_concat_features = torch.cat((A_hidden_states, audio_inputs[:,1:,:]), dim=2) # Shape is [batch_size, audio_length, 768*2]
        # text_concat_features, text_attn_mask = self.prepend_cls(text_concat_features, text_mask, 'text_mixed') # add cls token
        # audio_concat_features, audio_attn_mask = self.prepend_cls(audio_concat_features, audio_mask_new, 'audio_mixed') # add cls token
        # text_mixed_features = self.text_encoder(text_concat_features, src_key_padding_mask=(1-text_attn_mask).bool())
        # audio_mixed_features = self.audio_encoder(audio_concat_features, src_key_padding_mask=(1-audio_attn_mask).bool())
        # # fused features
        # fused_hidden_states = torch.cat((text_mixed_features[:,0,:], audio_mixed_features[:,0,:]), dim=1) # Shape is [batch_size, 768*4]
        
        

        # last linear output layer
        # fused_output = self.fused_output_layers(fused_hidden_states) # Shape is [batch_size, 5]
        
        # attention_output = self.GRU_layers(input_dim, seq_lengths, hidden)
        
        
        
        # return fused_output
        
        # return {
        #         'T': T_output, 
        #         'A': A_output, 
        #         'M': fused_output
        # }
    


