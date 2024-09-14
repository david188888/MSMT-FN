import torch
from torch import nn
from transformers import RobertaModel, HubertModel, AutoModel, Data2VecAudioModel
from utils.module import CMELayer, AttnConfig, GRU_context, GruConfig, Bottleneck, FCLayer, TextAttention
import gc
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Msmt_Fn(nn.Module):
    def __init__(self, config):
        super().__init__()

        # load text pre-trained model
        self.roberta_model = AutoModel.from_pretrained(
            "hfl/chinese-roberta-wwm-ext")

        # load audio pre-trained model
        self.hubert_model = AutoModel.from_pretrained(
            'TencentGameMate/chinese-hubert-base')

        # self.roberta_model = RobertaModel.from_pretrained('roberta-base')
        # self.data2vec_model = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base")

        # cls embedding layers
        self.text_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)
        self.audio_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)

        # CME layers
        Attn_config = AttnConfig(num_hidden_layers=config.num_hidden_layers,
                                 n_bottlenecks=config.n_bottlenecks, bottleneck_layers=config.bottleneck_layers)

        self.Text_layer = TextAttention(Attn_config)

        self.CME_layers = nn.ModuleList(
            [CMELayer(Attn_config)
             for _ in range(Attn_config.num_hidden_layers)]
        )

        self.Bottelenck_layer = nn.ModuleList(
            [Bottleneck(Attn_config)
             for _ in range(Attn_config.bottleneck_layers)]
        )
        if Attn_config.use_bottleneck:
            self.bottleneck = nn.Parameter(torch.randn(
                1, Attn_config.n_bottlenecks, Attn_config.hidden_size) * 0.02)
            self.bottleneck = self.bottleneck.to(dtype=torch.float32)

        GRU_config = GruConfig(
            hidden_size=config.hidden_size_gru, num_layers=config.num_layers_gru)
        self.GRU_layers = GRU_context(GRU_config)

        self.fc_layer = FCLayer(config)

        self.inter_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768*2, 768),
            nn.ReLU(),
        )

        # last linear output layer
        self.fused_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 5),
        )

        self.four_class_layer = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
        )

        self.three_class_layer = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )

        self.two_class_layer = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def prepend_cls(self, inputs, masks, layer_name):
        if layer_name == 'text':
            embedding_layer = self.text_cls_emb
        elif layer_name == 'audio':
            embedding_layer = self.audio_cls_emb
        index = torch.LongTensor([0]).to(device=inputs.device)
        cls_emb = embedding_layer(index)
        cls_emb = cls_emb.expand(inputs.size(0), 1, inputs.size(2))
        outputs = torch.cat((cls_emb, inputs), dim=1)

        cls_mask = torch.ones(inputs.size(0), 1).to(device=inputs.device)
        masks = torch.cat((cls_mask, masks), dim=1)
        return outputs, masks

    def forward(self, text_inputs, text_mask, audio_inputs, audio_mask, batch_size):

        dialog_len = text_inputs.size(0)//batch_size
        # text feature extraction
        raw_output = self.roberta_model(text_inputs, text_mask)
        T_hidden_states = raw_output.last_hidden_state

        # audio feature extraction
        audio_out = self.hubert_model(
            audio_inputs, audio_mask, output_attentions=True)
        # audio_out = self.data2vec_model(audio_inputs, audio_mask, output_attentions=True)
        A_hidden_states = audio_out.last_hidden_state
        # average over unmasked audio tokens
        audio_mask_idx_new = []
        for batch in range(A_hidden_states.shape[0]):
            layer = 0
            while layer < 12:
                try:
                    padding_idx = sum(
                        audio_out.attentions[layer][batch][0][0] != 0)
                    audio_mask_idx_new.append(padding_idx)
                    break
                except:
                    layer += 1

        # create new audio mask
        audio_mask_new = torch.zeros(
            A_hidden_states.shape[0], A_hidden_states.shape[1]).to(device)
        for batch in range(audio_mask_new.shape[0]):
            audio_mask_new[batch][:audio_mask_idx_new[batch]] = 1

        text_inputs, text_attn_mask = self.prepend_cls(
            T_hidden_states, text_mask, 'text')  # add cls token
        audio_inputs, audio_attn_mask = self.prepend_cls(
            A_hidden_states, audio_mask_new, 'audio')  # add cls token

        # pass through CME layers

        for layer_module in self.CME_layers:
            text_outputs = layer_module(text_inputs, text_attn_mask,
                                        audio_inputs, audio_attn_mask)

        text_inputs = self.Text_layer(text_inputs, text_attn_mask)

        expanded_bottleneck = torch.tile(
            self.bottleneck, (text_inputs.size(0), 1, 1))
        for layer_module in self.Bottelenck_layer:
            bottle = []
            fusion_output, fusion_bottleneck, lang_bottleneck = layer_module(
                text_inputs, text_attn_mask, text_outputs, text_attn_mask, expanded_bottleneck)
            bottle.append(fusion_bottleneck)
            bottle.append(lang_bottleneck)
            new_bottleneck = torch.mean(torch.stack(bottle, dim=-1), dim=-1)
            expanded_bottleneck = new_bottleneck
            torch.cuda.empty_cache()

        fusion_output = self.fc_layer(fusion_output)
        fusion_output = fusion_output.view(batch_size, dialog_len, -1)

        # pass through GRU layers
        gru_output = self.GRU_layers(fusion_output).squeeze(0)

        five_output = self.fused_output_layers(gru_output)
        four_output = self.four_class_layer(gru_output)
        three_output = self.three_class_layer(gru_output)
        two_output = self.two_class_layer(gru_output)

        gc.collect()
        return five_output, four_output, three_output, two_output
        # return five_output
