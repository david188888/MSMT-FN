import torch
from torch import nn
import math
import sys
import torch.nn.functional as F
import gc

class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self,
                 use_bottleneck=True,
                 n_bottlenecks=3,
                 hidden_size=768,
                 num_hidden_layers=3,
                 bottleneck_layers = 2,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="relu",
                 hidden_dropout_prob=0.3,
                 attention_probs_dropout_prob=0.3,
                 max_position_embeddings=512,
                 add_abs_pos_emb=False,
                 add_pos_enc=False):
        """Constructs BertConfig.
        Args:
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            add_abs_pos_emb: absolute positional embeddings
            add_pos_enc: positional encoding
        """
        self.use_bottleneck = use_bottleneck
        self.n_bottlenecks = n_bottlenecks 
        self.hidden_size = hidden_size
        self.bottleneck_layers = bottleneck_layers
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.add_abs_pos_emb = add_abs_pos_emb
        self.add_pos_enc = add_pos_enc
        
    

class GruConfig(object):
    def __init__(self,
                 input_dim=768,
                 num_layers=2,
                 bidirectional=True,
                 hidden_size=128,
                 output_size=768):
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = 0.3
        

BertLayerNorm = torch.nn.LayerNorm


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.add_abs_pos_emb = config.add_abs_pos_emb
        if self.add_abs_pos_emb:
            self.abs_pos_emb = nn.Parameter(
                torch.randn(512, self.attention_head_size))
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (b, s,h, d) -> (b, h, s, d)

    def forward(self, hidden_states, context, attention_mask=None):
        # print(context.size(),attention_mask.size())
        
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        if self.add_abs_pos_emb:
            pos_emb = self.abs_pos_emb[0:context.size(1), :]
            pos_emb_q = self.abs_pos_emb[0:hidden_states.size(1), :]
            pos_emb_q = pos_emb_q.expand(
                query_layer.size(0), query_layer.size(1), -1, -1)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))  # shape is (b, h, s_q, s_k)

        if self.add_abs_pos_emb:
            attention_pos_scores = torch.matmul(
                query_layer+pos_emb_q, pos_emb.transpose(-1, -2))
            attention_scores = (
                attention_scores+attention_pos_scores) / math.sqrt(self.attention_head_size)
        else:
            attention_scores = attention_scores / \
                math.sqrt(self.attention_head_size)
                

        # Apply the attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand(
                (-1, attention_scores.size(1), attention_scores.size(2), -1))
            attention_mask = attention_mask.masked_fill(
                attention_mask == 0, float('-inf'))
            attention_mask = attention_mask.masked_fill(
                attention_mask == 1, 0.0)
            # print(attention_mask.size())
            # print(attention_scores.size())
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        del mixed_query_layer, mixed_key_layer, mixed_value_layer, query_layer, key_layer, attention_scores, attention_mask
        gc.collect()
        torch.cuda.empty_cache()
        context_layer = torch.matmul(
            attention_probs, value_layer)  # shape is (b, h, s_q, d)
        context_layer = context_layer.permute(
            0, 2, 1, 3).contiguous()  # shape is (b, s_q, h, d)
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertAttOutput(nn.Module):
    def __init__(self, config):
        super(BertAttOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BottleneckFusion(nn.Module):
    def __init__(self, config):
        super(BottleneckFusion, self).__init__()
        self.self_attn = BertSelfattLayer(config)
        self.output = BertAttOutput(config)


    def forward(self, fusion_input ,fusion_attention_mask, lang_input, lang_attention_mask, bottleneck):
        t_mod_lang = fusion_input.size(1)
        t_mod_audio = lang_input.size(1)
        
        in_mod_fusion = torch.cat([fusion_input, bottleneck], dim=1)
        in_mod_lang = torch.cat([lang_input, bottleneck], dim=1)
    
       
        if fusion_attention_mask.size(1) < in_mod_fusion.size(1):
            pad_length = in_mod_fusion.size(1) - fusion_attention_mask.size(1)
            new_attention_mask_fusion = F.pad(fusion_attention_mask, (0, pad_length), "constant", 0)
        else:
            new_attention_mask_fusion = fusion_attention_mask
        
        if lang_attention_mask.size(1) < in_mod_lang.size(1):
            pad_length = in_mod_lang.size(1) - lang_attention_mask.size(1)
            new_attention_mask_lang = F.pad(lang_attention_mask, (0, pad_length), "constant", 0)
        else:
            new_attention_mask_lang = lang_attention_mask


        
        out_mod_fusion = self.self_attn(in_mod_fusion, new_attention_mask_fusion)
        output_fusion = self.output(out_mod_fusion, in_mod_fusion)        

        out_mod_lang = self.self_attn(in_mod_lang, new_attention_mask_lang)
        output_lang = self.output(out_mod_lang, in_mod_lang)


        input_out_fusion = output_fusion[:, :t_mod_lang]
        updated_bottleneck_fusion = output_fusion[:, t_mod_lang:]        
        updated_bottleneck_lang = output_lang[:, t_mod_audio:]
        del in_mod_fusion, in_mod_lang, out_mod_fusion, out_mod_lang, output_lang, output_fusion
        torch.cuda.empty_cache()  

        return input_out_fusion, updated_bottleneck_fusion, updated_bottleneck_lang


class BertCrossattLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor) # attention_output = self.output(output, input_tensor)
        return attention_output


class BertSelfattLayer(nn.Module):
    def __init__(self, config):
        super(BertSelfattLayer, self).__init__()
        self.self = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, attention_mask):
        # Self attention attends to itself, thus keys and querys are the same (input_tensor).

        self_output = self.self(input_tensor, input_tensor, attention_mask=attention_mask)
        attention_output = self.output(self_output, input_tensor)

        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertSelfattLayer(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


"""
---------------------------------------------------------------------------------------
      Above modules are copied from BERT (pytorch-transformer) with modifications.
---------------------------------------------------------------------------------------
"""

    
class GRU_context(nn.Module):
    def __init__(self, config):
        super(GRU_context, self).__init__()
        self.input_dim = config.input_dim
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.output_size = config.output_size
        self.bidirectional = config.bidirectional
        self.dropout = config.dropout
        self.n_directions = 2 if self.bidirectional else 1
        
        self.gru = nn.GRU(input_size = self.input_dim, hidden_size = self.hidden_size, num_layers = self.num_layers, batch_first = True, bidirectional = self.bidirectional)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size*self.n_directions, self.output_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        # self.fc2 = nn.Linear(768, self.output_size)
        
    
    def forward(self, inputs):
        # print(self.input_dim)
        output, hidden = self.gru(inputs)
        # print(f"the shape of hidden is :{hidden.size()}")
        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]
        concat_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        # print(f"the shape of concat_hidden is :{concat_hidden.size()}")
        fc_output_1 = self.fc(concat_hidden)
        return fc_output_1
    
    
class Bottleneck(nn.Module):
    def __init__(self, config):
        super(Bottleneck, self).__init__()
        self.self_attn = BertSelfattLayer(config)
        self.bottleneckfusion = BottleneckFusion(config)
        self.inter = BertIntermediate(config)
        self.output = BertOutput(config)
        
        
    def bottleneck_fusion(self, fusion_input, fusion_attention_mask,lang_feats,lang_attention_mask, bottleneck):
        lang_output, fusion_bottleneck, lang_bottleneck = self.bottleneckfusion(fusion_input, fusion_attention_mask,lang_feats,lang_attention_mask, bottleneck)
        return lang_output ,fusion_bottleneck, lang_bottleneck
    
    def output_fc(self, input):
        # FC layers
        fusion_inter_output = self.inter(input)
        # Layer output
        fusion_output = self.output(fusion_inter_output, input)
        return fusion_output
    
    
    def self_att(self, lang_input, lang_attention_mask):
        # Self Attention
        lang_att_output = self.lang_self_att(lang_input, lang_attention_mask)
        return lang_att_output
    
    
    def forward(self, fusion_input, fusion_attention_mask,
                lang_feats, lang_attention_mask, bottleneck):
        fusion_att_output, fusion_bottleneck, lang_bottleneck = self.bottleneck_fusion(fusion_input, 
        fusion_attention_mask, lang_feats, lang_attention_mask, bottleneck)
        fusion_output = self.output_fc(fusion_att_output)
        
        return fusion_output , fusion_bottleneck, lang_bottleneck
        
        
         

    
    

class CMELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        
        # The cross-attention Layer
        self.audio_attention = BertCrossattLayer(config)
        self.lang_attention = BertCrossattLayer(config)

        # Self-attention Layers
        self.lang_self_att = BertSelfattLayer(config)
        self.audio_self_att = BertSelfattLayer(config)

        # Intermediate and Output Layers (FFNs)
        self.lang_inter = BertIntermediate(config)
        self.lang_output = BertOutput(config)
        self.audio_inter = BertIntermediate(config)
        self.audio_output = BertOutput(config)
        

        
    
    def cross_att(self, lang_input, lang_attention_mask, audio_input, audio_attention_mask):
        # Cross Attention
        lang_att_output = self.lang_attention(
            lang_input, audio_input, ctx_att_mask=audio_attention_mask)
        audio_att_output = self.audio_attention(
            audio_input, lang_input, ctx_att_mask=lang_attention_mask)
        return lang_att_output, audio_att_output

    def self_att(self, lang_input, lang_attention_mask):
        # Self Attention
        lang_att_output = self.lang_self_att(lang_input, lang_attention_mask)
        return lang_att_output

    def output_fc(self, lang_input):
        # FC layers
        lang_inter_output = self.lang_inter(lang_input)

        # Layer output
        lang_output = self.lang_output(lang_inter_output, lang_input)
        return lang_output
    


    def forward(self, lang_feats, lang_attention_mask,
                audio_feats, audio_attention_mask):

        lang_att_output = lang_feats
        audio_att_output = audio_feats
    
        lang_att_output, audio_att_output = self.cross_att(lang_att_output, lang_attention_mask,
                                                           audio_att_output, audio_attention_mask)
        
        lang_att_output = self.self_att(lang_att_output, lang_attention_mask)
        audio_att_output = self.self_att(audio_att_output, audio_attention_mask)
        
        lang_output = self.output_fc(
            lang_att_output)
        
        # print(f"the shape of lang_output is :{lang_output.size()}")

        return lang_output



        
    