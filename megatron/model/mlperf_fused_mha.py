import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from apex.contrib.multihead_attn import fast_mask_softmax_dropout_func

from .mlperf_bmm1 import *
from .mlperf_bmm2 import *
from .mlperf_padding import *
from .mlperf_softmax import *

from megatron.module import MegatronModule
from megatron.global_vars import get_args


class FastUnpadSelfAttention(MegatronModule):
    def __init__(self, hidden_size, num_attention_heads, enable_stream=True, enable_sync=True, fuse_mask=True,
                 fuse_scale=True, fuse_qkv=True,
                 fuse_dropout=True, apex_softmax=True, pad=True, from_qkv=False):
        super(FastUnpadSelfAttention, self).__init__()
        args = get_args()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.hidden_size = hidden_size

        self.fuse_qkv = fuse_qkv
        self.fuse_scale = fuse_scale
        self.fuse_mask = fuse_mask
        self.fuse_dropout = fuse_dropout
        self.apex_softmax = apex_softmax
        self.pad = pad
        self.enable_stream = enable_stream
        self.from_qkv = from_qkv

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        if self.fuse_qkv:
            self.bmm1 = Bmm1Strided(None, None, self.num_attention_heads, self.attention_head_size,
                                    scale=self.fuse_scale, stream=enable_stream, sync=enable_sync, timer=False)
            self.bmm2 = Bmm2Strided(None, None, self.num_attention_heads, self.attention_head_size,
                                    stream=enable_stream, sync=enable_sync, timer=False)
        else:
            self.bmm1 = Bmm1(None, None, self.num_attention_heads, self.attention_head_size, scale=self.fuse_scale,
                             stream=enable_stream, sync=enable_sync)
            self.bmm2 = Bmm2(None, None, self.num_attention_heads, self.attention_head_size, stream=enable_stream,
                             sync=enable_sync)

        if not self.fuse_dropout:
            self.dropout = nn.Dropout(args.attention_dropout)

        if self.fuse_mask and self.fuse_dropout:
            self.softmax = FastMaskSoftmaxDropout(dim=-1, dropout_prob=args.attention_dropout,
                                                  stream=enable_stream, sync=(not self.pad), timer=False)
        elif self.fuse_mask:
            self.softmax = FastMaskSoftmax(dim=-1, stream=enable_stream, sync=enable_sync, timer=False)
        else:
            self.softmax = FastSoftmax(dim=-1, stream=enable_stream, sync=enable_sync, timer=False)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = torch.reshape(x, new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_key_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = torch.reshape(x, new_x_shape)
        return x.permute(0, 2, 3, 1)

    def pytorch_softmax(self, attention_scores, batch, seqlen, heads):
        ntokens2 = 0
        for i in range(batch):
            ntokens2 += seqlen[i] * seqlen[i] * self.num_attention_heads
        attention_probs = torch.zeros(ntokens2, device=torch.cuda.current_device(), dtype=torch.float16)
        ntokens2 = 0
        for i in range(batch):
            tokens2 = seqlen[i] * seqlen[i] * self.num_attention_heads
            attention_probs[ntokens2:ntokens2 + tokens2] = F.softmax(
                attention_scores[ntokens2:ntokens2 + tokens2].view(1, self.num_attention_heads, seqlen[i], seqlen[i]),
                dim=-1).flatten().contiguous()
            ntokens2 += tokens2
        return attention_probs

    def forward(self, hidden_states, attention_mask, seqlen, batch, is_training=True):

        self.batch = batch
        if not self.from_qkv:
            # QKV
            if self.fuse_qkv:
                weight = torch.cat(
                    [self.query.weight.view(self.num_attention_heads, self.attention_head_size, 1, self.hidden_size),
                     self.key.weight.view(self.num_attention_heads, self.attention_head_size, 1, self.hidden_size),
                     self.value.weight.view(self.num_attention_heads, self.attention_head_size, 1, self.hidden_size)],
                    dim=1).reshape(self.all_head_size * 3, self.hidden_size).contiguous()
                bias = torch.cat([self.query.bias.view(self.num_attention_heads, 1, self.attention_head_size),
                                  self.key.bias.view(self.num_attention_heads, 1, self.attention_head_size),
                                  self.value.bias.view(self.num_attention_heads, 1, self.attention_head_size)],
                                 dim=1).reshape(3 * self.hidden_size).contiguous()
                mixed_x_layer = torch.addmm(bias, hidden_states, weight.t())
            else:
                query_layer = self.query(hidden_states)
                key_layer = self.key(hidden_states)
                value_layer = self.value(hidden_states)
        else:
            query_layer, key_layer, value_layer = hidden_states

        # BMM1.
        if self.enable_stream: torch.cuda.synchronize()
        if self.fuse_qkv and not self.from_qkv:
            attention_scores, qkv_layer = self.bmm1(mixed_x_layer, self.batch, seqlen)
        else:
            attention_scores = self.bmm1(query_layer, key_layer, self.batch, seqlen)

        if self.enable_stream: torch.cuda.synchronize()
        if not self.fuse_scale:
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Softmax.
        if self.enable_stream: torch.cuda.synchronize()
        if self.fuse_mask and self.fuse_dropout:
            attention_probs = self.softmax(attention_scores, attention_mask, self.batch, seqlen,
                                           self.num_attention_heads, is_training)
        elif self.fuse_mask:
            attention_probs = self.softmax(attention_scores, attention_mask, self.batch, seqlen,
                                           self.num_attention_heads)
        else:
            attention_scores = attention_scores + attention_mask.view(-1)
            if self.apex_softmax:
                attention_probs = self.softmax(attention_scores, self.batch, seqlen, self.num_attention_heads)
            else:
                if self.pad == True:
                    attention_probs = F.softmax(
                        attention_scores.view(batch, self.num_attention_heads, seqlen[0], seqlen[0]),
                        dim=-1).flatten().contiguous()
                else:
                    attention_probs = self.pytorch_softmax(attention_scores, self.batch, seqlen,
                                                           self.num_attention_heads)

        # Dropout.
        if self.enable_stream: torch.cuda.synchronize()
        if not self.fuse_dropout:
            attention_probs = self.dropout(attention_probs)

        # BMM2.
        if self.enable_stream: torch.cuda.synchronize()
        if self.fuse_qkv and not self.from_qkv:
            context_layer = self.bmm2(attention_probs, qkv_layer, self.batch, seqlen)
        else:
            context_layer = self.bmm2(attention_probs, value_layer, self.batch, seqlen)

        if self.enable_stream: torch.cuda.synchronize()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = torch.reshape(context_layer, new_context_layer_shape)
        return context_layer


### used like this:

"""

# This module uses Apex C++ multihead attention implementation with fusions. 
class FastBertAttention(nn.Module):
    def __init__(self, config):
        super(FastBertAttention, self).__init__()
        self.multi_head_attention = SelfMultiheadAttn(config.hidden_size, config.num_attention_heads, dropout = config.attention_probs_dropout_prob, bias=True, include_norm_add=False, impl='fast', separate_qkv_params=True, mask_additive=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
    def forward(self, input_tensor, attention_mask):
        residual=input_tensor
        multi_head_attention_output,_ = self.multi_head_attention(query = input_tensor, key = input_tensor, value = input_tensor, key_padding_mask=attention_mask, need_weights=True,attn_mask = None, is_training = self.training)
        attention_output = self.dropout(multi_head_attention_output)
        attention_output = self.layer_norm(attention_output + residual)
        return attention_output

class FastUnpadBertAttention(nn.Module):
    def __init__(self, config):
        super(FastUnpadBertAttention, self).__init__()
        self.self = FastUnpadBertSelfAttention(config, enable_stream=config.enable_stream, enable_sync=False, fuse_mask=config.fuse_mask, fuse_scale=config.fuse_scale, fuse_qkv=config.fuse_qkv, fuse_dropout=config.fuse_dropout, apex_softmax=config.apex_softmax, pad=config.pad)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, seqlen, batch):
        self_output = self.self(input_tensor, attention_mask, seqlen, batch, is_training = self.training)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output
        
class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.unpad = config.unpad
        if config.fused_mha:
            self.attention = FastBertAttention(config)
        elif config.unpad:
            self.attention = FastUnpadBertAttention(config)
        else:
            self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, seqlen, batch):
        if self.unpad:
            attention_output = self.attention(hidden_states, attention_mask, seqlen, batch)
        else:
            attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
"""
