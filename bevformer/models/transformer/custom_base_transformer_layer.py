"""Custom base transformer layer for BEVFormer, similar to mmcv's BaseTransformerLayer."""
import copy
import warnings
from typing import Optional

import torch
import torch.nn as nn

try:
    from mmcv.cnn import Linear, build_activation_layer, build_norm_layer
    from mmcv.cnn.bricks.transformer import build_feedforward_network, build_attention
    from mmcv.runner.base_module import BaseModule, ModuleList
    from mmcv import ConfigDict
except Exception:
    # Fallback
    BaseModule = nn.Module
    ModuleList = nn.ModuleList
    ConfigDict = dict
    def build_attention(cfg): return None
    def build_feedforward_network(cfg): return None
    def build_norm_layer(cfg, dims): return None, nn.LayerNorm(dims)
    def build_activation_layer(cfg): return nn.ReLU()
    Linear = nn.Linear


try:
    from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER
except Exception:
        TRANSFORMER_LAYER = None

if TRANSFORMER_LAYER is not None:
    @TRANSFORMER_LAYER.register_module()
    class MyCustomBaseTransformerLayer(BaseModule):
        """Base transformer layer that can be built from config.
        
        Supports flexible attention and FFN configuration through attn_cfgs and ffn_cfgs.
        """
        def __init__(self,
                     attn_cfgs=None,
                     ffn_cfgs=dict(
                         type='FFN',
                         embed_dims=256,
                         feedforward_channels=1024,
                         num_fcs=2,
                         ffn_drop=0.,
                         act_cfg=dict(type='ReLU', inplace=True),
                     ),
                     operation_order=None,
                     norm_cfg=dict(type='LN'),
                     init_cfg=None,
                     batch_first=True,
                     **kwargs):
            super().__init__(init_cfg)
            
            self.batch_first = batch_first
            
            assert set(operation_order) & set(['self_attn', 'norm', 'ffn', 'cross_attn']) == \
                set(operation_order), f'The operation_order of {self.__class__.__name__} should ' \
                f'contains all four operation type {["self_attn", "norm", "ffn", "cross_attn"]}'
            
            num_attn = operation_order.count('self_attn') + operation_order.count('cross_attn')
            if isinstance(attn_cfgs, dict):
                attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
            else:
                assert num_attn == len(attn_cfgs), f'The length of attn_cfg {num_attn} is ' \
                    f'not consistent with the number of attention in operation_order {operation_order}.'
            
            self.num_attn = num_attn
            self.operation_order = operation_order
            self.norm_cfg = norm_cfg
            self.pre_norm = operation_order[0] == 'norm'
            self.attentions = ModuleList()
            
            index = 0
            for operation_name in operation_order:
                if operation_name in ['self_attn', 'cross_attn']:
                    if 'batch_first' in attn_cfgs[index]:
                        assert self.batch_first == attn_cfgs[index]['batch_first']
                    else:
                        attn_cfgs[index]['batch_first'] = self.batch_first
                    attention = build_attention(attn_cfgs[index])
                    attention.operation_name = operation_name
                    self.attentions.append(attention)
                    index += 1
            
            self.embed_dims = self.attentions[0].embed_dims
            
            self.ffns = ModuleList()
            num_ffns = operation_order.count('ffn')
            if isinstance(ffn_cfgs, dict):
                ffn_cfgs = ConfigDict(ffn_cfgs)
            if isinstance(ffn_cfgs, dict):
                ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
            assert len(ffn_cfgs) == num_ffns
            for ffn_index in range(num_ffns):
                if 'embed_dims' not in ffn_cfgs[ffn_index]:
                    ffn_cfgs[ffn_index]['embed_dims'] = self.embed_dims
                else:
                    assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims
                self.ffns.append(build_feedforward_network(ffn_cfgs[ffn_index]))
            
            self.norms = ModuleList()
            num_norms = operation_order.count('norm')
            for _ in range(num_norms):
                self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])
        
        def forward(self,
                    query,
                    key=None,
                    value=None,
                    query_pos=None,
                    key_pos=None,
                    attn_masks=None,
                    query_key_padding_mask=None,
                    key_padding_mask=None,
                    **kwargs):
            """Forward function.
            
            Args:
                query: [B, N, C] or [N, B, C] depending on batch_first
                key, value: same as query
                query_pos, key_pos: positional encodings
                **kwargs: additional arguments for attention modules
            """
            norm_index = 0
            attn_index = 0
            ffn_index = 0
            identity = query
            
            if attn_masks is None:
                attn_masks = [None for _ in range(self.num_attn)]
            elif isinstance(attn_masks, torch.Tensor):
                attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            else:
                assert len(attn_masks) == self.num_attn
            
            for layer in self.operation_order:
                if layer == 'self_attn':
                    temp_key = temp_value = query
                    query = self.attentions[attn_index](
                        query, temp_key, temp_value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=query_pos,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=query_key_padding_mask,
                        **kwargs)
                    attn_index += 1
                    identity = query
                
                elif layer == 'norm':
                    query = self.norms[norm_index](query)
                    norm_index += 1
                
                elif layer == 'cross_attn':
                    query = self.attentions[attn_index](
                        query, key, value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=key_pos,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=key_padding_mask,
                        **kwargs)
                    attn_index += 1
                    identity = query
                
                elif layer == 'ffn':
                    query = self.ffns[ffn_index](
                        query, identity if self.pre_norm else None)
                    ffn_index += 1
            
            return query

