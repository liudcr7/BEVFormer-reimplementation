import copy
import warnings

import torch

from mmcv.cnn import build_norm_layer
from mmcv.runner.base_module import BaseModule, ModuleList

from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER

# Avoid BC-breaking of importing MultiScaleDeformableAttention from this file
try:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention  # noqa F401
    warnings.warn(
        ImportWarning(
            '``MultiScaleDeformableAttention`` has been moved to '
            '``mmcv.ops.multi_scale_deform_attn``, please change original path '  # noqa E501
            '``from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention`` '  # noqa E501
            'to ``from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention`` '  # noqa E501
        ))
except ImportError:
    warnings.warn('Fail to import ``MultiScaleDeformableAttention`` from '
                  '``mmcv.ops.multi_scale_deform_attn``, '
                  'You should install ``mmcv-full`` if you need this module. ')
from mmcv.cnn.bricks.transformer import build_feedforward_network, build_attention


if TRANSFORMER_LAYER is not None:
    @TRANSFORMER_LAYER.register_module()
    class BEVFormerLayer(BaseModule):
        """BEVFormer layer that integrates TemporalSelfAttention and SpatialCrossAttention.
        
        This layer follows the original BEVFormer design:
        - self_attn: TemporalSelfAttention (fuses prev_bev and current_bev)
        - cross_attn: SpatialCrossAttention (aggregates multi-view image features)
        - ffn: Feed-forward network
        """
        
        def __init__(self,
                     attn_cfgs,
                     ffn_cfgs,
                     operation_order=None,
                     norm_cfg=dict(type='LN'),
                     batch_first=True,
                     **kwargs):
            super(BEVFormerLayer, self).__init__()
            
            self.fp16_enabled = False
            self.batch_first = batch_first
            
            assert len(operation_order) == 6
            assert set(operation_order) == set(['self_attn', 'norm', 'cross_attn', 'ffn'])
            
            # Build attention modules
            num_attn = operation_order.count('self_attn') + operation_order.count('cross_attn')
            if isinstance(attn_cfgs, dict):
                attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
            else:
                assert num_attn == len(attn_cfgs), \
                    f'The length of attn_cfg {num_attn} is not consistent with ' \
                    f'the number of attention in operation_order {operation_order}.'
            
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
            
            # Build FFN modules
            num_ffns = operation_order.count('ffn')
            if isinstance(ffn_cfgs, dict):
                if 'embed_dims' not in ffn_cfgs:
                    ffn_cfgs['embed_dims'] = self.embed_dims
                ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
            
            assert len(ffn_cfgs) == num_ffns, \
                f'Number of ffn_cfgs ({len(ffn_cfgs)}) must match number of FFNs ({num_ffns})'
            
            self.ffns = ModuleList()
            for ffn_cfg in ffn_cfgs:
                if 'embed_dims' not in ffn_cfg:
                    ffn_cfg['embed_dims'] = self.embed_dims
                self.ffns.append(build_feedforward_network(ffn_cfg))
            
            # Build norm modules
            num_norms = operation_order.count('norm')
            self.norms = ModuleList()
            for _ in range(num_norms):
                self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])
        
        def forward(self,
                    query,
                    key=None,
                    value=None,
                    bev_pos=None,
                    query_pos=None,
                    key_pos=None,
                    attn_masks=None,
                    query_key_padding_mask=None,
                    key_padding_mask=None,
                    ref_2d=None,
                    ref_3d=None,
                    bev_h=None,
                    bev_w=None,
                    reference_points_cam=None,
                    mask=None,
                    spatial_shapes=None,
                    level_start_index=None,
                    prev_bev=None,
                    **kwargs):
            """Forward function for BEVFormerLayer.
            
            Args:
                query: [bs, num_query, embed_dims] BEV queries
                key, value: [num_cam, num_value, bs, embed_dims] multi-camera features
                bev_pos: [bs, num_query, embed_dims] positional encoding
                ref_2d: [bs*2, num_query, num_levels, 2] 2D reference points for TSA
                ref_3d: [bs, num_query, D, 3] 3D reference points for SCA
                reference_points_cam: [num_cam, bs, num_query, D, 2] projected points
                bev_mask: [num_cam, bs, num_query, D] visibility mask
                prev_bev: [bs*2, num_query, embed_dims] previous BEV (stacked with current)
                
            Returns:
                query: [bs, num_query, embed_dims] updated BEV queries
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
                # Temporal self-attention
                if layer == 'self_attn':
                    query = self.attentions[attn_index](
                        query,
                        prev_bev,  # key
                        prev_bev,  # value
                        identity if self.pre_norm else None,  # residual
                        query_pos=bev_pos,
                        key_pos=bev_pos,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=query_key_padding_mask,
                        reference_points=ref_2d,
                        spatial_shapes=torch.tensor(
                            [[bev_h, bev_w]], device=query.device, dtype=torch.long),
                        level_start_index=torch.tensor([0], device=query.device, dtype=torch.long),
                        **kwargs)
                    attn_index += 1
                    identity = query
                
                elif layer == 'norm':
                    query = self.norms[norm_index](query)
                    norm_index += 1
                
                # Spatial cross-attention
                elif layer == 'cross_attn':
                    query = self.attentions[attn_index](
                        query,
                        key,
                        value,
                        identity if self.pre_norm else None,  # residual
                        query_pos=query_pos if query_pos is not None else bev_pos,
                        key_pos=key_pos,
                        reference_points=ref_3d,
                        reference_points_cam=reference_points_cam,
                        mask=mask,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=key_padding_mask,
                        spatial_shapes=spatial_shapes,
                        level_start_index=level_start_index,
                        bev_mask=mask,  # bev_mask is passed as mask
                        **kwargs)
                    attn_index += 1
                    identity = query
                
                elif layer == 'ffn':
                    query = self.ffns[ffn_index](
                        query, identity if self.pre_norm else None)
                    ffn_index += 1
            
            return query