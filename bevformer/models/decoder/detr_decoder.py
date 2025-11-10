"""DETR-style decoder for BEVFormer, aligned with original implementation."""
import torch
import torch.nn as nn

try:
    from mmdet.models.utils.transformer import TransformerLayerSequence, inverse_sigmoid
except Exception:
    try:
        from mmdet.models.utils.builder import TransformerLayerSequence
        from mmdet.models.utils.transformer import inverse_sigmoid
    except Exception:
        TransformerLayerSequence = nn.Module
        def inverse_sigmoid(x, eps=1e-5):
            x = x.clamp(min=0, max=1)
            x1 = x.clamp(min=eps)
            x2 = (1 - x).clamp(min=eps)
            return torch.log(x1 / x2)

try:
    from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
except Exception:
    try:
        from mmdet.models.utils.builder import TRANSFORMER_LAYER_SEQUENCE
    except Exception:
        try:
            from mmdet.models.builder import TRANSFORMER_LAYER_SEQUENCE
        except Exception:
            TRANSFORMER_LAYER_SEQUENCE = None


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DetectionTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    
    This decoder matches the original BEVFormer implementation:
    - Inherits from TransformerLayerSequence
    - Supports return_intermediate for multi-layer outputs
    - Iteratively updates reference_points using reg_branches
    - Returns (output, reference_points) or (intermediate_states, intermediate_references)
    
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
    """
    
    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(DetectionTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False
    
    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for DetectionTransformerDecoder.
        
        Args:
            query (Tensor): Input query with shape (num_query, bs, embed_dims).
            reference_points (Tensor): The reference points of offset.
                Has shape (bs, num_query, 4) when as_two_stage,
                otherwise has shape (bs, num_query, 3).
            reg_branches (nn.ModuleList): Used for refining the regression results.
                Only passed when with_box_refine is True, otherwise None.
            key_padding_mask (Tensor): Key padding mask.
            
        Returns:
            tuple: 
                - If return_intermediate=False: (output, reference_points)
                    - output: [num_query, bs, embed_dims]
                    - reference_points: [bs, num_query, 3]
                - If return_intermediate=True: (intermediate_states, intermediate_references)
                    - intermediate_states: [num_layers, num_query, bs, embed_dims]
                    - intermediate_references: [num_layers, bs, num_query, 3]
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        
        for lid, layer in enumerate(self.layers):
            # Prepare reference_points_input for layer
            # Layer expects [bs, num_query, num_levels, 2]
            reference_points_input = reference_points[..., :2].unsqueeze(2)  # [bs, num_query, 1, 2]
            
            # Call layer
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs)
            
            # Convert to batch_first for reg_branches
            output = output.permute(1, 0, 2)  # [bs, num_query, embed_dims]
            
            # Update reference_points if reg_branches is provided
            if reg_branches is not None:
                tmp = reg_branches[lid](output)  # [bs, num_query, code_size]
                
                assert reference_points.shape[-1] == 3, \
                    f"reference_points should have shape [bs, num_query, 3], got {reference_points.shape}"
                
                # Create new reference points
                new_reference_points = torch.zeros_like(reference_points)
                # Update x, y coordinates
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
                # Update z coordinate
                new_reference_points[..., 2:3] = tmp[..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
                
                # Apply sigmoid
                new_reference_points = new_reference_points.sigmoid()
                
                # Detach for next iteration
                reference_points = new_reference_points.detach()
            
            # Convert back to sequence_first format
            output = output.permute(1, 0, 2)  # [num_query, bs, embed_dims]
            
            # Store intermediate outputs if needed
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        
        return output, reference_points
