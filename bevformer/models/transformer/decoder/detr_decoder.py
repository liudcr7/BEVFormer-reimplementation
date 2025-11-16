"""DETR-style decoder for BEVFormer."""
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
            """Inverse sigmoid function."""
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
    """Detection transformer decoder for BEVFormer.
    
    This decoder processes object queries through multiple layers, where each layer:
    1. Applies self-attention between queries
    2. Applies cross-attention with BEV features
    3. Optionally updates reference points using regression branches (box refinement)
    
    Args:
        return_intermediate (bool): Whether to return intermediate outputs from all layers.
            If True, returns stacked outputs from all layers; otherwise returns only final output.
    """
    
    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(DetectionTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False
    
    def _update_reference_points(self, reg_pred: torch.Tensor, 
                                 reference_points: torch.Tensor) -> torch.Tensor:
        """Update reference points with regression predictions.
        
        Args:
            reg_pred: Regression predictions from reg_branch
                - Shape: [bs, num_query, code_size]
                - Format: [cx, cy, cz, log(w), log(l), log(h), sin(rot), cos(rot), vx, vy]
            reference_points: Current reference points
                - Shape: [bs, num_query, 3]
                - Format: (x, y, z) normalized to [0, 1]
        
        Returns:
            Updated reference points with same shape [bs, num_query, 3]
        """
        assert reference_points.shape[-1] == 3, \
            f"reference_points should have shape [bs, num_query, 3], got {reference_points.shape}"
        
        new_ref = torch.zeros_like(reference_points)
        # Update x, y coordinates (BEV plane) using cx, cy from regression
        new_ref[..., :2] = reg_pred[..., :2] + inverse_sigmoid(reference_points[..., :2])
        # Update z coordinate (height) using cz from regression (index 4)
        new_ref[..., 2:3] = reg_pred[..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
        
        # Normalize back to [0, 1] range and detach for next iteration
        return new_ref.sigmoid().detach()
    
    def forward(self,
                query: torch.Tensor,
                *args,
                reference_points: torch.Tensor = None,
                reg_branches: nn.ModuleList = None,
                key_padding_mask: torch.Tensor = None,
                **kwargs) -> tuple:
        """Forward function for DetectionTransformerDecoder.
        
        Args:
            query: Object queries
                - Shape: [num_query, bs, embed_dims] (sequence-first format)
                - num_query: number of object queries (typically 900)
            reference_points: Initial reference points for bounding box regression
                - Shape: [bs, num_query, 3]
                - Format: (x, y, z) normalized to [0, 1] range
                - x, y: BEV plane coordinates, z: height coordinate
            reg_branches: Regression branches for refining reference points (optional)
                - Type: nn.ModuleList of regression heads
                - Only passed when with_box_refine=True
                - Each head: [bs, num_query, embed_dims] -> [bs, num_query, code_size]
            key_padding_mask: Key padding mask (optional)
                - Used to mask out invalid positions in attention
            
        Returns:
            tuple: 
                - If return_intermediate=False: (output, reference_points)
                    - output: [num_query, bs, embed_dims] - final decoder output
                    - reference_points: [bs, num_query, 3] - final reference points
                - If return_intermediate=True: (intermediate_states, intermediate_references)
                    - intermediate_states: [num_layers, num_query, bs, embed_dims]
                    - intermediate_references: [num_layers, bs, num_query, 3]
        """
        output = query
        intermediate_states = []
        intermediate_refs = []
        
        for lid, layer in enumerate(self.layers):
            # Prepare reference points for layer: extract x, y and add level dimension
            # Layer expects [bs, num_query, num_levels, 2] format
            ref_input = reference_points[..., :2].unsqueeze(2)  # [bs, num_query, 1, 2]
            
            # Apply decoder layer
            output = layer(
                output,
                *args,
                reference_points=ref_input,
                key_padding_mask=key_padding_mask,
                **kwargs
            )
            
            # Convert to batch-first for regression branches
            output_batch = output.permute(1, 0, 2)  # [bs, num_query, embed_dims]
            
            # Update reference points if box refinement is enabled
            if reg_branches is not None:
                reg_pred = reg_branches[lid](output_batch)  # [bs, num_query, code_size]
                reference_points = self._update_reference_points(reg_pred, reference_points)
            
            # Convert back to sequence-first format
            output = output_batch.permute(1, 0, 2)  # [num_query, bs, embed_dims]
            
            # Store intermediate outputs if needed
            if self.return_intermediate:
                intermediate_states.append(output)
                intermediate_refs.append(reference_points)
        
        if self.return_intermediate:
            return torch.stack(intermediate_states), torch.stack(intermediate_refs)
        
        return output, reference_points
