import numpy as np
import torch
import torch.nn.functional as F

class UnitGaussianNormalizer(object):
    '''
    Normalize input data to have zero mean and unit variance.
    
    This normalizer is designed for function data where the last dimension
    typically represents channels (e.g., physical quantities, coordinates).
    It supports three normalization modes:
        1. Global: Normalize all channels together using global statistics
        2. Per-channel: Normalize each channel independently
        3. Selective: Normalize only specified channels, leave others untouched

            
        Example:
            >>> # Create normalizer for 1000 samples of 128x128 grid with 3 channels
            >>> data = torch.randn(1000, 128, 128, 3)
            >>> normalizer = UnitGaussianNormalizer(
            ...     data, 
            ...     normalization_dim=[0, 1, 2],  # Normalize across batch, height, width
            ...     non_normalized_dim=1          # Keep the last channel (e.g., coordinates) unchanged
            ... )
            >>> 
            >>> # Normalize data
            >>> data_norm = normalizer.encode(data)
            >>> 
            >>> # Later: Denormalize predictions
            >>> predictions = model(data_norm)
            >>> predictions_denorm = normalizer.decode(predictions)
        '''
        
        
    def __init__(self, x, normalization_dim = [], non_normalized_dim = 0, eps=1.0e-5):
        """
        Initialize the normalizer with statistics computed from input data.
        
        Args:
            x: Input tensor of shape (..., channels). The last dimension 
                is assumed to be the channel dimension.
                
            normalization_dim: List of dimensions to average over when 
                computing mean and std. Common patterns:
                - [] or None: Global normalization (use all dimensions)
                - [0,1,2]: Normalize across batch, height, width dimensions (normalize together)

            non_normalized_dim: Number of channels at the end to exclude from normalization. 
                Useful for: Coordinates (x, y) that should remain unchanged
                Default: 0 (normalize all channels)
            eps: Small constant added to standard deviation for numerical 
                stability. Prevents division by zero.
        
        Note:
            The normalization is applied only to the first 
            (n_channels - non_normalized_dim) channels. The remaining
            channels (last non_normalized_dim dimensions) are left unchanged
            during encode/decode operations.
        """
        
        super(UnitGaussianNormalizer, self).__init__()

        self.non_normalized_dim = non_normalized_dim
        self.mean = torch.mean(x[...,0:x.shape[-1] - non_normalized_dim],  dim=normalization_dim)
        self.std  = torch.std(x[...,0:x.shape[-1]  - non_normalized_dim],  dim=normalization_dim)
        self.eps  = eps

    def encode(self, x, inplace: bool = False):
        """
        Normalize input data to zero mean and unit variance.
        
        Applies transformation: x_norm = (x - mean) / std
        
        Args:
            x: Input tensor to normalize. Must have same channel dimension
               as the data used for initialization.
            inplace: If True, modify input tensor in-place to save memory.
                    If False, return a new tensor.
        
        Returns:
            Normalized tensor with same shape as input.
        
        Example:
            >>> normalizer = UnitGaussianNormalizer(data, normalization_dim=[0])
            >>> normalized = normalizer.encode(new_data)  # Returns new tensor
            >>> normalizer.encode(new_data, inplace=True)  # Modifies new_data in-place
        """
        std = self.std + self.eps 
        mean = self.mean
        y = x if inplace else x.clone()
        y[...,0:x.shape[-1] - self.non_normalized_dim] = (x[...,0:x.shape[-1]-self.non_normalized_dim] - mean) / std
        return y
    

    def decode(self, x, inplace: bool = False):
        """
        Denormalize data back to original scale.
        
        Reverses the normalization: x_denorm = (x_norm * std) + mean
        
        Args:
            x: Normalized tensor to denormalize.
            inplace: If True, modify input tensor in-place. If False,
                    return a new tensor.
        
        Returns:
            Denormalized tensor with same shape as input, restored to
            original scale.
        
        Example:
            >>> normalized = normalizer.encode(data)
            >>> # ... model prediction ...
            >>> prediction = model(normalized)
            >>> original_scale = normalizer.decode(prediction)
        """
        std = self.std + self.eps 
        mean = self.mean
        y = x if inplace else x.clone()
        y[...,0:x.shape[-1] - self.non_normalized_dim] = (x[...,0:x.shape[-1]-self.non_normalized_dim] * std) + mean
        return y
    
    
    def to(self, device):
        """
        Move the normalizer's statistics to the specified device.
        
        This method allows you to transfer the normalizer to GPU or CPU,
        which is necessary when processing data on different devices.
        
        Args:
            device: Target device (e.g., 'cuda', 'cuda:0', 'cpu', or torch.device)
        
        Returns:
            Self reference for method chaining.
        
        Example:
            >>> normalizer = UnitGaussianNormalizer(data)
            >>> normalizer.to('cuda')  # Move to GPU
            >>> # Now can normalize GPU tensors
            >>> gpu_data = gpu_tensor.to('cuda')
            >>> normalized = normalizer.encode(gpu_data)
        """
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        

