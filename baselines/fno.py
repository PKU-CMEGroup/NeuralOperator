"""
Fourier Neural Operator (FNO) Implementation for 1D and 2D Problems

This module implements the Fourier Neural Operator architecture as described in 
"Fourier Neural Operator for Parametric Partial Differential Equations" by Li et al.

Key concepts:
- Spectral convolution in Fourier space
- Learnable weight matrices for Fourier modes
- Combined with local convolution for better expressivity
"""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utility.adam import Adam
from utility.losses import LpLoss
from utility.normalizer import UnitGaussianNormalizer
from timeit import default_timer


# ============================================================================
# Utility Functions
# ============================================================================

def add_padding(x, pad_nums):
    """
    Add zero padding to the spatial dimensions of the input tensor.
    
    Args:
        x: Input tensor of shape (batch, channels, spatial_dims...)
        pad_nums: List of padding amounts for each spatial dimension
                  (ordered from last dimension to first in F.pad convention)
    
    Returns:
        Padded tensor with same number of dimensions
    """
    
    if x.ndim == 3:  # fourier1d
        res = F.pad(x, [0, pad_nums[0]], "constant", 0)
    elif x.ndim == 4:  # fourier2d
        res = F.pad(x, [0, pad_nums[1], 0, pad_nums[0]], "constant", 0)
    elif x.ndim == 5:  # fourier3d
        res = F.pad(x, [0, pad_nums[2], 0, pad_nums[1], 0, pad_nums[0]], "constant", 0)
    elif x.ndim == 6:  # fourier4d
        res = F.pad(
            x,
            [0, pad_nums[3], 0, pad_nums[2], 0, pad_nums[1], 0, pad_nums[0]],
            "constant",
            0,
        )
    else:
        print("error : x.ndim = ", x.ndim)

    return res


def remove_padding(x, pad_nums):
    """
    Remove zero padding from the spatial dimensions of the input tensor.
    
    Args:
        x: Input tensor of shape (batch, channels, spatial_dims...)
        pad_nums: List of padding amounts for each spatial dimension
    
    Returns:
        Tensor with padding removed
    """
    if x.ndim == 3:  # fourier1d
        res = x[..., : (None if pad_nums[0] == 0 else -pad_nums[0])]

    elif x.ndim == 4:  # fourier2d
        res = x[
            ...,
            : (None if pad_nums[0] == 0 else -pad_nums[0]),
            : (None if pad_nums[1] == 0 else -pad_nums[1]),
        ]

    elif x.ndim == 5:  # fourier3d
        res = x[
            ...,
            : (None if pad_nums[0] == 0 else -pad_nums[0]),
            : (None if pad_nums[1] == 0 else -pad_nums[1]),
            : (None if pad_nums[2] == 0 else -pad_nums[2]),
        ]

    elif x.ndim == 6:  # fourier4d
        res = x[
            ...,
            : (None if pad_nums[0] == 0 else -pad_nums[0]),
            : (None if pad_nums[1] == 0 else -pad_nums[1]),
            : (None if pad_nums[2] == 0 else -pad_nums[2]),
            : (None if pad_nums[3] == 0 else -pad_nums[3]),
        ]

    else:
        print("error : x.ndim = ", x.ndim)

    return res

def _get_act(act):
    if act == "tanh":
        return nn.Tanh()
    elif act == "gelu":
        return nn.GELU()
    elif act == "relu":
        return nn.ReLU(inplace=True)   # 与原 F.relu_ 行为一致
    elif act == "elu":
        return nn.ELU(inplace=True)
    elif act == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    elif act == "none":
        return None   # 不加激活
    else:
        raise ValueError(f"{act} is not supported")

@torch.jit.script
def compl_mul1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Complex-valued matrix multiplication for 1D Fourier modes.
    
    Args:
        a: Input tensor of shape (batch, in_channels, modes)
        b: Weight tensor of shape (in_channels, out_channels, modes)
    
    Returns:
        Output tensor of shape (batch, out_channels, modes)
    
    Note:
        The multiplication is element-wise along the mode dimension,
        which corresponds to the convolution theorem in Fourier space.
    """
    # Einstein summation: bix, iox -> box
    # b: batch, i: in_channels, x: modes
    # i: in_channels, o: out_channels, x: modes
    res = torch.einsum("bix,iox->box", a, b)
    return res


@torch.jit.script
def compl_mul2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Complex-valued matrix multiplication for 2D Fourier modes.
    
    Args:
        a: Input tensor of shape (batch, in_channels, modes1, modes2)
        b: Weight tensor of shape (in_channels, out_channels, modes1, modes2)
    
    Returns:
        Output tensor of shape (batch, out_channels, modes1, modes2)
    """
    # Einstein summation: bixy, ioxy -> boxy
    res = torch.einsum("bixy,ioxy->boxy", a, b)
    return res


# ============================================================================
# Spectral Convolution Layers
# ============================================================================

class SpectralConv1d(nn.Module):
    """
    1D Spectral Convolution Layer (Fourier layer)
    
    This layer performs:
    1. Real FFT of the input to Fourier space
    2. Linear transformation of Fourier coefficients (learnable)
    3. Inverse FFT back to physical space
    
    The linear transformation in Fourier space corresponds to convolution
    in physical space via the Convolution Theorem.
    """
    def __init__(self, in_channels, out_channels, modes1):
        """
        Initialize 1D Spectral Convolution layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes1: Number of Fourier modes to retain (k_max)
                   Should be <= floor(N/2) + 1 where N is spatial resolution
        """
        
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        # Initialize weights with scaling to maintain variance
        # Shape: (in_channels, out_channels, modes1) - complex-valued
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def forward(self, x):
        """
        Forward pass of 1D spectral convolution.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, spatial_dim)
        
        Returns:
            Output tensor of shape (batch_size, out_channels, spatial_dim)
        """
        batchsize = x.shape[0]
        
        # Step 1: Real FFT along the spatial dimension
        # rfftn computes FFT for real inputs, returns only positive frequencies
        # Shape: (batch, in_channels, spatial_dim//2 + 1)
        x_ft = torch.fft.rfftn(x, dim=[2])

        # Step 2: Apply learnable weights to Fourier coefficients
        # Initialize output Fourier tensor with zeros
        out_ft = torch.zeros(
            batchsize,
            self.in_channels,
            x.size(-1) // 2 + 1,    # Number of positive frequencies
            device=x.device,
            dtype=torch.cfloat,
        )
        
        # Multiply the first `modes1` Fourier coefficients with weights
        # This is the core spectral operation
        out_ft[:, :, : self.modes1] = compl_mul1d(
            x_ft[:, :, : self.modes1], self.weights1
        )
        # Note: Higher frequencies (modes1 to end) are set to zero
        # This acts as a low-pass filter

        # Step 3: Inverse FFT back to physical space
        # irfftn expects the full set of positive frequencies
        x = torch.fft.irfftn(out_ft, s=[x.size(-1)], dim=[2])
        return x


class SpectralConv2d(nn.Module):
    """
    2D Spectral Convolution Layer (Fourier layer)
    
    This layer performs convolution via multiplication in Fourier space.
    For 2D, we keep modes in both dimensions and use two sets of weights
    for symmetric handling of positive and negative frequencies.
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        """
        Initialize 2D Spectral Convolution layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes1: Number of Fourier modes to retain in first dimension (height)
            modes2: Number of Fourier modes to retain in second dimension (width)
        """
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        
        # Initialize two sets of weights:
        # - weights1: for positive frequencies (0 to modes1, 0 to modes2)
        # - weights2: for negative frequencies (-modes1 to -1, 0 to modes2)
        # This accounts for the symmetry of real-valued FFT
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )

    def forward(self, x):
        """
        Forward pass of 2D spectral convolution.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
        
        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        batchsize = x.shape[0]
                
        # Step 1: Real FFT along spatial dimensions (height, width)
        # Shape: (batch, in_channels, height, width//2 + 1)
        x_ft = torch.fft.rfftn(x, dim=[2, 3])

        # Step 2: Apply learnable weights
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),           # height dimension
            x.size(-1) // 2 + 1,  # width dimension (positive frequencies)
            device=x.device,
            dtype=torch.cfloat,
        )
        
        # Handle positive frequencies: indices [0:modes1, 0:modes2]
        out_ft[:, :, : self.modes1, : self.modes2] = compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        # Handle negative frequencies: indices [-modes1:, 0:modes2]
        # These correspond to frequencies with negative wavenumber in the first dimension
        out_ft[:, :, -self.modes1 :, : self.modes2] = compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        # Step 3: Inverse FFT back to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)), dim=[2, 3])
        return x
    

class StructuredNN(nn.Module):
    def __init__(self, in_dim, mid_layers, out_dim, act=nn.ReLU()):
        """
        Network structure: input -> mid_layers[0] -> ... -> mid_layers[-1] -> output
        
        Args:
            in_dim: Input dimension
            mid_layers: List of middle layer dimensions
            out_dim: Output dimension
            act: Activation function
        """
        super(StructuredNN, self).__init__()

        self.act = act
        self.mid_layers = mid_layers

        # Build the middle layers
        layers = []
        
        if len(mid_layers) == 0:
            # If no middle layers, directly connect input to output
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            # Input to first middle layer
            layers.append(nn.Linear(in_dim, mid_layers[0]))
            layers.append(self.act)
            # Intermediate middle layers
            for i in range(len(mid_layers) - 1):
                layers.append(nn.Linear(mid_layers[i], mid_layers[i+1]))
                layers.append(self.act)
            # Final output layer
            layers.append(nn.Linear(mid_layers[-1], out_dim))
            
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# ============================================================================
# Fourier Neural Operator (FNO) Models
# ============================================================================

class FNO1d(nn.Module):
    """
    1D Fourier Neural Operator
    
    Architecture:
    - Input lifting: Linear layer to expand to channel dimension
    - Multiple Fourier layers: Each combines spectral convolution + local convolution
    - Output projection: Linear layers to map to desired output dimension
    
    The operator learns mappings between infinite-dimensional function spaces
    by parameterizing the integral kernel in Fourier space.
    """
    def __init__(self,
                 modes,
                 layers=[32,32,32,32],
                 proj_layers=[128],
                 in_dim=2, out_dim=1,
                 act='gelu',
                 pad_ratio=0, 
                 cnn_kernel_size=1):
        """
        Initialize 1D FNO.
        
        Args:
            modes: Number of Fourier modes to retain
            layers: List of channel sizes for each Fourier layer, the first one is the input channel size.
                   Default, [32,32,32,32] (3 FNO layers of same width)
            proj_layers: List of middle layers width for the projection operator
                   Default, [128] (1 layers of same width)
            in_dim: Input feature dimension
            out_dim: Output feature dimension
            act: Activation function name
            pad_ratio: Fraction of input size to pad (helps with periodic boundary approximation)
            cnn_kernel_size: Kernel size for local (W) convolution
        """
        super(FNO1d, self).__init__()

        self.modes1 = modes

        self.pad_ratio = pad_ratio
        
        # Layer 1: Lift input to higher-dimensional channel space
        # Input shape: (batch, spatial_points, in_dim)
        # Output shape: (batch, spatial_points, layers[0])
        self.fc0 = nn.Linear(in_dim, layers[0])  # input channel is 2: (a(x), x)

        # Fourier layers: Each contains a spectral convolution and local convolution
        # The combination (W + K) allows both global (spectral) and local patterns
        self.sp_convs = nn.ModuleList([SpectralConv1d(
            in_size, out_size, num_modes) for in_size, out_size, num_modes in zip(layers, layers[1:], self.modes1)])

        # Local convolution layers (W operator) - captures local features
        # Using padding='same' via manual padding to preserve spatial dimensions
        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, kernel_size=cnn_kernel_size, padding=(cnn_kernel_size//2))
                                 for in_size, out_size in zip(layers, layers[1:])])
            
        self.act = _get_act(act)

        # Output projection layers
        self.proj = StructuredNN(in_dim=layers[-1], mid_layers=proj_layers, out_dim=out_dim, act=self.act)


    def forward(self, x):
        """
        Forward pass of 1D FNO.
        
        Args:
            x: Input tensor of shape (batch_size, nx_in, in_dim)
               Typically contains (function values, coordinates)
        
        Returns:
            Output tensor of shape (batch_size, nx_out, out_dim)
        """
        
        length = len(self.ws)
                
        # Step 1: Lift input to channel space
        # Shape: (batch, nx, in_dim) -> (batch, nx, layers[0])
        x = self.fc0(x)
            
        # Step 2: Permute to channel-first format for convolution
        # (batch, nx, channels) -> (batch, channels, nx)
        x = x.permute(0, 2, 1)
        
        # Step 3: Apply padding (if specified)
        # Padding helps mitigate boundary effects when domain isn't perfectly periodic
        pad_nums = [math.floor(self.pad_ratio * x.shape[-1])]
        x = add_padding(x, pad_nums=pad_nums)

        # Step 4: Apply Fourier layers
        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            # Spectral convolution (K operator) - captures global patterns
            x1 = speconv(x)
            # Local convolution (W operator) - captures local patterns
            x2 = w(x)
            # Combine: u' = (W + K)(u)
            x = x1 + x2
            # Apply activation (except after the last layer)
            if self.act is not None and i != length - 1:
                x = self.act(x)
                
        # Step 5: Remove padding
        x = remove_padding(x, pad_nums=pad_nums)
        
        # Step 6: Permute back to channel-last format
        # (batch, channels, nx) -> (batch, nx, channels)
        x = x.permute(0, 2, 1)

        # Step 7: Project to output dimension
        x = self.proj(x)
        
        
        return x



class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator
    
    Architecture similar to FNO1d but for 2D spatial domains.
    Handles input of the form (a(x,y), x, y) and outputs solution u(x,y).
    """
    def __init__(
        self,
        modes1,  
        modes2,
        layers=[32,32,32,32],
        proj_layers=[128],
        in_dim=3,
        out_dim=1,
        act="gelu",
        pad_ratio=0,
        cnn_kernel_size=1,
    ):
        super(FNO2d, self).__init__()

        """
        Initialize 2D FNO.
        
        Args:
            modes1: Number of Fourier modes in height dimension
            modes2: Number of Fourier modes in width dimension
            layers: List of channel sizes for each Fourier layer, the first one is the input channel size.
                   Default, [32,32,32,32] (3 FNO layers of same width)
            proj_layers: List of middle layers width for the projection operator
                   Default, [128] (1 layers of same width)
            in_dim: Input feature dimension (typically 3: coefficient + x + y)
            out_dim: Output feature dimension
            act: Activation function name
            pad_ratio: Fraction of input size to pad (helps with boundary approximation)
            cnn_kernel_size: Kernel size for local (W) convolution
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.pad_ratio = pad_ratio
        
        # Layer 1: Lift input to channel space
        # Input shape: (batch, height, width, in_dim)
        self.fc0 = nn.Linear(in_dim, layers[0])

        # Fourier layers
        self.sp_convs = nn.ModuleList(
            [
                SpectralConv2d(in_size, out_size, mode1_num, mode2_num)
                for in_size, out_size, mode1_num, mode2_num in zip(
                    self.layers, self.layers[1:], self.modes1, self.modes2
                )
            ]
        )
        # Local convolution layers (2D Conv2d)
        self.ws = nn.ModuleList(
            [
                nn.Conv2d(in_size, out_size, kernel_size=(cnn_kernel_size,cnn_kernel_size), padding=(cnn_kernel_size//2,cnn_kernel_size//2))
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )

        self.act = _get_act(act)

        # Output projection layers
        self.proj = StructuredNN(in_dim=layers[-1], mid_layers=proj_layers, out_dim=out_dim, act=self.act)



        

    def forward(self, x):
        """
        Forward pass of 2D FNO.
        
        Args:
            x: Input tensor of shape (batch_size, height, width, in_dim)
               Typically contains (coefficient function a(x,y), x, y)
        
        Returns:
            Output tensor of shape (batch_size, height, width, out_dim)
        """
        
        length = len(self.ws)
        
        
        # Step 1: Lift input to channel space
        # (batch, height, width, in_dim) -> (batch, height, width, layers[0])
        x = self.fc0(x)
        
        # Step 2: Permute to channel-first format
        # (batch, height, width, channels) -> (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2)
        # Step 3: Apply padding
        pad_nums = [
            math.floor(self.pad_ratio * x.shape[-2]),
            math.floor(self.pad_ratio * x.shape[-1]),
        ]
        x = add_padding(x, pad_nums=pad_nums)

        # Step 4: Apply Fourier layers
        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x)
            x = x1 + x2
            # Apply activation (except after last layer)
            if self.act is not None and i != length - 1:
                x = self.act(x)

        # Step 5: Remove padding
        x = remove_padding(x, pad_nums=pad_nums)

        # Step 6: Permute back to channel-last format
        # (batch, channels, height, width) -> (batch, height, width, channels)
        x = x.permute(0, 2, 3, 1)

        # Step 7: Project to output dimension
        x = self.proj(x)
        
        return x




# ============================================================================
# Training Function
# ============================================================================
def FNO_train(x_train, y_train, x_test, y_test, config, model, save_model_name="./FNO_model"):
    """
    Training function for FNO models.
    
    This function handles:
    - Data normalization
    - Training loop with configurable optimizer and scheduler
    - Evaluation on test set
    - Model checkpointing
    
    Args:
        x_train: Training input data (n_train, n_x, n_channels)
        y_train: Training target data (n_train, n_x, n_channels)
        x_test: Test input data (n_test, n_x, n_channels)
        y_test: Test target data (n_test, n_x, n_channels)
        config: Dictionary containing training configuration with keys:
            - train: sub-dictionary with:
                - normalization_x, normalization_y: bool for data normalization
                - normalization_dim_x, normalization_dim_y: dimensions to normalize
                - non_normalized_dim_x, non_normalized_dim_y: dimensions to skip
                - batch_size: Training batch size
                - base_lr: Learning rate
                - weight_decay: Weight decay for optimizer
                - scheduler: Learning rate scheduler (currently supports 'OneCycleLR')
                - epochs: Number of training epochs
        model: FNO model instance to train
        save_model_name: Path to save model checkpoints (without extension)
    
    Returns:
        train_rel_l2_losses: List of relative L2 training losses per epoch
        test_rel_l2_losses: List of relative L2 test losses per epoch
        test_l2_losses: List of absolute L2 test losses per epoch
    """
    
    n_train, n_test = x_train.shape[0], x_test.shape[0]
    train_rel_l2_losses = []
    test_rel_l2_losses = []
    test_l2_losses = []
    normalization_x, normalization_y = config["train"]["normalization_x"], config["train"]["normalization_y"]
    normalization_dim_x, normalization_dim_y = config["train"]["normalization_dim_x"], config["train"]["normalization_dim_y"]
    non_normalized_dim_x, non_normalized_dim_y = config["train"]["non_normalized_dim_x"], config["train"]["non_normalized_dim_y"]
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data normalization (important for stable training)
    if normalization_x:
        x_normalizer = UnitGaussianNormalizer(x_train, non_normalized_dim = non_normalized_dim_x, normalization_dim=normalization_dim_x)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)
        x_normalizer.to(device)
        
    if normalization_y:
        y_normalizer = UnitGaussianNormalizer(y_train, non_normalized_dim = non_normalized_dim_y, normalization_dim=normalization_dim_y)
        y_train = y_normalizer.encode(y_train)
        y_test = y_normalizer.encode(y_test)
        y_normalizer.to(device)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), 
                                               batch_size=config['train']['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), 
                                               batch_size=config['train']['batch_size'], shuffle=False)
    
    
    # Setup optimizer
    optimizer = Adam(model.parameters(), betas=(0.9, 0.999),
                     lr=config['train']['base_lr'], weight_decay=config['train']['weight_decay'])
    
    # Setup learning rate scheduler
    if config["train"]["scheduler"] == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config['train']['base_lr'], 
            div_factor=2, final_div_factor=100,pct_start=0.2,
            steps_per_epoch=len(train_loader), epochs=config['train']['epochs'])
    else:
        print("Scheduler ", config['train']['scheduler'], " has not implemented.")

    model.train()
    
    # Loss function: Relative L2 loss
    myloss = LpLoss(d=1, p=2, size_average=False)
    epochs = config['train']['epochs']

    # Training loop
    for ep in range(epochs):
        t1 = default_timer()
        train_rel_l2 = 0

        # Training phase
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            batch_size_ = x.shape[0]
            optimizer.zero_grad()
            out = model(x) 
            # Decode if normalized
            if normalization_y:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)

            loss = myloss(out.view(batch_size_,-1), y.view(batch_size_,-1))
            loss.backward()

            optimizer.step()
            scheduler.step()
            train_rel_l2 += loss.item()

        # Evaluation phase
        test_l2 = 0
        test_rel_l2 = 0
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                batch_size_ = x.shape[0]
                out = model(x) 

                if normalization_y:
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)

                test_rel_l2 += myloss(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()
                test_l2 += myloss.abs(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()




        
        # Average losses over dataset size
        train_rel_l2/= n_train
        test_l2 /= n_test
        test_rel_l2/= n_test
        
        # Store losses
        train_rel_l2_losses.append(train_rel_l2)
        test_rel_l2_losses.append(test_rel_l2)
        test_l2_losses.append(test_l2)
    
        t2 = default_timer()
        
        # Print progress
        print("Epoch : ", ep, " Time: ", round(t2-t1,3), " Rel. Train L2 Loss : ", train_rel_l2, " Rel. Test L2 Loss : ", test_rel_l2, " Test L2 Loss : ", test_l2, flush=True)
        
        # Save checkpoint every 100 epochs and at final epoch
        if (ep %100 == 99) or (ep == epochs -1):    
            if save_model_name:
                torch.save(model.state_dict(), save_model_name + ".pth")
    
    
    return train_rel_l2_losses, test_rel_l2_losses, test_l2_losses
