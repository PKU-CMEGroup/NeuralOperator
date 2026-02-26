import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utility.adam import Adam
from utility.losses import LpLoss
from utility.normalizer import UnitGaussianNormalizer


class SinCosPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani et al.) for length-N token sequences.

    Args:
        d_model: Embedding dimension of each token.
        max_len: Maximum supported sequence length N.

    Returns:
        In forward(), returns x + PE where PE is the sinusoidal positional encoding.

    Math:
        PE[pos, 2i]   = sin(pos / 10000^{2i/d_model})
        PE[pos, 2i+1] = cos(pos / 10000^{2i/d_model})

    Require:
        0 < N <= max_len in forward().
    """
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        self.register_buffer("pe", pe)  # [max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token embeddings of shape [B, N, d_model].

        Returns:
            Tensor of shape [B, N, d_model] with positional encoding added.

        Require:
            0 < N <= max_len (set in __init__).
        """
        N = x.size(1)
        return x + self.pe[:N].unsqueeze(0).to(x.dtype)



class Transformer(nn.Module):
    """
    Encoder-only Transformer that maps a field on a grid.

    Args:
        in_channels:  Number of input channels per spatial point (token).
        out_channels: Number of output channels per spatial point (token).
        d_model:      Transformer embedding dimension.
        nhead:        Number of attention heads.
        num_layers:   Number of TransformerEncoder layers.
        dim_feedforward: Hidden width of the MLP/FFN in each layer.
        dropout:      Dropout probability in encoder layers.
        max_len:      Maximum supported token length N for positional encoding.
        use_coord_channel: If True, append a normalized coordinate x in [0,1) to each token.

    Returns:
        Forward maps u -> y with:
          u: [B, N, in_channels]
          y: [B, N, out_channels]

    Notes:
        - Tokens correspond to spatial points.
        - batch_first=True so tensors are [B, N, d_model].
        - norm_first=True uses pre-norm Transformer blocks (often more stable).
    """
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        max_len: int = 4096,
        use_coord_channel: bool = True,
    ):
        super().__init__()
        self.use_coord_channel = use_coord_channel

        # Optionally append coordinate x as an extra channel
        input_dim = in_channels + (1 if use_coord_channel else 0)

        # Project raw token features -> d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # Fixed positional encoding
        self.pos_enc = SinCosPositionalEncoding(d_model, max_len=max_len)

        # Transformer encoder layer: MHSA + FFN + residual/LN
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        # Stack layers
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Final normalization + projection to output channels
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, out_channels)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: Input field on grid, shape [B, N, in_channels].

        Returns:
            Predicted field on grid, shape [B, N, out_channels].

        Require:
            If use_coord_channel=True, N must be the intended spatial resolution
            (used to generate coordinate channel).
        """
        B, N, C = u.shape
        if self.use_coord_channel:
            # normalized coordinate in [0, 1)
            x = torch.linspace(0.0, 1.0, N, device=u.device, dtype=u.dtype).view(1, N, 1).repeat(B, 1, 1)
            inp = torch.cat([u, x], dim=-1)
        else:
            inp = u

        h = self.input_proj(inp)        # [B, N, d_model]
        h = self.pos_enc(h)             # add positional encoding
        h = self.encoder(h)             # [B, N, d_model]
        h = self.norm(h)
        y = self.output_proj(h)         # [B, N, 2]
        return y





# x_train, y_train, x_test, y_test are [n_data, n_x, n_channel] arrays
def Transformer_train(x_train, y_train, x_test, y_test, config, model, save_model_name="./Transformer_model"):
    n_train, n_test = x_train.shape[0], x_test.shape[0]
    train_rel_l2_losses = []
    test_rel_l2_losses = []
    test_l2_losses = []
    normalization_x, normalization_y = config["train"]["normalization_x"], config["train"]["normalization_y"]
    normalization_dim_x, normalization_dim_y = config["train"]["normalization_dim_x"], config["train"]["normalization_dim_y"]
    non_normalized_dim_x, non_normalized_dim_y = config["train"]["non_normalized_dim_x"], config["train"]["non_normalized_dim_y"]
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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


    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), 
                                               batch_size=config['train']['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), 
                                               batch_size=config['train']['batch_size'], shuffle=False)
    
    
    # Load from checkpoint
    optimizer = Adam(model.parameters(), betas=(0.9, 0.999),
                     lr=config['train']['base_lr'], weight_decay=config['train']['weight_decay'])
    
    if config["train"]["scheduler"] == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config['train']['base_lr'], 
            div_factor=2, final_div_factor=100, pct_start=0.2,
            steps_per_epoch=len(train_loader), epochs=config['train']['epochs'])
    else:
        print("Scheduler ", config['train']['scheduler'], " has not implemented.")

    model.train()
    myloss = LpLoss(d=1, p=2, size_average=False)

    epochs = config['train']['epochs']


    for ep in range(epochs):
        train_rel_l2 = 0

        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            batch_size_ = x.shape[0]
            optimizer.zero_grad()
            out = model(x) #.reshape(batch_size_,  -1)
            if normalization_y:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)

            loss = myloss(out.view(batch_size_,-1), y.view(batch_size_,-1))
            loss.backward()

            optimizer.step()
            scheduler.step()
            train_rel_l2 += loss.item()

        test_l2 = 0
        test_rel_l2 = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                batch_size_ = x.shape[0]
                out = model(x) #.reshape(batch_size_,  -1)

                if normalization_y:
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)

                test_rel_l2 += myloss(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()
                test_l2 += myloss.abs(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()




        

        train_rel_l2/= n_train
        test_l2 /= n_test
        test_rel_l2/= n_test
        
        train_rel_l2_losses.append(train_rel_l2)
        test_rel_l2_losses.append(test_rel_l2)
        test_l2_losses.append(test_l2)
    

        if (ep %10 == 0) or (ep == epochs -1):
            print("Epoch : ", ep, " Rel. Train L2 Loss : ", train_rel_l2, " Rel. Test L2 Loss : ", test_rel_l2, " Test L2 Loss : ", test_l2, flush=True)
            torch.save(model, save_model_name)
    
    
    return train_rel_l2_losses, test_rel_l2_losses, test_l2_losses
