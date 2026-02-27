import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utility.adam import Adam
from utility.losses import LpLoss
from utility.normalizer import UnitGaussianNormalizer
from timeit import default_timer



class CoordEmbedding(nn.Module):
    """
    Embed physical coordinates into feature vectors.

    This module maps coordinates in R^{coord_dim} (e.g., 1D/2D/3D spatial coordinates)
    to an embedding of dimension d_coord, which can then be concatenated with other
    token features before feeding into a Transformer.

    Two modes are supported:

    1) "linear":
        A simple learned linear projection:
            emb = W coords + b

    2) "fourier":
        Random Fourier features followed by a learned linear projection.
        Let B ∈ R^{coord_dim × num_frequencies} be a fixed random matrix.
        Define:
            xb = coords @ B
            feats = [sin(xb), cos(xb)]   (and optionally coords itself)
            emb = Linear(feats)

        This is helpful for representing high-frequency spatial variation and for
        operator/PDE learning tasks.

    Args:
        coord_dim: Dimension of the coordinate vector at each token (typically 1, 2, or 3).
        d_coord: Output embedding dimension.
        mode: Either "linear" or "fourier".
        num_frequencies: Number of random Fourier frequencies (only used if mode="fourier").
        scale: Frequency scale for the Fourier matrix B (only used if mode="fourier").
               Larger values emphasize higher-frequency variation.
        include_raw: If True, concatenates the raw coords to the Fourier feature vector.

    Returns:
        In forward(), returns coord_emb of shape [B, N, d_coord].

    Require:
        coords must have shape [B, N, coord_dim].
    """
    def __init__(
        self,
        coord_dim: int,
        d_coord: int,
        mode: str = "fourier",
        num_frequencies: int = 16,
        scale: float = 2.0 * math.pi,
        include_raw: bool = True,
    ):
        super().__init__()
        assert mode in ["linear", "fourier"]
        self.coord_dim = coord_dim
        self.mode = mode
        self.include_raw = include_raw

        if mode == "linear":
            self.proj = nn.Linear(coord_dim, d_coord)
        else:
            # fixed random Fourier matrix
            B = torch.randn(coord_dim, num_frequencies) * scale
            self.register_buffer("B", B)  # [coord_dim, num_frequencies]
            in_dim = 2 * num_frequencies + (coord_dim if include_raw else 0)
            self.proj = nn.Linear(in_dim, d_coord)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: Physical coordinates with shape [B, N, coord_dim].

        Returns:
            Embedded coordinates with shape [B, N, d_coord].

        Math (fourier mode):
            xb = coords @ B
            feats = [sin(xb), cos(xb)] (and optionally coords)
            coord_emb = Linear(feats)
        """
        assert coords.dim() == 3 and coords.size(-1) == self.coord_dim, \
            f"coords should be [B,N,{self.coord_dim}] but got {tuple(coords.shape)}"

        if self.mode == "linear":
            return self.proj(coords)

        xb = coords @ self.B                          # [B, N, num_frequencies]
        feats = [torch.sin(xb), torch.cos(xb)]
        if self.include_raw:
            feats = [coords] + feats
        feats = torch.cat(feats, dim=-1)              # [B, N, in_dim]
        return self.proj(feats)                       # [B, N, d_coord]



class Transformer(nn.Module):
    """
    Encoder-only Transformer conditioned on physical coordinates (no positional encoding).

    Forward:
        y = model(u, coords)

    Args:
        in_channels: Number of input channels per token.
        out_channels: Number of output channels per token.
        coord_dim: Dimension of physical coordinates per token (1/2/3).
        d_model: Transformer embedding dimension.
        nhead: Number of attention heads.
        num_layers: Number of TransformerEncoder layers.
        dim_feedforward: Hidden width of the FFN in each encoder layer.
        dropout: Dropout probability.
        coord_mode: Coordinate embedding mode ("linear" or "fourier").
        d_coord: Width of coordinate embedding.
        num_frequencies: Number of Fourier frequencies (only used if coord_mode="fourier").

    Returns:
        In forward(), returns y of shape [B, N, out_channels].

    Require:
        u has shape [B, N, in_channels]
        coords has shape [B, N, coord_dim]
    """
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        coord_dim: int = 1,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        coord_mode: str = "fourier",   # "linear" or "fourier"
        d_coord: int = 64,
        num_frequencies: int = 16,
    ):
        super().__init__()

        # Embed coordinates: [B, N, coord_dim] -> [B, N, d_coord]
        self.coord_emb = CoordEmbedding(
            coord_dim=coord_dim,
            d_coord=d_coord,
            mode=coord_mode,
            num_frequencies=num_frequencies,
        )

        # Project concatenated features [u, coord_emb] -> d_model
        self.input_proj = nn.Linear(in_channels + d_coord, d_model)


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

    def forward(self, u: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: Input token features, shape [B, N, in_channels].
            coords: Physical coordinates, shape [B, N, coord_dim].

        Returns:
            y: Output token features, shape [B, N, out_channels].
        """

        if coords is None:
            raise ValueError("coords must be provided as [B, N, coord_dim].")

        # Coordinate conditioning
        c = self.coord_emb(coords)                 # [B, N, d_coord]

        # Combine field features + coordinate features
        h = torch.cat([u, c], dim=-1)              # [B, N, in_channels + d_coord]
        # Lift to model dimension
        h = self.input_proj(h)                     # [B, N, d_model]

        # Global mixing via self-attention
        h = self.encoder(h)                        # [B, N, d_model]

        # Readout
        h = self.norm(h)
        return self.output_proj(h)                 # [B, N, out_channels]
    





# x_train, y_train, x_test, y_test are [n_data, n_x, n_channel] arrays
def Transformer_train(x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, save_model_name="./Transformer_model"):
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

    node_mask_train, nodes_train = aux_train
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train, node_mask_train, nodes_train), 
                                               batch_size=config['train']['batch_size'], shuffle=True)
    node_mask_test, nodes_test = aux_test
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test, node_mask_test, nodes_test), 
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
        t1 = default_timer()
        train_rel_l2 = 0

        model.train()
        for x, y, node_mask, nodes in train_loader:
            x, y, node_mask, nodes = x.to(device), y.to(device), node_mask.to(device), nodes.to(device)

            batch_size_ = x.shape[0]
            optimizer.zero_grad()
            out = model(x, coords = nodes) #.reshape(batch_size_,  -1)
            if normalization_y:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)

            out = out * node_mask #mask the padded value with 0,(1 for node, 0 for padding)
            loss = myloss(out.view(batch_size_,-1), y.view(batch_size_,-1))
            loss.backward()

            optimizer.step()
            scheduler.step()
            train_rel_l2 += loss.item()

        test_l2 = 0
        test_rel_l2 = 0

        model.eval()
        with torch.no_grad():
            for x, y, node_mask, nodes in test_loader:
                x, y, node_mask, nodes = x.to(device), y.to(device), node_mask.to(device), nodes.to(device)
                batch_size_ = x.shape[0]
                out = model(x, coords = nodes) #.reshape(batch_size_,  -1)

                if normalization_y:
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)

                out = out*node_mask #mask the padded value with 0,(1 for node, 0 for padding)
                
                test_rel_l2 += myloss(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()
                test_l2 += myloss.abs(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()




        

        train_rel_l2/= n_train
        test_l2 /= n_test
        test_rel_l2/= n_test
        
        train_rel_l2_losses.append(train_rel_l2)
        test_rel_l2_losses.append(test_rel_l2)
        test_l2_losses.append(test_l2)
    
        t2 = default_timer()
        print("Epoch : ", ep, " Time: ", round(t2-t1,3), " Rel. Train L2 Loss : ", train_rel_l2, " Rel. Test L2 Loss : ", test_rel_l2, " Test L2 Loss : ", test_l2,flush=True)
        
        if (ep %100 == 99) or (ep == epochs -1):    
            if save_model_name:
                torch.save(model.state_dict(), save_model_name + ".pth")


    
    
    return train_rel_l2_losses, test_rel_l2_losses, test_l2_losses





# x_train, y_train, x_test, y_test are [n_data, n_x, n_channel] arrays
def Transformer_train_multidist(x_train, aux_train, y_train, x_test_list, aux_test_list, y_test_list, config, model, label_test_list, save_model_name="./Transformer_model"):
    assert len(x_test_list) == len(y_test_list) == len(aux_test_list), "The length of x_test_list, y_test_list and aux_test_list should be the same"
    n_distributions = len(x_test_list)
    n_train= x_train.shape[0]
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
        for i in range(n_distributions):
            x_test_list[i] = x_normalizer.encode(x_test_list[i])
        x_normalizer.to(device)
        
    if normalization_y:
        y_normalizer = UnitGaussianNormalizer(y_train, non_normalized_dim = non_normalized_dim_y, normalization_dim=normalization_dim_y)
        y_train = y_normalizer.encode(y_train)
        for i in range(n_distributions):
            y_test_list[i] = y_normalizer.encode(y_test_list[i])
        y_normalizer.to(device)


    node_mask_train, nodes_train = aux_train
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train, node_mask_train, nodes_train), 
                                               batch_size=config['train']['batch_size'], shuffle=True)
    
    test_loaders = []

    for i in range(n_distributions):
        node_mask_test, nodes_test = aux_test_list[i]
        sub_dataset = torch.utils.data.TensorDataset(x_test_list[i], y_test_list[i], node_mask_test, nodes_test)
        sub_loader = torch.utils.data.DataLoader(sub_dataset, batch_size=config['train']['batch_size'], shuffle=False)
        try:
            name = label_test_list[i]
        except:
            name = f"Distribution_{i}"
        test_loaders.append((name, sub_loader))
  
    
    myloss = LpLoss(d=1, p=2, size_average=False)

    optimizer = Adam(model.parameters(), betas=(0.9, 0.999),
                     lr=config['train']['base_lr'], weight_decay=config['train']['weight_decay'])
    
    
    if config["train"]["scheduler"] == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config['train']['base_lr'], 
            div_factor=2, final_div_factor=100, pct_start=0.2,
            steps_per_epoch=len(train_loader), epochs=config['train']['epochs'])
    else:
        print("Scheduler ", config['train']['scheduler'], " has not implemented.")
    
    epochs = config['train']['epochs']
    
    for ep in range(epochs):
        t1 = default_timer()
        train_rel_l2 = 0

        model.train()
        for x, y, node_mask, nodes in train_loader:
            x, y, node_mask, nodes = x.to(device), y.to(device), node_mask.to(device), nodes.to(device)

            batch_size_ = x.shape[0]
            optimizer.zero_grad()
            out = model(x, coords=nodes) #.reshape(batch_size_,  -1)
            if normalization_y:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
            out = out * node_mask #mask the padded value with 0,(1 for node, 0 for padding)
            
            loss = myloss(out.view(batch_size_,-1), y.view(batch_size_,-1))
            loss.backward()

            optimizer.step()
            scheduler.step()
            train_rel_l2 += loss.item()


        test_rel_l2_dict = {}
        test_l2_dict = {}

        model.eval()
        with torch.no_grad():
            for name, loader in test_loaders:
                test_l2 = 0
                test_rel_l2 = 0

                for x, y, node_mask, nodes in loader:
                    x, y, node_mask, nodes = x.to(device), y.to(device), node_mask.to(device), nodes.to(device)

                    batch_size_ = x.shape[0]
                    out = model(x, coords=nodes) #.reshape(batch_size_,  -1)
                    
                    if normalization_y:
                        out = y_normalizer.decode(out)
                        y = y_normalizer.decode(y)
                    out = out * node_mask #mask the padded value with 0,(1 for node, 0 for padding)

                    test_rel_l2 += myloss(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()
                    test_l2 += myloss.abs(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()
                
                test_l2 /= len(loader.dataset)
                test_rel_l2 /= len(loader.dataset)
                test_rel_l2_dict[name] = test_rel_l2
                test_l2_dict[name] = test_l2
    

        

        train_rel_l2/= n_train

        train_rel_l2_losses.append(train_rel_l2)
        test_rel_l2_losses.append(test_rel_l2_dict)
        test_l2_losses.append(test_l2_dict)
    

        t2 = default_timer()
        print("Epoch : ", ep, " Time: ", round(t2-t1,3), " Rel. Train L2 Loss : ", train_rel_l2, " Rel. Test L2 Loss : ", test_rel_l2_dict, " Test L2 Loss : ", test_l2_dict, flush=True)
        if (ep %100 == 99) or (ep == epochs -1):    
            if save_model_name:
                torch.save(model.state_dict(), save_model_name + ".pth")


            
    
    
    return train_rel_l2_losses, test_rel_l2_losses, test_l2_losses