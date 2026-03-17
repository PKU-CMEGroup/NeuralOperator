import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utility.adam import Adam
from utility.losses import LpLoss
from utility.normalizer import UnitGaussianNormalizer
from timeit import default_timer
from pcno.mpcno import compute_Fourier_bases, GeoEmbedding, GradientLayer, SpectralConv, compute_gradient

def _get_act(act):
    if act == "tanh":
        func = F.tanh
    elif act == "gelu":
        func = F.gelu
    elif act == "relu":
        func = F.relu_
    elif act == "elu":
        func = F.elu_
    elif act == "leaky_relu":
        func = F.leaky_relu_
    elif act == "softsign":
        func = F.softsign
    elif act == 'soft_identity':
        func = lambda x: x/(1+0.01*x**2)
    elif act == 'piecewise':
        def piecewise_smooth(x):
            return torch.where(
                x > 10, 
                10 + 5 * torch.tanh((x - 10) / 5), 
                torch.where(x < -10, -10 + 5 * torch.tanh((x + 10) / 5), x)
            )
        func = piecewise_smooth
    elif act == "none":
        func = lambda x: x
    else:
        raise ValueError(f"{act} is not supported")
    return func


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
            feats = [sin(xb), cos(xb)] (and optionally coords itself)
            emb = Linear(feats)

    Args:
        coord_dim: Dimension of the coordinate vector at each token (typically 1, 2, or 3).
        d_coord: Output embedding dimension.
        mode: Either "linear" or "fourier".
        num_frequencies: Number of random Fourier frequencies (only used if mode="fourier").
        scale: Frequency scale for the Fourier matrix B (only used if mode="fourier").
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


class MPCEncoderLayer(nn.Module):
    def __init__(
        self,
        modes,
        ndims,
        d_model=128,
        dim_feedforward=512,
        act='gelu',
        geoact='softsign',
        dropout=0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.act = _get_act(act)
        self.scale_factor = math.sqrt(4)


        self.speconv = SpectralConv(d_model, d_model, modes)
        self.spw = nn.Conv1d(d_model, d_model, 1, bias = False) 
        self.spconvnw = nn.Conv1d(d_model * (ndims + 1), d_model, 1, bias=False)
        self.spconvadjnw = nn.Conv1d(d_model * ndims, d_model, 1, bias = False)
        self.w = nn.Conv1d(d_model, d_model, 1)
        self.grad_layer = GradientLayer(ndims, d_model, d_model, geo_act=geoact)
        self.geo_emb = GeoEmbedding(ndims*(ndims+1), d_model, d_model, geo_act=geoact)


        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0, geo, directed_edges, edge_gradient_weights, outward_normals):

        x = x.permute(0,2,1) # [B, d_model, N]
        x1 = self.speconv( self.spconvnw(  torch.cat([x] + [x * outward_normals[:, i:i+1, :] for i in range(outward_normals.size(1))], dim=1)  ), bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0)
        x1 = self.spw(x1) + self.spconvadjnw(torch.cat([x1 * outward_normals[:, i:i+1, :] for i in range(outward_normals.size(1))], dim=1))

                
        x2 = self.w(x)

        x_grad = self.grad_layer(x, directed_edges, edge_gradient_weights)

        x_geo = self.geo_emb(geo,  x)

        x = x + self.act(self.scale_factor*(x1 + x2 + x_grad + x_geo))
        x = self.norm1(x.permute(0,2,1))

        x = x + self.ffn(x)
        x = self.norm2(x)

        return x
    

class MPCFormer(nn.Module):
    """
    Spectral Transformer (given Fourier bases per layer).

    Pipeline:
      (u, nodes) -> concat with coord embedding -> lift to d_model
                -> repeat: spectral slice/attn/deslice + FFN
                -> readout

    Args:
        ndims: dimension of the problem
        modes: Fourier mode vectors [K, ndims].
        in_channels: Number of input channels per point.
        out_channels: Number of output channels per point.
        coord_dim: Coordinate dimension ndims.
        d_model: Model width.
        nhead: Attention heads (applied in spectral token space).
        num_layers: Number of spectral encoder layers.
        dim_feedforward: FFN hidden size.
        num_frequencies: CoordEmbedding Fourier feature count.
        coord_mode: coordinate embedding mode, "linear" or "fourier"
        d_coord: Coordinate embedding width.
        dropout: Dropout probability.

    Forward Args:
        u:   [B, N, in_channels]
        aux: tuple(node_mask, nodes, node_weights, directed_edges, edge_gradient_weights)
             - nodes:        [B, N, coord_dim]
             - node_weights: [B, N] or [B, N] (quadrature weights; optional but recommended)
             - node_mask:    optional mask [B, N] with 0/1; we zero-out padding points.

    Returns:
        y: [B, N, out_channels]
    """
    def __init__(
        self,
        ndims,
        modes: torch.Tensor,
        in_dim: int = 2,
        out_dim: int = 2,
        d_model: int = 128,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        num_frequencies: int = 16,
        coord_mode = "fourier",
        d_coord: int = 64,
        dropout: float = 0.0,
        act: str = 'gelu',
        geo_act: str = 'softsign',
    ):
        super().__init__()
        self.ndims = ndims
        self.modes = modes
        
        # Coordinate embedding (your existing CoordEmbedding is fine)
        self.coord_emb = CoordEmbedding(
            coord_dim=ndims,
            d_coord=d_coord,
            mode=coord_mode,
            num_frequencies=num_frequencies,
        )

        self.input_proj = nn.Linear(in_dim + d_coord, d_model)

        self.blocks = nn.ModuleList([
            MPCEncoderLayer(        
                modes=modes,
                ndims=ndims,
                d_model=d_model,
                dim_feedforward=dim_feedforward,
                act=act,
                geoact=geo_act,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, out_dim)

    def forward(self, u: torch.Tensor, aux) -> torch.Tensor:
        """
        Args:
            u: [B, N, in_dim]
            aux: tuple(node_mask, nodes, node_weights, directed_edges, edge_gradient_weights)
             - nodes:        [B, N, coord_dim]
             - node_weights: [B, N] or [B, N] (quadrature weights; optional but recommended)
             - node_mask:    optional mask [B, N] with 0/1; we zero-out padding points.


        Returns:
            y: [B, N, out_channels]
        """

        # nodes: float[batch_size, nnodes, ndims]
        node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, outward_normals = aux
        # bases: float[batch_size, nnodes, nmodes]
        # scale the modes k  = k * ( inv_L_scale_min + (inv_L_scale_max - inv_L_scale_min)/(1 + exp(-self.inv_L_scale_latent) ))
        bases_c,  bases_s,  bases_0  = compute_Fourier_bases(nodes, self.modes) 
        # node_weights: float[batch_size, nnodes, nmeasures]
        # wbases: float[batch_size, nnodes, nmodes, nmeasures]
        # set nodes with zero measure to 0
        wbases_c = torch.einsum("bxkw,bxw->bxkw", bases_c, node_weights)
        wbases_s = torch.einsum("bxkw,bxw->bxkw", bases_s, node_weights)
        wbases_0 = torch.einsum("bxkw,bxw->bxkw", bases_0, node_weights)
        
        geo = torch.cat([outward_normals, compute_gradient(outward_normals, directed_edges, edge_gradient_weights)], dim=1)
        
        # nodes: float[batch_size, nnodes, ndims]
        c = self.coord_emb(nodes)          # [B, N, d_coord]
        h = self.input_proj(torch.cat([u, c], dim=-1))  # [B, N, d_model]       

        for blk in self.blocks:
            h = blk(h, bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0, geo, directed_edges, edge_gradient_weights, outward_normals)

        h = self.norm(h)
        return self.output_proj(h)
    




# x_train, y_train, x_test, y_test are [n_data, n_x, n_channel] arrays
def MPCFormer_train_multidist(x_train, aux_train, y_train, x_test_list, aux_test_list, y_test_list,  config, model, label_test_list = None, save_model_name="./MPCNO_model", checkpoint_path=None):
    assert len(x_test_list) == len(y_test_list) == len(aux_test_list), "The length of x_test_list, y_test_list and aux_test_list should be the same"
    n_distributions = len(x_test_list)
    n_train= x_train.shape[0]
    train_rel_l2_losses = []
    test_rel_l2_losses = []
    test_l2_losses = []
    normalization_x, normalization_y = config["train"]["normalization_x"], config["train"]["normalization_y"]
    normalization_dim_x, normalization_dim_y = config["train"]["normalization_dim_x"], config["train"]["normalization_dim_y"]
    non_normalized_dim_x, non_normalized_dim_y = config["train"]["non_normalized_dim_x"], config["train"]["non_normalized_dim_y"]
    
    ndims = model.ndims # n_train, size, n_channel
    print("In MPCFormer_train, ndims = ", ndims)
    
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


    node_mask_train, nodes_train, node_weights_train, directed_edges_train, edge_gradient_weights_train, geo_train = aux_train
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train, node_mask_train, nodes_train, node_weights_train, directed_edges_train, edge_gradient_weights_train, geo_train), 
                                               batch_size=config['train']['batch_size'], shuffle=True)
    
    test_loaders = []

    for i in range(n_distributions):
        node_mask_test, nodes_test, node_weights_test, directed_edges_test, edge_gradient_weights_test, geo_test = aux_test_list[i]
        sub_dataset = torch.utils.data.TensorDataset(
            x_test_list[i], 
            y_test_list[i], 
            node_mask_test, 
            nodes_test, 
            node_weights_test, 
            directed_edges_test, 
            edge_gradient_weights_test,
            geo_test
        )
        sub_loader = torch.utils.data.DataLoader(sub_dataset, batch_size=config['train']['batch_size'], shuffle=False)
        try:
            name = label_test_list[i]
        except:
            name = f"Distribution_{i}"
        test_loaders.append((name, sub_loader))
  
    
    myloss = LpLoss(d=1, p=2, size_average=False)

    optimizer = Adam(model.parameters(),
        betas=(0.9, 0.999),
        lr=config["train"]["base_lr"],
        weight_decay=config["train"]["weight_decay"],
        )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config['train']['base_lr'], 
        div_factor=2, final_div_factor=100,pct_start=0.2,
        steps_per_epoch=len(train_loader), epochs=config['train']['epochs'])
    
    current_epoch, epochs = 0, config['train']['epochs']
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # retrieve epoch and loss
        current_epoch = checkpoint['current_epoch'] + 1
        print("resetart from epoch : ", current_epoch)




    for ep in range(current_epoch, epochs):
        t1 = default_timer()
        train_rel_l2 = 0

        model.train()
        for x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo in train_loader:
            x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo = x.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device), geo.to(device)

            batch_size_ = x.shape[0]
            optimizer.zero_grad()
            out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo)) #.reshape(batch_size_,  -1)
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

                for x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo in loader:
                    x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo = x.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device), geo.to(device)

                    batch_size_ = x.shape[0]
                    out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo)) #.reshape(batch_size_,  -1)

                    if normalization_y:
                        out = y_normalizer.decode(out)
                        y = y_normalizer.decode(y)
                    out = out*node_mask #mask the padded value with 0,(1 for node, 0 for padding)
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
        print("Epoch : ", ep, " Time: ", round(t2-t1,3), " Rel. Train L2 Loss : ", train_rel_l2, " Rel. Test L2 Loss : ", test_rel_l2_dict, " Test L2 Loss : ", test_l2_dict,
              flush=True)
        if (ep %100 == 99) or (ep == epochs -1):    
            if save_model_name:
                torch.save(model.state_dict(), save_model_name + ".pth")

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'current_epoch': ep,  # optional: to track training progress
                }, "checkpoint.pth")

            
    
    
    return train_rel_l2_losses, test_rel_l2_losses, test_l2_losses
