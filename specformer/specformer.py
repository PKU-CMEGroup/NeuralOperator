import math
import numpy as np
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



def compute_Fourier_modes_helper(ndims, nks, Ls):
    '''
    Compute Fourier modes number k
    Fourier bases are cos(kx), sin(kx), 1
    * We cannot have both k and -k, cannot have 0

        Parameters:  
            ndims : int
            nks   : int[ndims]
            Ls    : float[ndims]

        Return :
            k_pairs : float[nmodes, ndims]
    '''
    assert(len(nks) == len(Ls) == ndims)    
    if ndims == 1:
        nk, Lx = nks[0], Ls[0]
        k_pairs    = np.zeros((nk, ndims))
        k_pair_mag = np.zeros(nk)
        i = 0
        for kx in range(1, nk + 1):
            k_pairs[i, :] = 2*np.pi/Lx*kx
            k_pair_mag[i] = np.linalg.norm(k_pairs[i, :])
            i += 1

    elif ndims == 2:
        nx, ny = nks
        Lx, Ly = Ls
        nk = 2*nx*ny + nx + ny
        k_pairs    = np.zeros((nk, ndims))
        k_pair_mag = np.zeros(nk)
        i = 0
        for kx in range(-nx, nx + 1):
            for ky in range(0, ny + 1):
                if (ky==0 and kx<=0): 
                    continue

                k_pairs[i, :] = 2*np.pi/Lx*kx, 2*np.pi/Ly*ky
                k_pair_mag[i] = np.linalg.norm(k_pairs[i, :])
                i += 1

    elif ndims == 3:
        nx, ny, nz = nks
        Lx, Ly, Lz = Ls
        nk = 4*nx*ny*nz + 2*(nx*ny + nx*nz + ny*nz) + nx + ny + nz
        k_pairs    = np.zeros((nk, ndims))
        k_pair_mag = np.zeros(nk)
        i = 0
        for kx in range(-nx, nx + 1):
            for ky in range(-ny, ny + 1):
                for kz in range(0, nz + 1):
                    if (kz==0 and (ky<0  or (ky==0 and kx<=0))): 
                        continue

                    k_pairs[i, :] = 2*np.pi/Lx*kx, 2*np.pi/Ly*ky, 2*np.pi/Lz*kz
                    k_pair_mag[i] = np.linalg.norm(k_pairs[i, :])
                    i += 1
    else:
        raise ValueError(f"{ndims} in compute_Fourier_modes is not supported")
    
    k_pairs = k_pairs[np.argsort(k_pair_mag, kind='stable'), :]
    return k_pairs


def compute_Fourier_modes(ndims, nks, Ls):
    '''
    Compute `nmeasures` sets of Fourier modes number k
    Fourier bases are cos(kx), sin(kx), 1
    * We cannot have both k and -k

        Parameters:  
            ndims : int
            nks   : int[ndims]
            Ls    : float[ndims]

        Return :
            k_pairs : float[nmodes, ndims]
    '''
    assert(len(nks) == len(Ls))
    
    k_pairs = compute_Fourier_modes_helper(ndims, nks, Ls)
    
    return k_pairs


def compute_Fourier_bases(nodes, modes):
    """
    Compute real Fourier bases on an (irregular) point cloud.

    Args:
        nodes: Point coordinates, shape [B, N, ndims].
        modes: Frequency vectors, shape [K, ndims].
               (You are responsible for scaling modes appropriately for your domain.)
        
               
    Returns:
        bases: Real Fourier feature matrix Phi(nodes), shape [B, N, M],
               where M = 2K +1, ordered as:
                 [cos(<x,k_1>),...,cos(<x,k_K>), sin(<x,k_1>),...,sin(<x,k_K>), 1]


    Require:
        nodes.shape[-1] == modes.shape[-1]
    """
    # temp : float[batch_size, nnodes, nmodes]
    temp  = torch.einsum("bxd,kd->bxk", nodes, modes) # [B,N,K]
    bases_c = torch.cos(temp) 
    bases_s = torch.sin(temp) 
    batch_size, nnodes, _ = temp.shape
    bases_0 = torch.ones(batch_size, nnodes, 1, dtype=temp.dtype, device=temp.device)
    return torch.cat([ bases_c, bases_s, bases_0], dim=-1)


class SpectralEncoderLayer(nn.Module):
    """
    Spectral attention block (project -> attention -> project back) + FFN.

    This layer uses *given bases* Phi to compress N point tokens into M spectral tokens,
    performs MHSA in the spectral token space, then broadcasts back to points.

    Args:
        d_model: Token embedding dimension.
        nhead: Number of attention heads (applied in spectral token space).
        dim_feedforward: FFN hidden size (applied point-wise on point tokens).
        dropout: Dropout probability.
        norm_first: If True, use pre-norm (recommended).

    Forward Args:
        x:      [B, N, d_model]   point tokens
        bases:  [B, N, M]         Phi
        wbases: [B, N, M]         Phi dx
                weighted bases, weighed by the node weight dx
                
    Returns:
        x_next: [B, N, d_model]

    Math:
        Z = Phi dx ⊙ x                (slice, points -> spectral)
        Z <- Z + MHSA(LN(Z))          (spectral attention)
        x <- x + Phi Z                (deslice, spectral -> points)
        x <- x + FFN(LN(x))           (pointwise feedforward)
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        norm_first: bool = True,
    ):
        super().__init__()
        self.norm_first = norm_first


        # Attention is computed on latent tokens Z in R^{B x M x d_model}
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        # FFN (token-wise MLP on point tokens)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    
        
    def forward(self, x: torch.Tensor, bases: torch.Tensor, wbases: torch.Tensor):
        # ---- Slice: points -> spectral tokens ----
        # Z = Phi dx ⊙ x: [B,N,M] @ [B,N,d_model] = [B,M,d_model]
        Z =  torch.einsum("bxd,bxt->btd", x, wbases)
        
        if self.norm_first:
            Z_in = self.norm1(Z)
            Z = Z + self.attn(Z_in, Z_in, Z_in, need_weights=False)[0]
        else:
            Z =  self.norm1(Z + self.attn(Z_in, Z_in, Z_in, need_weights=False)[0])
                
        # ---- Deslice: spectral tokens -> points ----
        # dx = Z Phi: [B,M,d_model] @ [B,N,M] = [B,N,d_model]
        dx = torch.einsum("btd,bxt->bxd", Z, bases)
        x = x + dx

        # ---- Pointwise FFN ----
        if self.norm_first:
            x = x + self.ffn(self.norm2(x))
        else:
            x = self.norm2(x + self.ffn(x))

        return x
    

class Specformer(nn.Module):
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
        in_channels: int = 2,
        out_channels: int = 2,
        coord_dim: int = 1,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        num_frequencies: int = 16,
        coord_mode = "fourier",
        d_coord: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ndims = ndims
        self.coord_dim = coord_dim
        self.modes = modes
        
        # Coordinate embedding (your existing CoordEmbedding is fine)
        self.coord_emb = CoordEmbedding(
            coord_dim=coord_dim,
            d_coord=d_coord,
            mode=coord_mode,
            num_frequencies=num_frequencies,
        )

        self.input_proj = nn.Linear(in_channels + d_coord, d_model)

        self.blocks = nn.ModuleList([
            SpectralEncoderLayer(        
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, out_channels)

    def forward(self, u: torch.Tensor, aux) -> torch.Tensor:
        """
        Args:
            u: [B, N, in_channels]
            aux: tuple(node_mask, nodes, node_weights, directed_edges, edge_gradient_weights)
             - nodes:        [B, N, coord_dim]
             - node_weights: [B, N] or [B, N] (quadrature weights; optional but recommended)
             - node_mask:    optional mask [B, N] with 0/1; we zero-out padding points.


        Returns:
            y: [B, N, out_channels]
        """
        node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = aux
        bases = compute_Fourier_bases(nodes, self.modes) # B, N, M
        
        
        # nodes: float[batch_size, nnodes, ndims]
        c = self.coord_emb(nodes)          # [B, N, d_coord]
        h = self.input_proj(torch.cat([u, c], dim=-1))  # [B, N, d_model]
        
        wbases = torch.einsum("bxk,bx->bxk", bases, node_weights)
        
        

        for blk in self.blocks:
            h = blk(h, bases, wbases)

        h = self.norm(h)
        return self.output_proj(h)
    




# x_train, y_train, x_test, y_test are [n_data, n_x, n_channel] arrays
def Specformer_train(x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, save_model_name="./Transformer_model"):
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

    node_mask_train, nodes_train, node_weights_train, directed_edges_train, edge_gradient_weights_train = aux_train
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train, node_mask_train, nodes_train, node_weights_train, directed_edges_train, edge_gradient_weights_train), 
                                               batch_size=config['train']['batch_size'], shuffle=True)
    
    node_mask_test, nodes_test, node_weights_test, directed_edges_test, edge_gradient_weights_test = aux_test
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test, node_mask_test, nodes_test, node_weights_test, directed_edges_test, edge_gradient_weights_test), 
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
        for x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights in train_loader:
            x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = x.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device)

            batch_size_ = x.shape[0]
            optimizer.zero_grad()
            out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights)) #.reshape(batch_size_,  -1)
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
            for x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights in test_loader:
                x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = x.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device)

                batch_size_ = x.shape[0]
                out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights)) #.reshape(batch_size_,  -1)

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