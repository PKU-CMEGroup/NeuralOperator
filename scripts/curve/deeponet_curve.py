import os
import math
import argparse
from timeit import default_timer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 基础组件
# ============================================================

def get_activation(name: str):
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "tanh":
        return nn.Tanh()
    if name == "elu":
        return nn.ELU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    if name == "softsign":
        return nn.Softsign()
    if name == 'sin':
        return lambda x: torch.sin(x)
    raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, act="gelu"):
        super().__init__()
        dims = [in_dim] + list(hidden_dims) + [out_dim]

        self.linears1 = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))
        self.linears2 = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))
        self.act = get_activation(act)

    def forward(self, x):
        for i,layer in enumerate(self.linears1):
            x_linear = self.linears2[i](x)
            x = layer(x)
            if i < len(self.linears1) - 1:
                x = self.act(x) + x_linear
        return x


class TensorGaussianNormalizer:
    """
    按最后一个通道维做高斯标准化。
    支持保留最后 exclude_last_n 个通道不做标准化。
    对当前数据脚本来说，x 默认不标准化，y 默认标准化。
    """
    def __init__(self, x: torch.Tensor, exclude_last_n: int = 0, eps: float = 1e-5):
        super().__init__()
        self.exclude_last_n = int(exclude_last_n)
        self.eps = eps
        c = x.shape[-1]
        self.norm_c = c - self.exclude_last_n
        if self.norm_c > 0:
            flat = x[..., :self.norm_c].reshape(-1, self.norm_c)
            self.mean = flat.mean(dim=0, keepdim=True)
            self.std = flat.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
        else:
            self.mean = None
            self.std = None

    def to(self, device):
        if self.mean is not None:
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_c <= 0:
            return x
        x1 = (x[..., :self.norm_c] - self.mean) / self.std
        if self.exclude_last_n > 0:
            return torch.cat([x1, x[..., self.norm_c:]], dim=-1)
        return x1

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_c <= 0:
            return x
        x1 = x[..., :self.norm_c] * self.std + self.mean
        if self.exclude_last_n > 0:
            return torch.cat([x1, x[..., self.norm_c:]], dim=-1)
        return x1


# ============================================================
# 损失函数
# ============================================================

def _expand_mask_like(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if mask.shape[-1] == 1 and target.shape[-1] > 1:
        return mask.expand(-1, -1, target.shape[-1]).float()
    return mask.float()


def relative_l2_sum(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    mask = _expand_mask_like(mask, target)
    diff = (pred - target) * mask
    tgt = target * mask
    num = torch.linalg.norm(diff.reshape(diff.shape[0], -1), dim=1)
    den = torch.linalg.norm(tgt.reshape(tgt.shape[0], -1), dim=1).clamp_min(eps)
    return (num / den).sum()



def absolute_l2_sum(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = _expand_mask_like(mask, target)
    diff = (pred - target) * mask
    return torch.linalg.norm(diff.reshape(diff.shape[0], -1), dim=1).sum()


# ============================================================
# DeepONet
# ============================================================

def masked_max(x: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """
    x: [B, N, C]
    node_mask: [B, N, 1]
    return: [B, C]
    """
    mask = node_mask.bool().expand_as(x)
    x_masked = x.masked_fill(~mask, float("-inf"))
    out = x_masked.max(dim=1).values
    out[out == float("-inf")] = 0.0
    return out


class CurveDeepONet(nn.Module):
    """
    面向当前 curve 数据集的 DeepONet：

    - Branch: 读入整条曲线上的离散输入函数 + 几何信息（就是 x 的全部节点）
      先做逐点编码，再做加权池化，得到全局 operator code。
    - Trunk: 对每个查询点使用 (x, y, nx, ny) 生成 basis。
    - 输出: 对每个节点做 branch/trunk 内积，得到该节点输出。

    这是真正的 branch/trunk 结构，不再是 MPCNO/FNO 风格的频域卷积。
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        trunk_dim: int = 2,
        point_encoder_dims=(128, 128),
        branch_dims=(128, 128),
        trunk_dims=(128, 128),
        latent_dim: int = 128,
        act: str = "gelu",
        dropout: float = 0.0,
        use_node_weights: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.trunk_dim = trunk_dim
        self.latent_dim = latent_dim
        self.use_node_weights = use_node_weights

        point_width = point_encoder_dims[-1]
        self.branch_point_encoder = MLP(in_dim, point_encoder_dims[:-1], point_width, act=act)
        self.branch_global = nn.Linear(point_width, out_dim * latent_dim)

        self.input_linear = nn.Linear(in_dim, point_width)

        self.trunk = MLP(trunk_dim, trunk_dims, out_dim * latent_dim, act=act)
        self.trunk_bias = MLP(trunk_dim, trunk_dims, out_dim, act=act)

        self.global_bias = nn.Parameter(torch.zeros(1, 1, out_dim))

    def forward(self, x: torch.Tensor, aux):
        """
        x:   [B, N, in_dim]
        aux: (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, outward_normals)
        """
        node_mask, nodes, node_weights, _, _, _ = aux
        # -------- branch --------
        point_feat = self.branch_point_encoder(x) * self.input_linear(x) * node_mask             # [B, N, Cp]
        # point_feat = self.branch_point_encoder(x) * node_mask             # [B, N, Cp]

        if self.use_node_weights and node_weights is not None:
            # 当前 curve 数据集只有一个 measure，直接取 [:, :, :1]
            pool_w = node_weights[..., :1] * node_mask
        else:
            pool_w = node_mask

        pooled_mean = (point_feat * pool_w).sum(dim=1)                     # [B, Cp]
        branch_code = self.branch_global(pooled_mean)                        # [B, out_dim * p]

        # -------- trunk --------
        B, N, _ = nodes.shape
        trunk_code = self.trunk(nodes)                               # [B, N, out_dim * p]
        trunk_bias = self.trunk_bias(nodes)                          # [B, N, out_dim]

        branch_code = branch_code.view(B, self.out_dim, self.latent_dim)
        trunk_code = trunk_code.view(B, N, self.out_dim, self.latent_dim)

        out = torch.einsum("bop,bnop->bno", branch_code, trunk_code)
        out = out / math.sqrt(self.latent_dim)
        out = out + trunk_bias + self.global_bias
        out = out * node_mask
        return out


# ============================================================
# 数据加载（沿用 mpcno_curve_test.py 的组织方式）
# ============================================================

def load_data_to_torch(data_file_path, to_divide=None, factor=1.0):
    data = np.load(data_file_path)
    nnodes = data["nnodes"]
    node_mask = data["node_mask"]
    nodes = data["nodes"]
    print(f"Loaded {nodes.shape[0]} samples from {data_file_path}", flush=True)

    node_weights = data["node_measures_raw"]
    if to_divide is None:
        to_divide = factor * np.amax(np.sum(node_weights, axis=1))
    node_weights = node_weights / to_divide

    node_measures = data["node_measures"]
    directed_edges = data["directed_edges"]
    edge_gradient_weights = data["edge_gradient_weights"]
    features = data["features"]

    node_measures_raw = data["node_measures_raw"]
    indices = np.isfinite(node_measures_raw)
    node_rhos = np.copy(node_weights)
    node_rhos[indices] = node_rhos[indices] / node_measures[indices]

    nnodes = torch.from_numpy(nnodes)
    node_mask = torch.from_numpy(node_mask)
    nodes = torch.from_numpy(nodes.astype(np.float32))
    node_weights = torch.from_numpy(node_weights.astype(np.float32))
    node_rhos = torch.from_numpy(node_rhos.astype(np.float32))
    features = torch.from_numpy(features.astype(np.float32))
    directed_edges = torch.from_numpy(directed_edges.astype(np.int64))
    edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))

    return nnodes, node_mask, nodes, node_weights, node_rhos, features, directed_edges, edge_gradient_weights, to_divide



def gen_data_tensors(data_indices, nodes, features, node_mask, node_weights, directed_edges, edge_gradient_weights, f_in_dim, f_out_dim):
    """
    保持和 mpcno_curve_test.py 同样的数据切法：
    x = [输入函数, nx, ny, x, y]
    y = [目标输出]
    aux = (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, outward_normals)
    """
    idx = torch.as_tensor(data_indices, dtype=torch.long)
    nodes_input = nodes.clone()

    x = torch.cat(
        [features[idx][..., : f_in_dim + 2], nodes_input[idx, ...]],
        dim=-1,
    )
    y = features[idx][..., -f_out_dim:]
    nx = features[idx][..., f_in_dim : f_in_dim + 2]
    aux = (
        node_mask[idx],
        nodes[idx],
        node_weights[idx],
        directed_edges[idx],
        edge_gradient_weights[idx],
        nx.permute(0, 2, 1).contiguous(),
    )
    return x, y, aux


# ============================================================
# 训练与测试
# ============================================================

def build_loader(x, y, aux, batch_size, shuffle):
    node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, outward_normals = aux
    dataset = TensorDataset(
        x,
        y,
        node_mask,
        nodes,
        node_weights,
        directed_edges,
        edge_gradient_weights,
        outward_normals,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate_loader(model, loader, device, y_normalizer=None):
    model.eval()
    total_rel = 0.0
    total_abs = 0.0
    total_n = 0

    for batch in loader:
        x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, outward_normals = batch
        x = x.to(device)
        y = y.to(device)
        node_mask = node_mask.to(device)
        nodes = nodes.to(device)
        node_weights = node_weights.to(device)
        directed_edges = directed_edges.to(device)
        edge_gradient_weights = edge_gradient_weights.to(device)
        outward_normals = outward_normals.to(device)

        pred = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, outward_normals))
        if y_normalizer is not None:
            pred = y_normalizer.decode(pred)
            y = y_normalizer.decode(y)

        batch_rel = relative_l2_sum(pred, y, node_mask)
        batch_abs = absolute_l2_sum(pred, y, node_mask)

        bs = x.shape[0]
        total_rel += batch_rel.item()
        total_abs += batch_abs.item()
        total_n += bs

    return total_rel / total_n, total_abs / total_n



def train_deeponet_multidist(
    x_train,
    aux_train,
    y_train,
    x_test_list,
    aux_test_list,
    y_test_list,
    config,
    model,
    label_test_list=None,
    save_model_name=None,
):
    assert len(x_test_list) == len(aux_test_list) == len(y_test_list)

    device = next(model.parameters()).device
    n_train = x_train.shape[0]
    train_cfg = config["train"]

    # ---------------- normalization ----------------
    x_normalizer = None
    if train_cfg.get("normalization_x", False):
        x_normalizer = TensorGaussianNormalizer(
            x_train,
            exclude_last_n=train_cfg.get("x_exclude_last_n", 0),
        )
        x_train = x_normalizer.encode(x_train)
        x_test_list = [x_normalizer.encode(xi) for xi in x_test_list]
        x_normalizer.to(device)

    y_normalizer = None
    if train_cfg.get("normalization_y", True):
        y_normalizer = TensorGaussianNormalizer(
            y_train,
            exclude_last_n=train_cfg.get("y_exclude_last_n", 0),
        )
        y_train = y_normalizer.encode(y_train)
        y_test_list = [y_normalizer.encode(yi) for yi in y_test_list]
        y_normalizer.to(device)

    # ---------------- loaders ----------------
    train_loader = build_loader(
        x_train,
        y_train,
        aux_train,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
    )

    test_loaders = []
    for i in range(len(x_test_list)):
        name = label_test_list[i] if label_test_list is not None else f"Distribution_{i}"
        loader = build_loader(
            x_test_list[i],
            y_test_list[i],
            aux_test_list[i],
            batch_size=train_cfg["batch_size"],
            shuffle=False,
        )
        test_loaders.append((name, loader))

    # ---------------- optimizer ----------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["base_lr"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )

    scheduler_name = train_cfg.get("scheduler", "OneCycleLR")
    if scheduler_name == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=train_cfg["base_lr"],
            div_factor=2.0,
            final_div_factor=100.0,
            pct_start=0.2,
            steps_per_epoch=len(train_loader),
            epochs=train_cfg["epochs"],
        )
        step_per_batch = True
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_cfg["epochs"],
        )
        step_per_batch = False

    train_rel_history = []
    test_rel_history = []
    test_abs_history = []
    best_default_rel = float("inf")

    for ep in range(train_cfg["epochs"]):
        t1 = default_timer()
        model.train()
        train_rel_sum = 0.0

        for batch in train_loader:
            x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, outward_normals = batch
            x = x.to(device)
            y = y.to(device)
            node_mask = node_mask.to(device)
            nodes = nodes.to(device)
            node_weights = node_weights.to(device)
            directed_edges = directed_edges.to(device)
            edge_gradient_weights = edge_gradient_weights.to(device)
            outward_normals = outward_normals.to(device)

            optimizer.zero_grad()
            pred = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, outward_normals))
            if y_normalizer is not None:
                pred = y_normalizer.decode(pred)
                y_phys = y_normalizer.decode(y)
            else:
                y_phys = y

            loss = relative_l2_sum(pred, y_phys, node_mask)
            loss.backward()
            optimizer.step()
            if step_per_batch:
                scheduler.step()

            train_rel_sum += loss.item()

        if not step_per_batch:
            scheduler.step()

        train_rel = train_rel_sum / n_train
        train_rel_history.append(train_rel)

        rel_dict = {}
        abs_dict = {}
        for name, loader in test_loaders:
            rel_l2, abs_l2 = evaluate_loader(model, loader, device, y_normalizer=y_normalizer)
            rel_dict[name] = rel_l2
            abs_dict[name] = abs_l2

        test_rel_history.append(rel_dict)
        test_abs_history.append(abs_dict)

        default_key = list(rel_dict.keys())[0]
        if rel_dict[default_key] < best_default_rel:
            best_default_rel = rel_dict[default_key]
            if save_model_name is not None:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "best_default_rel": best_default_rel,
                    },
                    save_model_name,
                )

        t2 = default_timer()
        print(
            f"Epoch: {ep:04d} | Time: {t2 - t1:.3f}s | "
            f"Train RelL2: {train_rel:.6e} | "
            f"Test RelL2: {rel_dict} | Test AbsL2: {abs_dict}",
            flush=True,
        )

    return train_rel_history, test_rel_history, test_abs_history


# ============================================================
# 主脚本
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="DeepONet for the same curve dataset used by mpcno_curve_test.py")

    parser.add_argument("--data_path", type=str, default="../../data/curve/")
    parser.add_argument(
        "--kernel_type",
        type=str,
        default="sp_laplace",
        choices=[
            "sp_laplace",
            "dp_laplace",
            "adjoint_dp_laplace",
            "stokes",
            "modified_dp_laplace",
            "exterior_laplace_neumann",
            "weighted_sp_laplace",
            "weighted_dp_laplace",
        ],
    )
    parser.add_argument("--to_divide_factor", type=float, default=20.0)
    parser.add_argument("--n_train", type=int, default=2000)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--n_two_circles_test", type=int, default=0)
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--ep", type=int, default=500)
    parser.add_argument("--base_lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--act", type=str, default="gelu")
    parser.add_argument("--dropout", type=float, default=0.0)

    # DeepONet 结构参数
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--point_encoder_dims", type=str, default="128,128")
    parser.add_argument("--branch_dims", type=str, default="128,128")
    parser.add_argument("--trunk_dims", type=str, default="128,128")

    parser.add_argument("--save_model", type=str, default=None)

    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    f_in_dim = 2 if args.kernel_type in ["stokes"] else 1
    f_out_dim = 2 if args.kernel_type in ["modified_dp_laplace", "stokes"] else 1

    point_encoder_dims = [int(v) for v in args.point_encoder_dims.split(",") if v.strip()]
    branch_dims = [int(v) for v in args.branch_dims.split(",") if v.strip()]
    trunk_dims = [int(v) for v in args.trunk_dims.split(",") if v.strip()]

    # ---------------- load default dataset ----------------
    data_file_path = os.path.join(
        args.data_path,
        f"pcno_curve_data_1_1_5_2d_{args.kernel_type}_panel.npz",
    )

    nnodes, node_mask, nodes, node_weights, node_rhos, features, directed_edges, edge_gradient_weights, to_divide = load_data_to_torch(
        data_file_path,
        to_divide=None,
        factor=args.to_divide_factor,
    )

    x_train, y_train, aux_train = gen_data_tensors(
        np.arange(args.n_train),
        nodes,
        features,
        node_mask,
        node_weights,
        directed_edges,
        edge_gradient_weights,
        f_in_dim=f_in_dim,
        f_out_dim=f_out_dim,
    )
    x_test, y_test, aux_test = gen_data_tensors(
        np.arange(-args.n_test, 0),
        nodes,
        features,
        node_mask,
        node_weights,
        directed_edges,
        edge_gradient_weights,
        f_in_dim=f_in_dim,
        f_out_dim=f_out_dim,
    )

    x_test_list = [x_test]
    y_test_list = [y_test]
    aux_test_list = [aux_test]
    label_list = ["Default"]

    # ---------------- optional two-circles test set ----------------
    if args.n_two_circles_test > 0:
        data_file_path_2 = os.path.join(
            args.data_path,
            f"pcno_curve_data_1_1_5_2d_{args.kernel_type}_panel_two_circles.npz",
        )
        nnodes2, node_mask2, nodes2, node_weights2, node_rhos2, features2, directed_edges2, edge_gradient_weights2, _ = load_data_to_torch(
            data_file_path_2,
            to_divide=to_divide,
        )
        x_test2, y_test2, aux_test2 = gen_data_tensors(
            np.arange(args.n_two_circles_test),
            nodes2,
            features2,
            node_mask2,
            node_weights2,
            directed_edges2,
            edge_gradient_weights2,
            f_in_dim=f_in_dim,
            f_out_dim=f_out_dim,
        )
        x_test_list.append(x_test2)
        y_test_list.append(y_test2)
        aux_test_list.append(aux_test2)
        label_list.append("Two Circles")

    print(
        f"x_train shape {tuple(x_train.shape)}, x_test shape {[tuple(x.shape) for x in x_test_list]}, "
        f"y_train shape {tuple(y_train.shape)}, y_test shape {[tuple(y.shape) for y in y_test_list]}",
        flush=True,
    )
    print(
        "Domain range per dimension:",
        torch.amax(nodes, dim=[0, 1]) - torch.amin(nodes, dim=[0, 1]),
        flush=True,
    )

    # ---------------- model ----------------
    model = CurveDeepONet(
        in_dim=x_train.shape[-1],
        out_dim=y_train.shape[-1],
        trunk_dim=2,         
        point_encoder_dims=point_encoder_dims,
        branch_dims=branch_dims,
        trunk_dims=trunk_dims,
        latent_dim=args.latent_dim,
        act=args.act,
        dropout=args.dropout,
        use_node_weights=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print("------ DeepONet Parameters ------")
    print(f"kernel_type = {args.kernel_type}")
    print(f"n_train = {args.n_train}, n_test = {args.n_test}")
    print(f"latent_dim = {args.latent_dim}")
    print(f"point_encoder_dims = {point_encoder_dims}")
    print(f"branch_dims = {branch_dims}")
    print(f"trunk_dims = {trunk_dims}")
    print(f"activation = {args.act}")
    print(f"#params = {total_params}")
    print(f"batch_size = {args.bsz}")

    config = {
        "train": {
            "base_lr": args.base_lr,
            "weight_decay": args.weight_decay,
            "epochs": args.ep,
            "scheduler": "OneCycleLR",
            "batch_size": args.bsz,
            "normalization_x": False,
            "normalization_y": True,
            "x_exclude_last_n": 4,   # 若将来打开 x 标准化，默认不标准化 [nx, ny, x, y]
            "y_exclude_last_n": 0,
        }
    }

    train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = train_deeponet_multidist(
        x_train,
        aux_train,
        y_train,
        x_test_list,
        aux_test_list,
        y_test_list,
        config,
        model,
        label_test_list=label_list,
        save_model_name=args.save_model,
    )

    print("\nTraining finished.")
    print(f"Last train RelL2: {train_rel_l2_losses[-1]:.6e}")
    print(f"Last test RelL2:  {test_rel_l2_losses[-1]}")
    print(f"Last test AbsL2:  {test_l2_losses[-1]}")


if __name__ == "__main__":
    main()
