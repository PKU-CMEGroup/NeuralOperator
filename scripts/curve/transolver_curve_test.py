import argparse
import os
import sys
import numpy as np
import torch
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
# allow importing from repo root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from baselines.transolver_plus import Model


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class CurveDataset(Dataset):
    def __init__(self, npz_path: str, indices: np.ndarray):
        self.npz_path = npz_path
        self.data = np.load(npz_path, mmap_mode="r")
        self.nodes = self.data["nodes"]
        self.features = self.data["features"]
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        nodes = self.nodes[i].astype(np.float32)
        features = self.features[i].astype(np.float32)
        f_in = features[..., 0:1]
        f_out = features[..., -1:]
        x = np.concatenate([nodes, f_in], axis=-1)
        return torch.from_numpy(x), torch.from_numpy(f_out), torch.from_numpy(nodes)


def rel_l2_loss(pred, target, eps=1e-12):
    return torch.norm(pred - target) / (torch.norm(target) + eps)


def run_epoch(model, loader, device, optimizer=None, scheduler=None):
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    losses = []
    for x, y, pos in loader:
        x = x.to(device)
        y = y.to(device)
        pos = pos.to(device)

        if is_train:
            optimizer.zero_grad()
            pred = model((x, pos, None))
            loss = rel_l2_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        else:
            with torch.no_grad():
                pred = model((x, pos, None))
                loss = rel_l2_loss(pred, y)

        losses.append(loss.item())

    return float(np.mean(losses)) if losses else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel_type', type=str, default='sp_laplace', choices=['sp_laplace', 'dp_laplace', 'adjoint_dp_laplace', 'stokes', 'modified_dp_laplace', 'exterior_laplace_neumann'])
    parser.add_argument("--train_n", type=int, default=2000)
    parser.add_argument("--test_n", type=int, default=1000)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "../../data/curve/"
    data_file_path = data_path+f"pcno_curve_data_1_1_5_2d_{args.kernel_type}_panel"
    curve1_data = np.load(data_file_path + ".npz", mmap_mode="r")
    n1 = curve1_data["nodes"].shape[0]
    curve2_data = np.load(data_file_path + "_two_circles.npz", mmap_mode="r")
    n2 = curve2_data["nodes"].shape[0]


    train_indices = np.arange(0, args.train_n)
    test1_indices = np.arange(n1 - args.test_n, n1)
    test2_indices = np.arange(n2 - args.test_n, n2)

    train_ds = CurveDataset(data_file_path + ".npz", train_indices)
    test1_ds = CurveDataset(data_file_path + ".npz", test1_indices)
    test2_ds = CurveDataset(data_file_path + "_two_circles.npz", test2_indices)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test1_loader = DataLoader(test1_ds, batch_size=args.batch_size, shuffle=False)
    test2_loader = DataLoader(test2_ds, batch_size=args.batch_size, shuffle=False)

    model = Model(
        space_dim=2,
        fun_dim=1,
        out_dim=1,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        n_head=8,
        dropout=0.1,
        mlp_ratio=2,
        slice_num=32,
        ref=8,
        unified_pos=False,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        final_div_factor=1000.0,
    )

    for epoch in range(args.epochs):
        start_t = time.time()
        train_loss = run_epoch(model, train_loader, device, optimizer=optimizer, scheduler=scheduler)
        test1_loss = run_epoch(model, test1_loader, device, optimizer=None)
        test2_loss = run_epoch(model, test2_loader, device, optimizer=None)
        epoch_time = time.time() - start_t
        print(f"Epoch {epoch} train loss: {train_loss}, test loss curve1: {test1_loss}, test loss curve2: {test2_loss}, time: {epoch_time:.2f}s",flush=True)


if __name__ == "__main__":
    main()
