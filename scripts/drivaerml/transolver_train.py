import os
import torch
import sys
import argparse

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from timeit import default_timer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utility.adam import Adam  # noqa: E402
from utility.losses import LpLoss  # noqa: E402
from utility.normalizer import UnitGaussianNormalizer  # noqa: E402
torch.set_printoptions(precision=16)

from baselines.transolver_plus import Model  # noqa: E402


torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###################################
# load parameters
###################################

parser = argparse.ArgumentParser(description="Train Transolver++ on preprocessed HiFi3D point data.")
parser.add_argument("--n_train", type=int, default=16)
parser.add_argument("--n_test", type=int, default=4)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--k_max", type=int, default=4)
parser.add_argument("--layer_sizes", type=str, default="64,64,64,64")
parser.add_argument("--fc_dim", type=int, default=128)
parser.add_argument("--lr", type=float, default=5.0e-4)
parser.add_argument("--weight_decay", type=float, default=1.0e-4)
parser.add_argument('--to_divide_factor', type=float, default=20.0)
parser.add_argument("--Ls", type=str, default="")
parser.add_argument("--save_model_name", type=str, default="")
parser.add_argument("--normalization_y", type=str, default="True", choices=["True", "False"])
parser.add_argument("--scheduler_step", type=str, default="batch", choices=["epoch", "batch"])
parser.add_argument("--max_nodes", type=int, default=0)
parser.add_argument("--transolver_nhead", type=int, default=8)
parser.add_argument("--transolver_slice_num", type=int, default=32)
parser.add_argument("--transolver_dropout", type=float, default=0.0)
parser.add_argument("--transolver_mlp_ratio", type=int, default=2)
parser.add_argument("--transolver_ref", type=int, default=8)
args = parser.parse_args()

def load_data_to_torch(data_file_path, to_divide = None, factor = 1.0):
    data = np.load(data_file_path)
    
    node_weights = data["node_measures"]
    if to_divide is None:
        to_divide = factor * np.amax(np.sum(node_weights, axis = 1))
    node_weights = node_weights/to_divide

    return data, node_weights

def gen_data_tensors(
    data: np.lib.npyio.NpzFile,
    indices: np.ndarray,
    node_weights: np.ndarray,
    max_nodes: int,
):
    node_slice = slice(None if max_nodes <= 0 else max_nodes)
    node_mask = torch.from_numpy(data["node_mask"][indices, node_slice].astype(np.float32))
    nodes = torch.from_numpy(data["nodes"][indices, node_slice].astype(np.float32))
    node_weights = torch.from_numpy(node_weights[indices, node_slice].astype(np.float32))
    features = torch.from_numpy(data["features"][indices, node_slice].astype(np.float32))
    normals = features[..., :3]
    y = features[..., -1:] * node_mask
    x = torch.cat([nodes, normals], dim=-1)
    x = x * node_mask
    condition = torch.empty((len(indices), 0), dtype=torch.float32)
    aux = (node_mask, nodes, node_weights, condition)
    return x, y, aux

def Transolverpp_train(
    x_train: torch.Tensor,
    aux_train: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    y_train: torch.Tensor,
    test_sets: list[tuple[str | None, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]],
    config: dict[str, object],
    model: torch.nn.Module,
    save_model_name: str | None = None,
):
    train_rel_l2_losses = []
    test_rel_l2_losses = []
    test_l2_losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config["normalization_y"]:
        y_normalizer = UnitGaussianNormalizer(y_train, non_normalized_dim=0, normalization_dim=[])
        y_train = y_normalizer.encode(y_train)
        test_sets = [(label, x_test, aux_test, y_normalizer.encode(y_test)) for label, x_test, aux_test, y_test in test_sets]
        y_normalizer.to(device)
    else:
        y_normalizer = None

    train_loader = _make_loader(x_train, y_train, aux_train, int(config["batch_size"]), shuffle=True)
    test_loaders = [(label, _make_loader(x_test, y_test, aux_test, int(config["batch_size"]), shuffle=False)) for label, x_test, aux_test, y_test in test_sets]

    optimizer = Adam(model.parameters(), betas=(0.9, 0.999), lr=float(config["base_lr"]), weight_decay=float(config["weight_decay"]))
    steps_per_epoch = len(train_loader) if str(config["scheduler_step"]) == "batch" else 1
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(config["base_lr"]),
        div_factor=2,
        final_div_factor=100,
        pct_start=0.2,
        steps_per_epoch=steps_per_epoch,
        epochs=int(config["epochs"]),
    )
    loss_fn = LpLoss(d=1, p=2, size_average=False)

    for ep in range(int(config["epochs"])):
        t1 = default_timer()
        model.train()
        train_rel_l2 = 0.0
        n_train = len(train_loader.dataset)
        for x, y, node_mask, nodes, node_weights, condition in train_loader:
            x = x.to(device)
            y = y.to(device)
            node_mask = node_mask.to(device)
            nodes = nodes.to(device)
            node_weights = node_weights.to(device)
            condition = condition.to(device)
            optimizer.zero_grad()
            out = _forward(model, x, node_mask, nodes, node_weights, condition)
            if y_normalizer is not None:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
            out = out * node_mask
            y = y * node_mask
            batch_size = x.shape[0]
            loss = loss_fn(out.view(batch_size, -1), y.view(batch_size, -1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if str(config["scheduler_step"]) == "batch":
                scheduler.step()
            train_rel_l2 += loss.item()
        if str(config["scheduler_step"]) == "epoch":
            scheduler.step()

        test_rel_l2, test_l2 = _evaluate(model, test_loaders, loss_fn, y_normalizer, device)
        train_rel_l2 /= n_train
        train_rel_l2_losses.append(train_rel_l2)
        test_rel_l2_losses.append(test_rel_l2)
        test_l2_losses.append(test_l2)

        t2 = default_timer()
        print(
            "Epoch : ", ep,
            " Time: ", round(t2 - t1, 3),
            " Rel. Train L2 Loss : ", train_rel_l2,
            " Rel. Test L2 Loss : ", test_rel_l2,
            " Test L2 Loss : ", test_l2,
            flush=True,
        )
        if save_model_name and ((ep % 100 == 99) or (ep == int(config["epochs"]) - 1)):
            torch.save(model.state_dict(), save_model_name + ".pth")

    return train_rel_l2_losses, test_rel_l2_losses, test_l2_losses


def _make_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    aux: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    batch_size: int,
    shuffle: bool,
):
    node_mask, nodes, node_weights, condition = aux
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y, node_mask, nodes, node_weights, condition),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def _forward(model, x, node_mask, nodes, node_weights, condition):
    condition_arg = condition if condition.shape[-1] > 0 else None
    return model((x, nodes, condition_arg)) * node_mask


def _evaluate(model, test_loaders, loss_fn, y_normalizer, device):
    model.eval()
    results_rel: dict[str, float] = {}
    results_abs: dict[str, float] = {}
    with torch.no_grad():
        for label, loader in test_loaders:
            rel = 0.0
            abs_loss = 0.0
            n_test = len(loader.dataset)
            for x, y, node_mask, nodes, node_weights, condition in loader:
                x = x.to(device)
                y = y.to(device)
                node_mask = node_mask.to(device)
                nodes = nodes.to(device)
                node_weights = node_weights.to(device)
                condition = condition.to(device)
                out = _forward(model, x, node_mask, nodes, node_weights, condition)
                if y_normalizer is not None:
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)
                out = out * node_mask
                y = y * node_mask
                batch_size = x.shape[0]
                rel += loss_fn(out.view(batch_size, -1), y.view(batch_size, -1)).item()
                abs_loss += loss_fn.abs(out.view(batch_size, -1), y.view(batch_size, -1)).item()
            key = label if label is not None else ""
            results_rel[key] = rel / n_test
            results_abs[key] = abs_loss / n_test

    if len(results_rel) == 1 and "" in results_rel:
        return results_rel[""], results_abs[""]
    return results_rel, results_abs

data_path = "../../data/hifi3d_processed/test"
n_train, n_test = args.n_train, args.n_test
to_divide_factor = args.to_divide_factor
data_file_path = data_path+f"/drivaerml_vertex_centered.npz"
data, node_weights = load_data_to_torch(data_file_path, to_divide = None, factor = to_divide_factor)

ndata = data["nodes"].shape[0]
rng = np.random.default_rng(0)
order = rng.permutation(ndata)
train_idx = order[:n_train]
test_idx = order[n_train:n_train + n_test]
x_train, y_train, aux_train = gen_data_tensors(data, train_idx, node_weights, args.max_nodes)
x_test, y_test, aux_test = gen_data_tensors(data, test_idx, node_weights, args.max_nodes)

layers = [int(size) for size in args.layer_sizes.split(",") if size]
ndim = 3
if args.Ls:
    Ls = [float(value) for value in args.Ls.split(",")]
    if len(Ls) != ndim:
        raise ValueError(f"Expected {ndim} values in --Ls, got {Ls}")
else:
    lengths = torch.amax(aux_train[1], dim=(0, 1)) - torch.amin(aux_train[1], dim=(0, 1))
    Ls = [float(length.item())*2+0.2 for length in lengths]

print(f"Using model= Transolver++", flush=True)
print(f"Using Ls={Ls}, k_max={args.k_max}, layers={layers}", flush=True)
print(f"Using device={device}", flush=True)
print(
        f"x_train={tuple(x_train.shape)} y_train={tuple(y_train.shape)} "
        f"x_test={tuple(x_test.shape)} y_test={tuple(y_test.shape)}",
        flush=True,
)

model = Model(
        space_dim=3,
        fun_dim=x_train.shape[-1] - 3,
        out_dim=y_train.shape[-1],
        n_layers=len(layers),
        n_hidden=layers[0],
        n_head=args.transolver_nhead,
        dropout=args.transolver_dropout,
        mlp_ratio=args.transolver_mlp_ratio,
        slice_num=args.transolver_slice_num,
        ref=args.transolver_ref,
        unified_pos=False,
    ).to(device)

config = {
        "base_lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "normalization_y": args.normalization_y.lower() == "true",
        "scheduler_step": args.scheduler_step,
}

save_model_name = args.save_model_name if args.save_model_name else None
train_rel_l2, test_rel_l2, test_l2 = Transolverpp_train(
        x_train,
        aux_train,
        y_train,
        [(None, x_test, aux_test, y_test)],
        config,
        model,
        save_model_name=save_model_name,
    )

