import torch
import torch.nn as nn
from timeit import default_timer
from utility.losses import LpLoss
from utility.normalizer import UnitGaussianNormalizer

from .pcno import (
    _get_act,
    scaled_sigmoid,
    scaled_logit,
    compute_Fourier_modes_helper,
    compute_Fourier_modes,
    compute_Fourier_bases,
    SpectralConv,
    compute_gradient,
    CombinedOptimizer,
    Combinedscheduler_OneCycleLR,
    PCNO_train,
)


class StructuredNN(nn.Module):
    def __init__(self, in_dim, mid_layers, out_dim, act=None):
        """
        Network structure: input -> mid_layers[0] -> ... -> mid_layers[-1] -> output

        Args:
            in_dim: Input dimension
            mid_layers: List of middle layer dimensions
            out_dim: Output dimension
            act: Activation function (callable)
        """
        super(StructuredNN, self).__init__()

        self.act = act
        self.layers = nn.ModuleList()

        prev_dim = in_dim
        for hidden_dim in mid_layers:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        self.layers.append(nn.Linear(prev_dim, out_dim))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1 and self.act is not None:
                x = self.act(x)
        return x


class PCNO(nn.Module):
    def __init__(
        self,
        ndims,
        modes,
        nmeasures,
        layers,
        fc_dim=128,
        proj_layers=None,
        in_dim=3,
        out_dim=1,
        inv_L_scale_hyper=['independently', 0.5, 2.0],
        act="gelu",
        proj_act="gelu",
    ):
        super(PCNO, self).__init__()

        self.register_buffer('modes', modes)
        self.nmeasures = nmeasures

        self.layers = layers
        self.fc_dim = fc_dim
        self.proj_layers = proj_layers

        self.ndims = ndims
        self.in_dim = in_dim

        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList(
            [
                SpectralConv(in_size, out_size, modes)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )
        self.train_inv_L_scale, self.inv_L_scale_min, self.inv_L_scale_max = (
            inv_L_scale_hyper[0],
            inv_L_scale_hyper[1],
            inv_L_scale_hyper[2],
        )
        self.inv_L_scale_latent = nn.Parameter(
            torch.full(
                (ndims, nmeasures),
                scaled_logit(
                    torch.tensor(1.0),
                    self.inv_L_scale_min,
                    self.inv_L_scale_max,
                ),
            ),
            requires_grad=bool(self.train_inv_L_scale),
        )

        self.ws = nn.ModuleList(
            [
                nn.Conv1d(in_size, out_size, 1)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )

        self.gws = nn.ModuleList(
            [
                nn.Conv1d(ndims * in_size, out_size, 1)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )

        self.act = _get_act(act)
        self.proj_act = _get_act(proj_act)
        if proj_layers is None:
            proj_layers = [fc_dim] if fc_dim > 0 else []
        elif isinstance(proj_layers, int):
            proj_layers = [proj_layers] if proj_layers > 0 else []
        else:
            proj_layers = [int(width) for width in proj_layers if int(width) > 0]
        self.proj = StructuredNN(
            in_dim=layers[-1],
            mid_layers=proj_layers,
            out_dim=out_dim,
            act=self.proj_act,
        )

        self.softsign = torch.nn.functional.softsign

        self.normal_params = []
        self.inv_L_scale_params = []
        for _, param in self.named_parameters():
            if param is not self.inv_L_scale_latent:
                self.normal_params.append(param)
            else:
                if self.train_inv_L_scale == 'together':
                    self.normal_params.append(param)
                elif self.train_inv_L_scale == 'independently':
                    self.inv_L_scale_params.append(param)
                elif self.train_inv_L_scale is False:
                    continue
                else:
                    raise ValueError(f"{self.train_inv_L_scale} is not supported")

    def forward(self, x, aux):
        length = len(self.ws)

        node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = aux

        bases_c, bases_s, bases_0 = compute_Fourier_bases(
            nodes,
            self.modes
            * (
                scaled_sigmoid(
                    self.inv_L_scale_latent,
                    self.inv_L_scale_min,
                    self.inv_L_scale_max,
                )
            ),
        )

        wbases_c = torch.einsum("bxkw,bxw->bxkw", bases_c, node_weights)
        wbases_s = torch.einsum("bxkw,bxw->bxkw", bases_s, node_weights)
        wbases_0 = torch.einsum("bxkw,bxw->bxkw", bases_0, node_weights)

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, w, gw) in enumerate(zip(self.sp_convs, self.ws, self.gws)):
            x1 = speconv(x, bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0)
            x2 = w(x)
            x3 = gw(self.softsign(compute_gradient(x, directed_edges, edge_gradient_weights)))
            x = x1 + x2 + x3
            if self.act is not None and i != length - 1:
                x = self.act(x)

        x = x.permute(0, 2, 1)
        x = self.proj(x)

        return x


def PCNO_train_multidist(
    x_train,
    aux_train,
    y_train,
    x_test_list,
    aux_test_list,
    y_test_list,
    config,
    model,
    label_test_list=None,
    save_model_name="./PCNO_model",
    checkpoint_path=None,
):
    assert len(x_test_list) == len(y_test_list) == len(aux_test_list), (
        "The length of x_test_list, y_test_list and aux_test_list should be the same"
    )
    n_distributions = len(x_test_list)
    n_train = x_train.shape[0]
    train_rel_l2_losses = []
    test_rel_l2_losses = []
    test_l2_losses = []
    normalization_x = config["train"]["normalization_x"]
    normalization_y = config["train"]["normalization_y"]
    normalization_dim_x = config["train"]["normalization_dim_x"]
    normalization_dim_y = config["train"]["normalization_dim_y"]
    non_normalized_dim_x = config["train"]["non_normalized_dim_x"]
    non_normalized_dim_y = config["train"]["non_normalized_dim_y"]

    ndims = model.ndims
    print("In PCNO_train_multidist, ndims = ", ndims)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if normalization_x:
        x_normalizer = UnitGaussianNormalizer(
            x_train,
            non_normalized_dim=non_normalized_dim_x,
            normalization_dim=normalization_dim_x,
        )
        x_train = x_normalizer.encode(x_train)
        for i in range(n_distributions):
            x_test_list[i] = x_normalizer.encode(x_test_list[i])
        x_normalizer.to(device)

    if normalization_y:
        y_normalizer = UnitGaussianNormalizer(
            y_train,
            non_normalized_dim=non_normalized_dim_y,
            normalization_dim=normalization_dim_y,
        )
        y_train = y_normalizer.encode(y_train)
        for i in range(n_distributions):
            y_test_list[i] = y_normalizer.encode(y_test_list[i])
        y_normalizer.to(device)

    node_mask_train, nodes_train, node_weights_train, directed_edges_train, edge_gradient_weights_train = aux_train
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            x_train,
            y_train,
            node_mask_train,
            nodes_train,
            node_weights_train,
            directed_edges_train,
            edge_gradient_weights_train,
        ),
        batch_size=config["train"]["batch_size"],
        shuffle=True,
    )

    test_loaders = []
    for i in range(n_distributions):
        node_mask_test, nodes_test, node_weights_test, directed_edges_test, edge_gradient_weights_test = aux_test_list[i]
        sub_dataset = torch.utils.data.TensorDataset(
            x_test_list[i],
            y_test_list[i],
            node_mask_test,
            nodes_test,
            node_weights_test,
            directed_edges_test,
            edge_gradient_weights_test,
        )
        sub_loader = torch.utils.data.DataLoader(
            sub_dataset, batch_size=config["train"]["batch_size"], shuffle=False
        )
        name = label_test_list[i] if label_test_list is not None else f"Distribution_{i}"
        test_loaders.append((name, sub_loader))

    myloss = LpLoss(d=1, p=2, size_average=False)

    optimizer = CombinedOptimizer(
        model.normal_params,
        model.inv_L_scale_params,
        betas=(0.9, 0.999),
        lr=config["train"]["base_lr"],
        lr_ratio=config["train"]["lr_ratio"],
        weight_decay=config["train"]["weight_decay"],
    )

    scheduler = Combinedscheduler_OneCycleLR(
        optimizer,
        max_lr=config["train"]["base_lr"],
        lr_ratio=config["train"]["lr_ratio"],
        div_factor=2,
        final_div_factor=100,
        pct_start=0.2,
        steps_per_epoch=len(train_loader),
        epochs=config["train"]["epochs"],
    )

    current_epoch, epochs = 0, config["train"]["epochs"]

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        current_epoch = checkpoint["current_epoch"] + 1
        print("resetart from epoch : ", current_epoch)

    for ep in range(current_epoch, epochs):
        t1 = default_timer()
        train_rel_l2 = 0

        model.train()
        for x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights in train_loader:
            x = x.to(device)
            y = y.to(device)
            node_mask = node_mask.to(device)
            nodes = nodes.to(device)
            node_weights = node_weights.to(device)
            directed_edges = directed_edges.to(device)
            edge_gradient_weights = edge_gradient_weights.to(device)

            batch_size_ = x.shape[0]
            optimizer.zero_grad()
            out = model(
                x,
                (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights),
            )
            if normalization_y:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
            out = out * node_mask
            loss = myloss(out.view(batch_size_, -1), y.view(batch_size_, -1))

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
                for x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights in loader:
                    x = x.to(device)
                    y = y.to(device)
                    node_mask = node_mask.to(device)
                    nodes = nodes.to(device)
                    node_weights = node_weights.to(device)
                    directed_edges = directed_edges.to(device)
                    edge_gradient_weights = edge_gradient_weights.to(device)

                    batch_size_ = x.shape[0]
                    out = model(
                        x,
                        (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights),
                    )

                    if normalization_y:
                        out = y_normalizer.decode(out)
                        y = y_normalizer.decode(y)
                    out = out * node_mask
                    test_rel_l2 += myloss(out.view(batch_size_, -1), y.view(batch_size_, -1)).item()
                    test_l2 += myloss.abs(out.view(batch_size_, -1), y.view(batch_size_, -1)).item()

                test_l2 /= len(loader.dataset)
                test_rel_l2 /= len(loader.dataset)
                test_rel_l2_dict[name] = test_rel_l2
                test_l2_dict[name] = test_l2

        train_rel_l2 /= n_train
        train_rel_l2_losses.append(train_rel_l2)
        test_rel_l2_losses.append(test_rel_l2_dict)
        test_l2_losses.append(test_l2_dict)

        t2 = default_timer()
        print(
            "Epoch : ",
            ep,
            " Time: ",
            round(t2 - t1, 3),
            " Rel. Train L2 Loss : ",
            train_rel_l2,
            " Rel. Test L2 Loss : ",
            test_rel_l2_dict,
            " Test L2 Loss : ",
            test_l2_dict,
            " inv_L_scale: ",
            [
                round(float(x[0]), 3)
                for x in (
                    scaled_sigmoid(
                        model.inv_L_scale_latent,
                        model.inv_L_scale_min,
                        model.inv_L_scale_max,
                    )
                ).cpu().tolist()
            ],
            flush=True,
        )
        if (ep % 100 == 99) or (ep == epochs - 1):
            if save_model_name:
                torch.save(model.state_dict(), save_model_name + ".pth")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "current_epoch": ep,
                    },
                    "checkpoint.pth",
                )

    return train_rel_l2_losses, test_rel_l2_losses, test_l2_losses
