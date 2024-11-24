import torch
import operator
from functools import reduce
from timeit import default_timer
# import dgl

from models.adam import Adam
from models.losses import LpLoss
from .normalizer import UnitGaussianNormalizer
from .basics import count_params


def model_train(
    data_dict,
    config,
    model,
    should_print=True,
    boundary_indices=False,
    save_model_name=False,
):
    if should_print == True:
        for section, settings in config.items():
            print("#" * 60, "config of", section, "#" * 60, flush=True)
            for key, value in settings.items():
                print(f"{key}: {value}", flush=True)

    n_train = config["data"]["n_train"]
    n_test = config["data"]["n_test"]
    epochs = config["train"]["epochs"]
    batch_size = config["train"]["batch_size"]
    device = torch.device(config["train"]["device"])

    x_train = data_dict["xtrain"]
    y_train = data_dict["ytrain"]
    x_test = data_dict["xtest"]
    y_test = data_dict["ytest"]

    # normalization
    normalization_x, normalization_y, normalization_dim = (
        config["train"]["normalization_x"],
        config["train"]["normalization_y"],
        config["train"]["normalization_dim"],
    )

    aux_dim = config["model"]["aux_dim"]
    if normalization_x:
        x_normalizer = UnitGaussianNormalizer(
            x_train, normalization_dim, aux_dim=aux_dim
        )
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)
        x_normalizer.to(device)
    if normalization_y:
        y_normalizer = UnitGaussianNormalizer(
            y_train, normalization_dim, aux_dim=0)
        y_train = y_normalizer.encode(y_train)
        y_test = y_normalizer.encode(y_test)
        y_normalizer.to(device)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=config["train"]["batch_size"],
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test),
        batch_size=config["train"]["batch_size"],
        shuffle=False,
    )

    # Load from checkpoint
    optimizer = Adam(
        model.parameters(),
        betas=(0.9, 0.999),
        lr=config["train"]["base_lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    if config["train"]["scheduler"] == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config["train"]["milestones"],
            gamma=config["train"]["scheduler_gamma"],
        )
    elif config["train"]["scheduler"] == "CosineAnnealingLR":
        T_max = (config["train"]["epochs"] // 10) * (
            n_train // config["train"]["batch_size"]
        )
        eta_min = 0.0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )
    elif config["train"]["scheduler"] == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config["train"]["base_lr"],
            div_factor=2,
            final_div_factor=100,
            pct_start=0.2,
            steps_per_epoch=1,
            epochs=config["train"]["epochs"],
        )

    else:
        raise KeyError(
            f"Scheduler {config['train']['scheduler']} has not implemented.")

    # start training
    myloss = LpLoss(d=1, p=2, size_average=False)
    print("#" * 60, "Start Training", "#" * 60, flush=True)
    print(f"number of parameters is {count_params(model)}", flush=True)
    if boundary_indices == False:
        print("no boundary loss", flush=True)
    else:
        print("compute boundary loss", flush=True)
    for ep in range(epochs):
        start_time = default_timer()
        train_rel_l2 = 0
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            batch_size_ = x.shape[0]
            optimizer.zero_grad()
            out = model(x)

            if normalization_y:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)

            loss = myloss(out.view(batch_size_, -1), y.view(batch_size_, -1))
            loss.backward()

            optimizer.step()
            train_rel_l2 += loss.item()

        test_abs_l2 = 0
        test_rel_l2 = 0
        test_boundary_l2 = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                batch_size_ = x.shape[0]
                out = model(x)

                if normalization_y:
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)

                test_abs_l2 += myloss.abs(
                    out.view(batch_size_, -1), y.view(batch_size_, -1)
                ).item()
                test_rel_l2 += myloss(
                    out.view(batch_size_, -1), y.view(batch_size_, -1)
                ).item()
                test_boundary_l2 += myloss.abs(
                    out.view(batch_size_, -1)[:, boundary_indices],
                    y.view(batch_size_, -1)[:, boundary_indices],
                ).item()

        scheduler.step()

        train_rel_l2 /= n_train
        test_abs_l2 /= n_test
        test_rel_l2 /= n_test
        test_boundary_l2 /= n_test

        end_time = default_timer()
        print(
            f"epoch {ep}",
            " rel.Train:",
            f"{train_rel_l2:.10f}",
            " rel.Test:",
            f"{test_rel_l2:.10f}",
            " abs.Test:",
            f"{test_abs_l2:.10f}",
            # " abs.TestBoundary:", f"{test_boundary_l2:.6f}",
            " time:",
            f"{end_time - start_time:.3f}",
            # f" L:[{model.bases.Lx.item():.6f},{model.bases.Ly.item():.6f}]",
            flush=True,
        )

        if save_model_name:
            torch.save(model, save_model_name)


def graph_train(
    train_loader,
    test_loader,
    y_normalizer,
    config,
    model,
    should_print=True,
    boundary_indices=False,
    save_model_name=False,
):
    if should_print == True:
        for section, settings in config.items():
            print("#" * 60, "config of", section, "#" * 60, flush=True)
            for key, value in settings.items():
                print(f"{key}: {value}", flush=True)

    n_train = config["data"]["n_train"]
    n_test = config["data"]["n_test"]
    epochs = config["train"]["epochs"]
    normalization_y = config["train"]["normalization_y"]

    device = torch.device(config["train"]["device"])

    # normalization

    # Load from checkpoint
    optimizer = Adam(
        model.parameters(),
        betas=(0.9, 0.999),
        lr=config["train"]["base_lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    if config["train"]["scheduler"] == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config["train"]["milestones"],
            gamma=config["train"]["scheduler_gamma"],
        )
    elif config["train"]["scheduler"] == "CosineAnnealingLR":
        T_max = (config["train"]["epochs"] // 10) * (
            n_train // config["train"]["batch_size"]
        )
        eta_min = 0.0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )
    elif config["train"]["scheduler"] == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config["train"]["base_lr"],
            div_factor=2,
            final_div_factor=100,
            pct_start=0.2,
            steps_per_epoch=1,
            epochs=config["train"]["epochs"],
        )

    else:
        raise KeyError(
            f"Scheduler {config['train']['scheduler']} has not implemented.")

    # start training
    myloss = LpLoss(d=1, p=2, size_average=False)
    print("#" * 60, "Start Training", "#" * 60, flush=True)
    print(f"number of parameters is {count_params(model)}", flush=True)
    if boundary_indices == False:
        print("no boundary loss", flush=True)
    else:
        print("compute boundary loss", flush=True)
    for ep in range(epochs):
        start_time = default_timer()
        train_rel_l2 = 0
        model.train()
        for g, x, y in train_loader:
            g, x, y = g.to(device), x.to(device), y.to(device)
            batch_size_ = x.shape[0]
            optimizer.zero_grad()
            out = model(g, x)

            if normalization_y:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)

            loss = myloss(out.view(batch_size_, -1), y.view(batch_size_, -1))
            loss.backward()

            optimizer.step()
            train_rel_l2 += loss.item()

        test_abs_l2 = 0
        test_rel_l2 = 0
        test_boundary_l2 = 0
        with torch.no_grad():
            for g, x, y in test_loader:
                g, x, y = g.to(device), x.to(device), y.to(device)
                batch_size_ = x.shape[0]
                out = model(g, x)

                if normalization_y:
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)

                test_abs_l2 += myloss.abs(
                    out.view(batch_size_, -1), y.view(batch_size_, -1)
                ).item()
                test_rel_l2 += myloss(
                    out.view(batch_size_, -1), y.view(batch_size_, -1)
                ).item()
                test_boundary_l2 += myloss.abs(
                    out.view(batch_size_, -1)[:, boundary_indices],
                    y.view(batch_size_, -1)[:, boundary_indices],
                ).item()

        scheduler.step()

        train_rel_l2 /= n_train
        test_abs_l2 /= n_test
        test_rel_l2 /= n_test
        test_boundary_l2 /= n_test

        end_time = default_timer()
        print(
            f"epoch{ep}",
            " rel.Train:",
            train_rel_l2,
            " rel.Test:",
            test_rel_l2,
            " abs.Test:",
            test_abs_l2,
            " abs.TestBoundary:",
            test_boundary_l2,
            " time:",
            end_time - start_time,
            flush=True,
        )

        if save_model_name:
            torch.save(model, save_model_name)
