import numpy as np
import torch
import os
from torch.utils.data import Dataset
from timeit import default_timer

from utility.losses import LpLoss
from utility.normalizer import UnitGaussianNormalizer
from pcno.pcno import CombinedOptimizer, Combinedscheduler_OneCycleLR


class BNOAuxedDataset(Dataset):
    def __init__(self, x, y, truth, aux):

        self.x = x
        self.y = y
        self.truth = truth
        self.aux = aux

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        aux_items = tuple(a[idx] for a in self.aux)
        return (self.x[idx], self.y[idx], self.truth[idx], aux_items)


def compute_epoch_losses(loss_list, m_tests):

    loss = torch.cat(loss_list)
    loss_list = [torch.sum(loss).item() / sum(m_tests)]
    median_loss_list = [torch.median(loss).item()]
    max_loss_list = [torch.max(loss).item()]

    i = 0
    for m in m_tests:
        losses_shape = loss[i:i + m]
        loss_list.append(torch.sum(losses_shape).item() / m)
        median_loss_list.append(torch.median(losses_shape).item())
        max_loss_list.append(torch.max(losses_shape).item())
        i += m

    return loss_list, median_loss_list, max_loss_list


def BNO_train(model,
              x_train, y_train, truth_train, aux_train,
              x_test, y_test, truth_test, aux_test,
              config, checkpoint_path=None, save_model_ID=None):

    model_type = model.model_type.lower()
    os.makedirs(f"./saves/{model_type}", exist_ok=True)
    os.makedirs(f"./losses/{model_type}", exist_ok=True)
    model_save_path = f"./saves/{model_type}/{save_model_ID}_onlystate.pth"
    losses_save_path = f"./losses/{model_type}/{save_model_ID}_losses.npz"
    print(f"The save path of model and optimizer has been set to {model_save_path}", flush=True)
    print(f"The save path of losses has been set to {losses_save_path}", flush=True)

    m_tests = config["test"]["m_tests"]
    shape_types = config["test"]["shape_types"]
    if len(shape_types) > 1:
        shape_types = ['mixed'] + shape_types
    n_train, n_test = y_train.shape[0], y_test.shape[0]
    device = config["train"]["device"]

    normalization_x, normalization_y = config["train"]["normalization_x"], config["train"]["normalization_y"]
    normalization_dim_x, normalization_dim_y = config["train"]["normalization_dim_x"], config["train"]["normalization_dim_y"]
    non_normalized_dim_x, non_normalized_dim_y = config["train"]["non_normalized_dim_x"], config["train"]["non_normalized_dim_y"]
    normalization_out = config["train"]["normalization_out"]
    normalization_dim_out = config["train"]["normalization_dim_out"]
    non_normalized_dim_out = config["train"]["non_normalized_dim_out"]
    if normalization_x:
        x_normalizer = UnitGaussianNormalizer(x_train,
                                              non_normalized_dim=non_normalized_dim_x, normalization_dim=normalization_dim_x)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)
        x_normalizer.to(device)
    if normalization_y:
        y_normalizer = UnitGaussianNormalizer(y_train,
                                              non_normalized_dim=non_normalized_dim_y, normalization_dim=normalization_dim_y)
        y_train = y_normalizer.encode(y_train)
        y_test = y_normalizer.encode(y_test)
        y_normalizer.to(device)
    if normalization_out:
        out_normalizer = UnitGaussianNormalizer(truth_train,
                                                non_normalized_dim=non_normalized_dim_out, normalization_dim=normalization_dim_out)
        truth_train = out_normalizer.encode(truth_train)
        truth_test = out_normalizer.encode(truth_test)
        out_normalizer.to(device)

    batch_size = config['train']['batch_size']
    train_loader = torch.utils.data.DataLoader(BNOAuxedDataset(x_train, y_train, truth_train, aux_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(BNOAuxedDataset(x_test, y_test, truth_test, aux_test), batch_size=batch_size, shuffle=False)

    optimizer = CombinedOptimizer(model.normal_params, model.sp_L_params,
                                  betas=(0.9, 0.999), lr=config["train"]["base_lr"], lr_ratio=config["train"]["lr_ratio"],
                                  weight_decay=config["train"]["weight_decay"])
    scheduler = Combinedscheduler_OneCycleLR(optimizer,
                                             max_lr=config['train']['base_lr'], lr_ratio=config["train"]["lr_ratio"], div_factor=2, final_div_factor=100, pct_start=0.2, steps_per_epoch=1, epochs=config['train']['epochs'])
    print("Combined optimizer and scheduler are being used.")

    current_epoch, epochs = 0, config['train']['epochs']
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # retrieve epoch and loss
        current_epoch = checkpoint['current_epoch'] + 1
        print("resetart from epoch : ", current_epoch)

    myloss = LpLoss(d=1, p=2, size_average=False)
    eploss = LpLoss(d=1, p=2, reduction=False)

    train_l2_losses = []
    train_rel_l2_losses = []
    test_l2_losses = []

    test_rel_l2_losses_dict = {shape: [] for shape in shape_types}
    test_rel_l2_median_losses_dict = {shape: [] for shape in shape_types}
    test_rel_l2_worst_losses_dict = {shape: [] for shape in shape_types}

    print("\nStart Training...")
    for ep in range(current_epoch, epochs):

        t1 = default_timer()

        model.train()
        train_l2 = 0
        train_rel_l2 = 0
        for x, y, truth, aux in train_loader:
            optimizer.zero_grad()

            x, y, truth = x.to(device), y.to(device), truth.to(device)
            aux = tuple(a.to(device) for a in aux)

            mask_x = aux[0]
            batch_size_ = y.shape[0]

            pred = model(x, y, aux)
            if normalization_out:
                pred = out_normalizer.decode(pred)
                truth = out_normalizer.decode(truth)
            pred = pred * mask_x

            loss = myloss(pred.view(batch_size_, -1), truth.view(batch_size_, -1))
            loss.backward()

            optimizer.step()
            train_l2 += myloss.abs(pred.view(batch_size_, -1), truth.view(batch_size_, -1)).item()
            train_rel_l2 += loss.item()

        model.eval()
        test_l2 = 0
        test_rel_l2_list = []
        with torch.no_grad():
            for x, y, truth, aux in test_loader:

                x, y, truth = x.to(device), y.to(device), truth.to(device)
                aux = tuple(a.to(device) for a in aux)
                mask_x = aux[0]
                batch_size_ = y.shape[0]

                pred = model(x, y, aux)

                if normalization_out:
                    pred = out_normalizer.decode(pred)
                    truth = out_normalizer.decode(truth)
                pred = pred * mask_x

                test_l2 += myloss.abs(pred.view(batch_size_, -1), truth.view(batch_size_, -1)).item()
                test_rel_l2_list.append(eploss.rel(pred.view(batch_size_, -1), truth.view(batch_size_, -1)))
        scheduler.step()
        t2 = default_timer()

        train_l2 /= n_train
        train_rel_l2 /= n_train
        test_l2 /= n_test

        train_l2_losses.append(train_l2)
        train_rel_l2_losses.append(train_rel_l2)
        test_l2_losses.append(test_l2)

        loss_list, median_loss_list, worst_loss_list = compute_epoch_losses(test_rel_l2_list, m_tests)
        for j, shape in enumerate(shape_types):
            test_rel_l2_losses_dict[shape].append(loss_list[j])
            test_rel_l2_median_losses_dict[shape].append(median_loss_list[j])
            test_rel_l2_worst_losses_dict[shape].append(worst_loss_list[j])

        print(f"[{ep:03d}] time={(t2 - t1):.02}  1/L={[round(float(x[0]), 3) for x in model.sp_L.cpu().tolist()]}", flush=True)
        print(f"{' ' * 6}Train: rel.l2={train_rel_l2} abs.l2={train_l2}", flush=True)
        for shape in shape_types:
            print(f"{' ' * 6}{shape}: average={test_rel_l2_losses_dict[shape][-1]} median={test_rel_l2_median_losses_dict[shape][-1]} worst={test_rel_l2_worst_losses_dict[shape][-1]}", flush=True)

        if (ep % 100 == 99) or (ep == epochs - 1):
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'current_epoch': ep,  # optional: to track training progress
            }, model_save_path)
            np.savez(losses_save_path,
                     train_l2_losses=train_l2_losses,
                     train_rel_l2_losses=train_rel_l2_losses,
                     test_l2_losses=test_l2_losses,
                     test_rel_l2_losses_dict=test_rel_l2_losses_dict,
                     test_rel_l2_median_losses_dict=test_rel_l2_median_losses_dict,
                     test_rel_l2_worst_losses_dict=test_rel_l2_worst_losses_dict)
