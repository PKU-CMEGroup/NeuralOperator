import h5py
import torch
import scipy.io
import operator
import numpy as np
from Adam import Adam
from functools import reduce
import matplotlib.pyplot as plt
from timeit import default_timer


class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path, mode="r")
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size() + (2,) if p.is_complex() else p.size()))
    return c


def mytrain(
    model,
    x,
    y,
    grid,
    learning_rate,
    step_size,
    # milestones,
    gamma,
    epochs,
    batch_size,
    ntrain,
    ntest,
    save_path,
):

    print("ntrain:", ntrain, "ntest:", ntest)
    print("using l2 loss in training and testing")

    x_train = x[:ntrain, :, :]
    y_train = y[:ntrain, :]

    x_test = x[-ntest:, :, :]
    y_test = y[-ntest:, :]

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=learning_rate,
    #     pct_start=0.5,
    #     div_factor=5,
    #     final_div_factor=10,
    #     steps_per_epoch=len(train_loader),
    #     epochs=epochs,
    # )
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=milestones, gamma=gamma
    # )

    myloss = LpLoss(size_average=False)

    Loss_train = []
    Loss_test = []
    last_lr = 0
    t_start = default_timer()

    print("=" * 60)
    print("begin training")
    for ep in range(epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr != last_lr:
            print(f"learning rate changed: {current_lr}")
        last_lr = current_lr
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)

            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            l2.backward()
            optimizer.step()
            train_l2 += l2.item()

        scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                out = model(x)
                test_l2 += myloss(
                    out.view(batch_size, -1), y.view(batch_size, -1)
                ).item()
        train_l2 /= ntrain
        test_l2 /= ntest

        t2 = default_timer()
        print(
            f"{ep + 1:03} | time {(t2 - t1):.2f} | train {train_l2:.12f} | test {test_l2:.12f}",
        )

        Loss_train.append(train_l2)
        Loss_test.append(test_l2)

    t_end = default_timer()
    print("Training Time In Total:", t_end - t_start)
    torch.save(model, save_path)

    ### plotting
    nplot = 4
    samples = []
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if i == nplot:
                break
            else:
                y_pred = model(x)[0]
                x, y = x[0, :, 0], y[0]
                samples.append(
                    (x.squeeze().numpy(), y.squeeze().numpy(), y_pred.squeeze().numpy())
                )

    _, axes = plt.subplots(nplot + 1, 1, figsize=(10, 2.5 * (nplot + 1)))
    for i, (x, y, y_pred) in enumerate(samples):
        ax = axes[i]
        ax.scatter(grid, x, marker=".", label="input")
        ax.plot(grid, y, label="ground truth")
        ax.plot(grid, y_pred, linestyle="--", label="prediction")
        ax.legend()

    axes[nplot].plot(np.arange(epochs), Loss_test, label="test loss")
    axes[nplot].plot(np.arange(epochs), Loss_train, label="train loss")
    axes[nplot].set_yscale("log")
    axes[nplot].legend()

    plt.show()

    return Loss_train, Loss_test


