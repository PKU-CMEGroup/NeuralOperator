import torch


class UnitGaussianNormalizer(object):
    def __init__(self, x, dim, aux_dim=0, eps=1.0e-5):
        super(UnitGaussianNormalizer, self).__init__()
        # x: ndata, nx, nchannels
        # when dim = [], mean and std are both scalars
        self.aux_dim = aux_dim
        self.mean = torch.mean(x[..., 0 : x.shape[-1] - aux_dim], dim)
        self.std = torch.std(x[..., 0 : x.shape[-1] - aux_dim], dim)
        self.eps = eps

    def encode(self, x):
        x[..., 0 : x.shape[-1] - self.aux_dim] = (
            x[..., 0 : x.shape[-1] - self.aux_dim] - self.mean
        ) / (self.std + self.eps)
        return x

    def decode(self, x):
        std = self.std + self.eps  # n
        mean = self.mean
        x[..., 0 : x.shape[-1] - self.aux_dim] = (
            x[..., 0 : x.shape[-1] - self.aux_dim] * std
        ) + mean
        return x

    def to(self, device):
        if device == torch.device("cuda:0"):
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()
        else:
            self.mean = self.mean.cpu()
            self.std = self.std.cpu()


# class UnitGaussianNormalizer(object):
#     def __init__(self, x, dim=[], eps=1.0e-5):
#         super(UnitGaussianNormalizer, self).__init__()

#         # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
#         # when dim = [], mean and std are both scalars
#         self.mean = torch.mean(x, dim)
#         self.std = torch.std(x, dim)
#         self.eps = eps

#     def encode(self, x):
#         return (x - self.mean) / (self.std + self.eps)

#     # inplace function
#     def encode_(self, x):
#         x -= self.mean
#         x /= self.std + self.eps

#     def decode(self, x):
#         std = self.std + self.eps  # n
#         mean = self.mean
#         return (x * std) + mean

#     # inplace function
#     def decode_(self, x):
#         std = self.std + self.eps  # n
#         mean = self.mean
#         x *= std
#         x += mean

#     def to(self, device):
#         if device == torch.device("cuda:0"):
#             self.mean = self.mean.cuda()
#             self.std = self.std.cuda()
#         elif device == torch.device("mps"):
#             self.mean = self.mean.to("mps")
#             self.std = self.std.to("mps")
#         else:
#             self.mean = self.mean.cpu()
#             self.std = self.std.cpu()
