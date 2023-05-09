import torch.nn.functional as F


def add_padding(x, pad_nums):

    if x.ndim == 3: #fourier1d
        res = F.pad(x, [0, pad_nums[0]], 'constant', 0)
    elif x.ndim == 4: #fourier2d
        res = F.pad(x, [0, pad_nums[1], 0, pad_nums[0]], 'constant', 0)
    elif x.ndim == 5: #fourier3d
        res = F.pad(x, [0, pad_nums[2], 0, pad_nums[1], 0, pad_nums[0]], 'constant', 0)
    else:
        print("error : x.ndim = ", x.ndim)
            
    return res


def remove_padding(x, pad_nums):
    
    if x.ndim == 3: #fourier1d
        res = x[..., :(None if pad_nums[0] == 0 else -pad_nums[0])]
        
    elif x.ndim == 4: #fourier2d
        res = x[..., :(None if pad_nums[0] == 0 else -pad_nums[0]), :(None if pad_nums[1] == 0 else -pad_nums[1])]

    elif x.ndim == 5: #fourier3d
        res = x[..., :(None if pad_nums[0] == 0 else -pad_nums[0]), :(None if pad_nums[1] == 0 else -pad_nums[1]), :(None if pad_nums[2] == 0 else -pad_nums[2])]
        
    else:
        print("error : x.ndim = ", x.ndim)
    
    return res


def _get_act(act):
    if act == 'tanh':
        func = F.tanh
    elif act == 'gelu':
        func = F.gelu
    elif act == 'relu':
        func = F.relu_
    elif act == 'elu':
        func = F.elu_
    elif act == 'leaky_relu':
        func = F.leaky_relu_
    elif act == 'none':
        func = None
    else:
        raise ValueError(f'{act} is not supported')
    return func

