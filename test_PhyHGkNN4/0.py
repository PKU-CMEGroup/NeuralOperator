import torch
import numpy as np
import math
# x = torch.arange(-7, 8) 
# y = torch.arange(-7, 8) 
# X, Y = torch.meshgrid(x, y)
# baseweight_Fourier = 2*np.pi*torch.stack((X.flatten(), Y.flatten()), dim=1)
# torch.save(baseweight_Fourier, 'para/darcy/uniform_Fourier_225.pt')
# Fourier_para_path = 'para/airfoil/base_Fourier225.pt'
# Fourier_weight_list = torch.load(Fourier_para_path)
# Fourier_weight,a = Fourier_weight_list
# torch.save(Fourier_weight,'para/airfoil/base_Fourier225[0].pt')
def uniform_points(range_pts,num_pts,dim):
    a = round(math.pow(num_pts, 1 / dim))
    index_tensors = []
    for k in range(dim):
        xmin,xmax = range_pts[k][0],range_pts[k][1]
        idx = xmin + (xmax-xmin)*torch.arange(a).float().add(0.5).div(a)
        idx = idx.view((1,) * k+ (-1,) + (1,) * (dim - k - 1))
        index_tensors.append(idx.expand(a, *([a] * (dim - 1))))
    num_pts1 = int(torch.pow(torch.tensor(a),dim))
    x = torch.stack(index_tensors, dim=dim).reshape(num_pts1,dim)
    return x
basepts = uniform_points([[0,1]],200,1)
baseweight = 1000*torch.ones(200,1)
# # print(basepts)
torch.save([basepts,baseweight],'para/advection/Gauss200_1000_uniform.pt')
# Fourier_weight = torch.arange(-225,225).unsqueeze(-1)
# torch.save(Fourier_weight,'para/advection/Fourier449_uniform.pt')