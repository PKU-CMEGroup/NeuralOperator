import torch


A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = torch.tensor([1, 2, 3])
print(A)
print(A / b.unsqueeze(1))
