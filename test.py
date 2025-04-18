import torch
tensor = torch.tensor([[0.1, 0.2],
                       [0.4, 0.5],
                       [0.7, 0.8]])
action_batch = torch.tensor([1, 0, 0])
action = action_batch.view(1, -1)
print(action)
result = tensor.gather(1, action)
print(result)