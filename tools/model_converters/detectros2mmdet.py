import torch

ckpt = torch.load('data/pretrained/DetectoRS_X101-ed983634.pth', map_location='cpu')

new_state_dict = {}
results = []
for k, v in ckpt['state_dict'].items():
    # if
    # k_ =
    # new_state_dict
    results.append(k)
    # print()
    if 'rfp_modules' in k:
        k_ = 'neck.' + k


print(results)