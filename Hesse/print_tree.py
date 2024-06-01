import torch
# torch.set_printoptions(profile="full")

vec = torch.load("1.3b-70b_tree.pt")
# tree = torch.load("demo_tree.pt")
tree_mask :torch.Tensor = vec["mask"]
tree_mask = (tree_mask == 0).type(torch.float16)

tree_mask.masked_fill_(tree_mask > 0, torch.finfo(torch.float16).min)
print(vec)