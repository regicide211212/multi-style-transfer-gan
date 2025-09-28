import torch

ckpt_path = r"E:\desktop\gan_proj_0217\monet2photo_uvcgan2\models\generator_pretrain_epoch_350.pth"
checkpoint = torch.load(ckpt_path, map_location="cpu")

# 如果里面存的是字典，通常会包含 'model_state_dict' 等内容
if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
    for name, param in state_dict.items():
        print(name, param.shape)
else:
    # 如果直接保存了 model.state_dict()，可以这样
    for name, param in checkpoint.items():
        print(name, param.shape)

