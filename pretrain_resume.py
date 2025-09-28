import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import numpy as np
from pathlib import Path

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 数据集类
class MonetPhotoDataset(Dataset):
    def __init__(self, root_dir, domain, split='train', img_size=256):
        self.root_dir = Path(root_dir)
        self.domain = domain
        self.split = split
        self.img_size = img_size
        
        # 获取图像路径
        self.image_paths = list((self.root_dir / f"{split}{domain}").glob("*.jpg"))
        self.image_paths.extend(list((self.root_dir / f"{split}{domain}").glob("*.png")))
        
        # 数据转换
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # 创建mask (32x32网格，随机遮盖40%的块)
        mask = torch.ones_like(image)
        patch_size = self.img_size // 8  # 32x32 patches
        for i in range(8):
            for j in range(8):
                if random.random() < 0.4:  # 40% 概率遮盖
                    mask[:, i*patch_size:(i+1)*patch_size, 
                         j*patch_size:(j+1)*patch_size] = 0
        
        masked_image = image * mask
        return masked_image, image, mask

# 生成器网络
class Generator(nn.Module):
    def __init__(self, channels=128):
        super(Generator, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels*2, 4, 2, 1),
            nn.BatchNorm2d(channels*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels*2, channels*4, 4, 2, 1),
            nn.BatchNorm2d(channels*4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels*4, channels*8, 4, 2, 1),
            nn.BatchNorm2d(channels*8),
            nn.LeakyReLU(0.2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels*8, channels*4, 4, 2, 1),
            nn.BatchNorm2d(channels*4),
            nn.ReLU(),
            nn.ConvTranspose2d(channels*4, channels*2, 4, 2, 1),
            nn.BatchNorm2d(channels*2),
            nn.ReLU(),
            nn.ConvTranspose2d(channels*2, channels, 4, 2, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.ConvTranspose2d(channels, 3, 4, 2, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train(data_root, save_dir, num_epochs=200, batch_size=1, lr=2e-4,resume_path=None):
    set_seed()
    
    # 严格的内存管理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # 限制GPU内存使用
        torch.cuda.set_per_process_memory_fraction(0.7)  # 只使用70%的GPU内存
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建数据加载器
    monet_dataset = MonetPhotoDataset(data_root, domain='A')
    photo_dataset = MonetPhotoDataset(data_root, domain='B')
    print(f"Monet图像数量: {len(monet_dataset)}")
    print(f"Photo图像数量: {len(photo_dataset)}")
    
    monet_loader = DataLoader(monet_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=1, pin_memory=True, drop_last=True)
    photo_loader = DataLoader(photo_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=1, pin_memory=True, drop_last=True)
    
    # 创建生成器 (减小通道数)
    generator = Generator(channels=128).to(device)
    print(f"生成器参数数量: {sum(p.numel() for p in generator.parameters())}")
    
    optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    criterion = nn.L1Loss()

    start_epoch = 0
    if resume_path is not None and os.path.exists(resume_path):
        print(f"[INFO] 加载已有的checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)

        # 如果是保存的字典，就按关键字加载
        if 'model_state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 如果直接保存的是 model.state_dict()，也可以直接这样:
            generator.load_state_dict(checkpoint)

        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1

        print(f"[INFO] 从第 {start_epoch} 轮继续训练。")
    else:
        print("[INFO] 未指定resume_path或文件不存在，开始新训练。")

    print("开始预训练...")
    print(f"总轮数: {num_epochs}, 批次大小: {batch_size}, 初始学习率: {lr}")
    
    best_loss = float('inf')
    
    try:
        # 训练循环
        for epoch in range(num_epochs):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # 每个epoch开始时清理缓存
            
            generator.train()
            total_loss = 0
            monet_batch_loss = 0
            photo_batch_loss = 0
            
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            print("训练 Monet 域...")
            
            # 训练Monet域
            for i, (masked_imgs, real_imgs, masks) in enumerate(monet_loader):
                masked_imgs = masked_imgs.to(device)
                real_imgs = real_imgs.to(device)
                masks = masks.to(device)
                
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():  # 使用混合精度训练
                    generated_imgs = generator(masked_imgs)
                    loss = criterion(generated_imgs * (1 - masks), real_imgs * (1 - masks))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                optimizer.step()
                
                monet_batch_loss += loss.item()
                
                # 释放不需要的张量
                del generated_imgs, loss
                torch.cuda.empty_cache()
                
                if (i + 1) % 10 == 0:
                    print(f"  Batch [{i+1}/{len(monet_loader)}], Loss: {monet_batch_loss/10:.4f}")
                    monet_batch_loss = 0
            
            print("训练 Photo 域...")
            # 训练Photo域
            for i, (masked_imgs, real_imgs, masks) in enumerate(photo_loader):
                masked_imgs = masked_imgs.to(device)
                real_imgs = real_imgs.to(device)
                masks = masks.to(device)
                
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    generated_imgs = generator(masked_imgs)
                    loss = criterion(generated_imgs * (1 - masks), real_imgs * (1 - masks))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                optimizer.step()
                
                photo_batch_loss += loss.item()
                
                # 释放不需要的张量
                del generated_imgs, loss
                torch.cuda.empty_cache()
                
                if (i + 1) % 10 == 0:
                    print(f"  Batch [{i+1}/{len(photo_loader)}], Loss: {photo_batch_loss/10:.4f}")
                    photo_batch_loss = 0
            
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # 保存检查点
            if (epoch + 1) % 50 == 0:
                save_path = os.path.join(save_dir, f'generator_pretrain_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': generator.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': total_loss,
                }, save_path)
                print(f"保存检查点: Epoch {epoch+1}")
                
            print(f"Epoch [{epoch+1}/{num_epochs}] 完成, 当前学习率: {current_lr:.6f}")
            
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        if torch.cuda.is_available():
            print(f"当前GPU内存使用情况: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
            print(f"当前GPU内存缓存: {torch.cuda.memory_reserved()/1024**2:.1f}MB")
    
    finally:
        # 清理资源
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)

    data_root = os.path.join(project_dir, "uvcgan2-main", "data", "monet2photo")
    save_dir = os.path.join(current_dir, "models")

    print("UVCGAN2 预训练开始")
    print(f"当前目录: {current_dir}")
    print(f"项目根目录: {project_dir}")
    print(f"数据集路径: {data_root}")
    print(f"模型保存路径: {save_dir}")

    # 如果想手动指定 resume_path，可以这样：
    resume_path = os.path.join(save_dir, "generator_pretrain_epoch_350.pth")

    train(data_root=data_root,
          save_dir=save_dir,
          num_epochs=500,  # 比如你想再训练到500轮
          batch_size=1,
          lr=2e-4,
          resume_path=resume_path)  # <-- 传进去


if __name__ == "__main__":
    main()