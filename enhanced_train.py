import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import itertools
from pathlib import Path
import random

from enhanced_generator import EnhancedGenerator, EnhancedDiscriminator
from pretrain import MonetPhotoDataset, set_seed

class EnhancedCycleGAN:
    def __init__(self, pretrained_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 进一步减小模型大小
        self.G_AB = EnhancedGenerator(channels=16, num_transformer_blocks=1).to(self.device)  # Monet to Photo
        self.G_BA = EnhancedGenerator(channels=16, num_transformer_blocks=1).to(self.device)  # Photo to Monet
        self.D_A = EnhancedDiscriminator(channels=16).to(self.device)  # Monet 判别器
        self.D_B = EnhancedDiscriminator(channels=16).to(self.device)  # Photo 判别器
        
        # 启用梯度检查点以节省内存
        self.G_AB.gradient_checkpointing_enable()
        self.G_BA.gradient_checkpointing_enable()
        
        # 加载预训练权重
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"加载预训练模型: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=self.device)
            # 只加载encoder部分的权重
            self.G_AB.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.G_BA.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # 创建优化器，降低学习率以提高稳定性
        self.g_optimizer = optim.Adam(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()),
            lr=5e-5, betas=(0.5, 0.999)
        )
        self.d_optimizer = optim.Adam(
            itertools.chain(self.D_A.parameters(), self.D_B.parameters()),
            lr=2e-4, betas=(0.5, 0.999)
        )
        
        # 使用混合精度训练
        self.scaler = torch.cuda.amp.GradScaler()
        
        # 损失函数
        self.criterion_gan = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        self.criterion_structure = nn.L1Loss()
        
        # 调整损失权重
        self.lambda_cycle = 10.0
        self.lambda_identity = 2.0  # 降低身份损失权重
        self.lambda_structure = 0.5  # 降低结构损失权重
    
    def train_step(self, real_A, real_B):
        # 使用混合精度训练
        with torch.cuda.amp.autocast():
            # 生成假图像
            fake_B = self.G_AB(real_A)
            fake_A = self.G_BA(real_B)
            
            # 训练判别器
            self.d_optimizer.zero_grad(set_to_none=True)
            
            # 真实图像的判别器损失
            real_A_score, _ = self.D_A(real_A)
            real_B_score, _ = self.D_B(real_B)
            d_real_loss = (self.criterion_gan(real_A_score, torch.ones_like(real_A_score)) +
                          self.criterion_gan(real_B_score, torch.ones_like(real_B_score))) * 0.5
            
            # 生成图像的判别器损失
            fake_A_score, _ = self.D_A(fake_A.detach())
            fake_B_score, _ = self.D_B(fake_B.detach())
            d_fake_loss = (self.criterion_gan(fake_A_score, torch.zeros_like(fake_A_score)) +
                          self.criterion_gan(fake_B_score, torch.zeros_like(fake_B_score))) * 0.5
            
            d_loss = d_real_loss + d_fake_loss
        
        # 反向传播判别器损失
        self.scaler.scale(d_loss).backward()
        self.scaler.step(self.d_optimizer)
        
        # 训练生成器
        with torch.cuda.amp.autocast():
            self.g_optimizer.zero_grad(set_to_none=True)
            
            # 身份损失
            idt_A = self.G_BA(real_A)
            idt_B = self.G_AB(real_B)
            identity_loss = (self.criterion_identity(idt_A, real_A) +
                           self.criterion_identity(idt_B, real_B)) * self.lambda_identity
            
            # GAN损失
            fake_A_score, _ = self.D_A(fake_A)
            fake_B_score, _ = self.D_B(fake_B)
            g_loss = (self.criterion_gan(fake_A_score, torch.ones_like(fake_A_score)) +
                     self.criterion_gan(fake_B_score, torch.ones_like(fake_B_score)))
            
            # 循环一致性损失
            recon_A = self.G_BA(fake_B)
            recon_B = self.G_AB(fake_A)
            cycle_loss = (self.criterion_cycle(recon_A, real_A) +
                         self.criterion_cycle(recon_B, real_B)) * self.lambda_cycle
            
            # 结构保持损失
            _, real_A_struct = self.D_A(real_A)
            _, fake_A_struct = self.D_A(fake_A)
            _, real_B_struct = self.D_B(real_B)
            _, fake_B_struct = self.D_B(fake_B)
            structure_loss = (self.criterion_structure(real_A_struct, fake_A_struct) +
                            self.criterion_structure(real_B_struct, fake_B_struct)) * self.lambda_structure
            
            # 总生成器损失
            total_g_loss = g_loss + cycle_loss + identity_loss + structure_loss
        
        # 反向传播生成器损失
        self.scaler.scale(total_g_loss).backward()
        self.scaler.step(self.g_optimizer)
        self.scaler.update()
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'cycle_loss': cycle_loss.item(),
            'identity_loss': identity_loss.item(),
            'structure_loss': structure_loss.item()
        }
    
    def save_models(self, save_dir, epoch):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 分别保存模型以减少内存使用
        torch.save({
            'epoch': epoch,
            'G_AB_state_dict': self.G_AB.state_dict(),
        }, save_path / f'G_AB_epoch_{epoch}.pth')
        
        torch.save({
            'epoch': epoch,
            'G_BA_state_dict': self.G_BA.state_dict(),
        }, save_path / f'G_BA_epoch_{epoch}.pth')
        
        torch.save({
            'epoch': epoch,
            'D_A_state_dict': self.D_A.state_dict(),
            'D_B_state_dict': self.D_B.state_dict(),
        }, save_path / f'discriminators_epoch_{epoch}.pth')

def train(data_root, save_dir, pretrained_path=None, num_epochs=200, batch_size=1):
    # 设置内存优化选项
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 限制GPU内存使用为30%
        torch.cuda.set_per_process_memory_fraction(0.3)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    # 创建数据加载器
    monet_dataset = MonetPhotoDataset(data_root, domain='A')
    photo_dataset = MonetPhotoDataset(data_root, domain='B')
    
    monet_loader = DataLoader(monet_dataset, batch_size=1, shuffle=True,
                            num_workers=0, pin_memory=False)
    photo_loader = DataLoader(photo_dataset, batch_size=1, shuffle=True,
                            num_workers=0, pin_memory=False)
    
    # 创建模型
    model = EnhancedCycleGAN(pretrained_path)
    
    print("开始训练增强版CycleGAN...")
    print(f"数据集路径: {data_root}")
    print(f"保存路径: {save_dir}")
    print(f"使用设备: {model.device}")
    print(f"批次大小: {batch_size}")
    
    try:
        for epoch in range(num_epochs):
            for i, (monet_batch, photo_batch) in enumerate(zip(monet_loader, photo_loader)):
                real_A = monet_batch[0].to(model.device)
                real_B = photo_batch[0].to(model.device)
                
                losses = model.train_step(real_A, real_B)
                
                if (i + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(monet_loader)}]")
                    for k, v in losses.items():
                        print(f"{k}: {v:.4f}")
                    
                    if torch.cuda.is_available():
                        print(f"GPU内存使用: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
                        print(f"GPU内存缓存: {torch.cuda.memory_reserved()/1024**2:.1f}MB")
                    print()
            
            # 每20个epoch保存一次模型
            if (epoch + 1) % 20 == 0:
                model.save_models(save_dir, epoch + 1)
                print(f"保存模型检查点: Epoch {epoch+1}")
    
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        if torch.cuda.is_available():
            print(f"GPU内存使用: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
            print(f"GPU内存缓存: {torch.cuda.memory_reserved()/1024**2:.1f}MB")

if __name__ == "__main__":
    # 使用绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    
    data_root = os.path.join(project_dir, "uvcgan2-main", "data", "monet2photo")
    save_dir = os.path.join(current_dir, "models")
    pretrained_path = os.path.join(save_dir, "generator_pretrain_epoch_200.pth")
    
    print("增强版 UVCGAN2 CycleGAN 训练开始")
    print(f"当前目录: {current_dir}")
    print(f"项目根目录: {project_dir}")
    print(f"数据集路径: {data_root}")
    print(f"模型保存路径: {save_dir}")
    print(f"预训练模型路径: {pretrained_path}")
    
    # 检查路径是否存在
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"数据集路径不存在: {data_root}")
    
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"预训练模型不存在: {pretrained_path}")
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    print(f"创建/确认保存目录: {save_dir}")
    
    train(data_root, save_dir, pretrained_path) 