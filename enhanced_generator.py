import torch
import torch.nn as nn
import torch.nn.functional as F
from structural_transformer import StructuralTransformerBlock

class LocalAttention(nn.Module):
    def __init__(self, channels, window_size=8):
        super().__init__()
        self.window_size = window_size
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        # 分割成局部窗口
        x = x.view(B, C, H // self.window_size, self.window_size, 
                  W // self.window_size, self.window_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(-1, C, self.window_size * self.window_size)
        
        # 计算注意力
        qkv = self.qkv(x.view(-1, C, self.window_size, self.window_size))
        q, k, v = qkv.chunk(3, dim=1)
        
        attn = (F.normalize(q, dim=1).flatten(2) @ 
                F.normalize(k, dim=1).flatten(2).transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v.flatten(2)).view(-1, C, self.window_size, self.window_size)
        x = self.proj(x)
        
        # 重组回原始形状
        x = x.view(B, H // self.window_size, W // self.window_size, 
                  C, self.window_size, self.window_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, C, H + pad_h, W + pad_w)
        
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]
        
        return x

class MultiScaleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels//4, 1),
            nn.InstanceNorm2d(channels//4),
            nn.ReLU(True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels//4, 3, padding=1, dilation=1),
            nn.InstanceNorm2d(channels//4),
            nn.ReLU(True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels//4, 3, padding=2, dilation=2),
            nn.InstanceNorm2d(channels//4),
            nn.ReLU(True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(channels, channels//4, 3, padding=4, dilation=4),
            nn.InstanceNorm2d(channels//4),
            nn.ReLU(True)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True)
        )
        
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return self.fusion(out) + x

class EnhancedGenerator(nn.Module):
    def __init__(self, channels=64, num_transformer_blocks=3):
        super().__init__()
        
        # 初始卷积层
        self.initial = nn.Sequential(
            nn.Conv2d(3, channels, 7, 1, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True)
        )
        
        # 下采样层
        self.down1 = nn.Sequential(
            nn.Conv2d(channels, channels*2, 4, 2, 1),
            nn.InstanceNorm2d(channels*2),
            nn.ReLU(True),
            LocalAttention(channels*2, window_size=4),
            MultiScaleBlock(channels*2)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(channels*2, channels*4, 4, 2, 1),
            nn.InstanceNorm2d(channels*4),
            nn.ReLU(True),
            LocalAttention(channels*4, window_size=4),
            MultiScaleBlock(channels*4)
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            StructuralTransformerBlock(dim=channels*4)
            for _ in range(num_transformer_blocks)
        ])
        
        # 上采样层
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(channels*4, channels*2, 4, 2, 1),
            nn.InstanceNorm2d(channels*2),
            nn.ReLU(True),
            LocalAttention(channels*2, window_size=4),
            MultiScaleBlock(channels*2)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(channels*2, channels, 4, 2, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            LocalAttention(channels, window_size=4),
            MultiScaleBlock(channels)
        )
        
        # 输出层
        self.output = nn.Sequential(
            nn.Conv2d(channels, 3, 7, 1, 3),
            nn.Tanh()
        )
        
        # 风格编码器
        self.style_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels*4, channels*4),
            nn.ReLU(True)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.InstanceNorm2d):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def gradient_checkpointing_enable(self):
        """启用梯度检查点"""
        self.down1.requires_grad_(True)
        self.down2.requires_grad_(True)
        self.transformer_blocks.requires_grad_(True)
        self.up1.requires_grad_(True)
        self.up2.requires_grad_(True)
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        self.use_checkpointing = True
        self.custom_forward = create_custom_forward
    
    def forward(self, x):
        # 保存原始输入用于结构提取
        orig_input = x.clone()
        
        if hasattr(self, 'use_checkpointing'):
            # 使用梯度检查点的前向传播
            x = self.initial(x)
            x = torch.utils.checkpoint.checkpoint(self.custom_forward(self.down1), x)
            x = torch.utils.checkpoint.checkpoint(self.custom_forward(self.down2), x)
            
            # 提取风格特征
            style = self.style_encoder(x)
            
            # 保存特征图的形状
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
            
            # 对每个transformer块，传入原始输入图像
            for block in self.transformer_blocks:
                x = torch.utils.checkpoint.checkpoint(
                    self.custom_forward(block),
                    x,
                    style,
                    orig_input  # 传入原始输入图像
                )
            
            x = x.transpose(1, 2).view(B, C, H, W)
            
            x = torch.utils.checkpoint.checkpoint(self.custom_forward(self.up1), x)
            x = torch.utils.checkpoint.checkpoint(self.custom_forward(self.up2), x)
            return self.output(x)
        else:
            # 常规前向传播
            x = self.initial(x)
            x = self.down1(x)
            x = self.down2(x)
            
            style = self.style_encoder(x)
            
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
            
            # 对每个transformer块，传入原始输入图像
            for block in self.transformer_blocks:
                x = block(x, style, orig_input)  # 传入原始输入图像
            
            x = x.transpose(1, 2).view(B, C, H, W)
            x = self.up1(x)
            x = self.up2(x)
            return self.output(x)

class EnhancedDiscriminator(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        
        # 主干网络
        self.main = nn.Sequential(
            # 输入层
            nn.Conv2d(3, channels, 4, 2, 1),
            nn.LeakyReLU(0.2),
            
            # 中间层
            nn.Conv2d(channels, channels*2, 4, 2, 1),
            nn.InstanceNorm2d(channels*2),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(channels*2, channels*4, 4, 2, 1),
            nn.InstanceNorm2d(channels*4),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(channels*4, channels*8, 4, 2, 1),
            nn.InstanceNorm2d(channels*8),
            nn.LeakyReLU(0.2),
        )
        
        # 批统计头
        self.batch_head = nn.Sequential(
            nn.Conv2d(channels*8, 1, 4, 1, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 结构保持头
        self.structure_head = nn.Sequential(
            nn.Conv2d(channels*8, channels*8, 3, 1, 1),
            nn.InstanceNorm2d(channels*8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels*8, 1, 4, 1, 1)
        )
        
        # 应用谱归一化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.utils.spectral_norm(m)
    
    def forward(self, x):
        features = self.main(x)
        return self.batch_head(features).squeeze(), self.structure_head(features) 