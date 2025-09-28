import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import inception_v3, Inception_V3_Weights
import torchvision.transforms as transforms
import numpy as np
from scipy import linalg
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from pretrain import MonetPhotoDataset
from enhanced_train import EnhancedCycleGAN
import cv2

class InceptionV3Feature:
    def __init__(self):
        # 使用新的API加载模型
        weights = Inception_V3_Weights.DEFAULT
        self.model = inception_v3(weights=weights).cuda()
        self.model.eval()
        self.model.fc = nn.Identity()  # 移除全连接层
        self.model.dropout = nn.Identity()  # 移除dropout层
        
        # 使用权重中预定义的transforms
        self.preprocess = weights.transforms()
    
    @torch.no_grad()  # 使用装饰器来避免gradient warning
    def get_features(self, x):
        # 调整图像大小并进行预处理
        x = torch.nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = self.preprocess(x)
        features = self.model(x)
        return features


def calculate_fid(real_features, fake_features):
    # 计算均值
    mu1, sigma1 = real_features.mean(0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(0), np.cov(fake_features, rowvar=False)
    
    # 计算两个分布之间的距离
    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def process_image(img_tensor):
    """处理图像张量，改善视觉质量"""
    # 确保输入是在CPU上的张量
    img = img_tensor.detach().cpu()
    
    # 将图像从[-1, 1]范围转换到[0, 1]范围
    img = img * 0.5 + 0.5
    
    # 调整通道顺序 (C,H,W) -> (H,W,C)
    img = img.permute(1, 2, 0)
    
    # 转换为numpy数组并确保值在[0,1]范围内
    img = img.numpy()
    
    # 应用gamma校正来增强对比度
    gamma = 1.1
    img = np.power(img, gamma)
    
    # 应用直方图均衡化来增强细节
    img_yuv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB) / 255.0
    
    # 确保值在[0,1]范围内
    img = np.clip(img, 0, 1)
    
    return img

@torch.no_grad()  # 使用装饰器来避免gradient warning
def run_model(model_path, data_root, save_dir, num_test=100):
    print("开始测试增强版CycleGAN...")
    print(f"数据集路径: {data_root}")
    print(f"模型路径: {model_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建保存目录
    os.makedirs(os.path.join(save_dir, 'test_results'), exist_ok=True)
    
    # 加载数据集
    test_monet = MonetPhotoDataset(data_root, domain='A', split='test')
    test_photo = MonetPhotoDataset(data_root, domain='B', split='test')
    
    test_monet_loader = DataLoader(test_monet, batch_size=1, shuffle=False)
    test_photo_loader = DataLoader(test_photo, batch_size=1, shuffle=False)
    
    # 加载模型
    model = EnhancedCycleGAN()
    # 将生成器移动到设备
    model.G_AB = model.G_AB.to(device)
    model.G_BA = model.G_BA.to(device)
    model.G_AB.eval()
    model.G_BA.eval()
    
    # 加载生成器权重
    g_ab_checkpoint = torch.load(os.path.join(model_path, 'G_AB_epoch_200.pth'), map_location=device)
    g_ba_checkpoint = torch.load(os.path.join(model_path, 'G_BA_epoch_200.pth'), map_location=device)
    
    model.G_AB.load_state_dict(g_ab_checkpoint['G_AB_state_dict'])
    model.G_BA.load_state_dict(g_ba_checkpoint['G_BA_state_dict'])
    
    # 初始化Inception模型
    inception = InceptionV3Feature()
    
    # 存储特征
    real_features_monet = []
    fake_features_monet = []
    real_features_photo = []
    fake_features_photo = []
    
    print("生成测试图像...")
    # Monet -> Photo
    for i, data in enumerate(tqdm(test_monet_loader)):
        if i >= num_test:
            break
            
        real_monet = data[0].to(device)
        fake_photo = model.G_AB(real_monet)
        
        # 保存图像
        save_path = os.path.join(save_dir, 'test_results', f'monet2photo_{i}.png')
        plt.figure(figsize=(10, 5))
        
        # 处理真实图像
        plt.subplot(1, 2, 1)
        real_img = process_image(real_monet[0])
        plt.imshow(real_img)
        plt.title('Real Monet', fontsize=12)
        plt.axis('off')
        
        # 处理生成图像
        plt.subplot(1, 2, 2)
        fake_img = process_image(fake_photo[0])
        plt.imshow(fake_img)
        plt.title('Generated Photo', fontsize=12)
        plt.axis('off')
        
        # 调整子图之间的间距
        plt.tight_layout(pad=2.0)
        
        # 使用高质量设置保存图像
        plt.savefig(save_path, 
                   dpi=300,
                   bbox_inches='tight',
                   pad_inches=0.1,
                   facecolor='white',
                   edgecolor='none',
                   format='png',
                   transparent=False)
        plt.close()
        
        # 提取特征
        real_features_monet.append(inception.get_features(real_monet).cpu().numpy())
        fake_features_photo.append(inception.get_features(fake_photo).cpu().numpy())
    
    # Photo -> Monet
    for i, data in enumerate(tqdm(test_photo_loader)):
        if i >= num_test:
            break
            
        real_photo = data[0].to(device)
        fake_monet = model.G_BA(real_photo)
        
        # 保存图像
        save_path = os.path.join(save_dir, 'test_results', f'photo2monet_{i}.png')
        plt.figure(figsize=(10, 5))
        
        # 处理真实图像
        plt.subplot(1, 2, 1)
        real_img = process_image(real_photo[0])
        plt.imshow(real_img)
        plt.title('Real Photo', fontsize=12)
        plt.axis('off')
        
        # 处理生成图像
        plt.subplot(1, 2, 2)
        fake_img = process_image(fake_monet[0])
        plt.imshow(fake_img)
        plt.title('Generated Monet', fontsize=12)
        plt.axis('off')
        
        # 调整子图之间的间距
        plt.tight_layout(pad=2.0)
        
        # 使用高质量设置保存图像
        plt.savefig(save_path, 
                   dpi=300,
                   bbox_inches='tight',
                   pad_inches=0.1,
                   facecolor='white',
                   edgecolor='none',
                   format='png',
                   transparent=False)
        plt.close()
        
        # 提取特征
        real_features_photo.append(inception.get_features(real_photo).cpu().numpy())
        fake_features_monet.append(inception.get_features(fake_monet).cpu().numpy())

    # 计算FID分数
    real_features_monet = np.concatenate(real_features_monet)
    fake_features_photo = np.concatenate(fake_features_photo)
    real_features_photo = np.concatenate(real_features_photo)
    fake_features_monet = np.concatenate(fake_features_monet)
    
    fid_monet2photo = calculate_fid(real_features_photo, fake_features_photo)
    fid_photo2monet = calculate_fid(real_features_monet, fake_features_monet)
    
    # 保存测试结果
    with open(os.path.join(save_dir, 'test_results', 'test_results.txt'), 'w') as f:
        f.write(f"测试样本数量: {num_test}\n")
        f.write(f"Monet -> Photo FID: {fid_monet2photo:.4f}\n")
        f.write(f"Photo -> Monet FID: {fid_photo2monet:.4f}\n")
        f.write(f"平均 FID: {(fid_monet2photo + fid_photo2monet) / 2:.4f}\n")
    
    print("测试完成！结果已保存到test_results目录")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    
    data_root = os.path.join(project_dir, "uvcgan2-main", "data", "monet2photo")
    save_dir = os.path.join(current_dir, "models")
    
    run_model(save_dir, data_root, save_dir)