import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import argparse
import time
from tqdm import tqdm
from pathlib import Path
import glob
from scipy.ndimage import gaussian_filter

# 导入必要的模型和工具
# 请确保以下文件在项目目录中
from enhanced_generator import EnhancedGenerator

class Generator(nn.Module):
    """定义与GUI应用程序相同的生成器模型"""
    def __init__(self, channels=64):
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

def load_models(device):
    """加载所需的各种模型"""
    models = {}
    
    # 加载CycleGAN模型
    try:
        cyclegan_path = "models/cyclegan_epoch_200.pth"
        if os.path.exists(cyclegan_path):
            # 创建CycleGAN模型
            models['cyclegan_AB'] = Generator(channels=64).to(device)
            models['cyclegan_BA'] = Generator(channels=64).to(device)
            
            # 加载模型权重
            checkpoint = torch.load(cyclegan_path, map_location=device)
            
            if "G_AB_state_dict" in checkpoint and "G_BA_state_dict" in checkpoint:
                models['cyclegan_AB'].load_state_dict(checkpoint["G_AB_state_dict"])
                models['cyclegan_BA'].load_state_dict(checkpoint["G_BA_state_dict"])
            elif "G_A" in checkpoint and "G_B" in checkpoint:
                models['cyclegan_AB'].load_state_dict(checkpoint["G_A"])
                models['cyclegan_BA'].load_state_dict(checkpoint["G_B"])
            
            models['cyclegan_AB'].eval()
            models['cyclegan_BA'].eval()
            print("成功加载CycleGAN模型")
    except Exception as e:
        print(f"加载CycleGAN模型出错: {str(e)}")
    
    # 加载增强型局部风格模型
    try:
        # 加载AB模型（莫奈 -> 照片）
        model_path_AB = "models/G_AB_epoch_200.pth"
        if os.path.exists(model_path_AB):
            checkpoint = torch.load(model_path_AB, map_location=device)
            if "G_AB_state_dict" in checkpoint:
                models['enhanced_AB'] = EnhancedGenerator(channels=16, num_transformer_blocks=1).to(device)
                models['enhanced_AB'].load_state_dict(checkpoint["G_AB_state_dict"])
                models['enhanced_AB'].eval()
                print("成功加载增强型莫奈->照片模型")
            else:
                # 尝试直接加载
                models['enhanced_AB'] = EnhancedGenerator(channels=16, num_transformer_blocks=1).to(device)
                models['enhanced_AB'].load_state_dict(checkpoint)
                models['enhanced_AB'].eval()
                print("成功加载增强型莫奈->照片模型")
        
        # 加载BA模型（照片 -> 莫奈）
        model_path_BA = "models/G_BA_epoch_200.pth"
        if os.path.exists(model_path_BA):
            checkpoint = torch.load(model_path_BA, map_location=device)
            if "G_BA_state_dict" in checkpoint:
                models['enhanced_BA'] = EnhancedGenerator(channels=16, num_transformer_blocks=1).to(device)
                models['enhanced_BA'].load_state_dict(checkpoint["G_BA_state_dict"])
                models['enhanced_BA'].eval()
                print("成功加载增强型照片->莫奈模型")
            else:
                # 尝试直接加载
                models['enhanced_BA'] = EnhancedGenerator(channels=16, num_transformer_blocks=1).to(device)
                models['enhanced_BA'].load_state_dict(checkpoint)
                models['enhanced_BA'].eval()
                print("成功加载增强型照片->莫奈模型")
    except Exception as e:
        print(f"加载增强型模型出错: {str(e)}")
    
    return models

def detect_sky(img, threshold=0.7):
    """检测图像中的天空区域"""
    if isinstance(img, Image.Image):
        img_np = np.array(img)
    else:
        img_np = img.copy()
    
    # 转换为HSV颜色空间
    img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    
    # 天空通常有高亮度和低饱和度
    h, s, v = cv2.split(img_hsv)
    
    # 创建掩码：高亮度且低饱和度的区域可能是天空
    bright_mask = (v > 150)
    low_sat_mask = (s < 100)
    
    # 结合掩码
    sky_mask = bright_mask & low_sat_mask
    
    # 如果天空像素占比大于阈值，认为存在天空
    if np.sum(sky_mask) / (img_np.shape[0] * img_np.shape[1]) > threshold:
        return True, sky_mask
    
    return False, sky_mask

def smooth_transitions(img, mask, radius=5, iterations=2):
    """平滑过渡区域"""
    # 确保输入是numpy数组
    if isinstance(img, Image.Image):
        img_np = np.array(img)
    else:
        img_np = img.copy()
    
    # 对掩码进行膨胀和侵蚀以找到边界
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=iterations)
    boundary = dilated - eroded
    
    # 在边界区域应用高斯模糊
    blurred = cv2.GaussianBlur(img_np, (radius*2+1, radius*2+1), 0)
    
    # 在边界处混合原始图像和模糊图像
    result = img_np.copy()
    boundary_coords = np.where(boundary > 0)
    result[boundary_coords] = (img_np[boundary_coords] * 0.5 + blurred[boundary_coords] * 0.5)
    
    return result

def process_cyclegan(model, img_path, output_dir, device, direction='photo2monet'):
    """使用CycleGAN处理单个图像"""
    try:
        # 读取图像
        original_img = Image.open(img_path).convert('RGB')
        width, height = original_img.size
        
        # 调整图像大小以适应模型
        target_size = (256, 256)
        
        # 保持原始宽高比
        if width > height:
            new_width = target_size[0]
            new_height = int(height * (target_size[0] / width))
        else:
            new_height = target_size[1]
            new_width = int(width * (target_size[1] / height))
        
        resized_img = original_img.resize((new_width, new_height), Image.LANCZOS)
        
        # 创建画布
        canvas = Image.new('RGB', target_size, (255, 255, 255))
        offset_x = (target_size[0] - new_width) // 2
        offset_y = (target_size[1] - new_height) // 2
        canvas.paste(resized_img, (offset_x, offset_y))
        
        # 准备输入张量
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        input_tensor = transform(canvas).unsqueeze(0).to(device)
        
        # 处理图像
        with torch.no_grad():
            output = model(input_tensor)
        
        # 后处理
        output = (output + 1.0) / 2.0
        output = torch.clamp(output, 0, 1)
        output_np = output.cpu().squeeze(0).permute(1, 2, 0).numpy()
        output_img = Image.fromarray((output_np * 255).astype(np.uint8))
        
        # 裁剪回原始比例
        if width != height:
            aspect_ratio = width / height
            if aspect_ratio > 1:  # 宽大于高
                crop_width = target_size[0]
                crop_height = int(target_size[0] / aspect_ratio)
            else:  # 高大于宽
                crop_height = target_size[1]
                crop_width = int(target_size[1] * aspect_ratio)
            
            left = (target_size[0] - crop_width) // 2
            top = (target_size[1] - crop_height) // 2
            right = left + crop_width
            bottom = top + crop_height
            
            output_img = output_img.crop((left, top, right, bottom))
        
        # 调整回原始大小
        if width * height <= 1024 * 1024:  # 仅当原始图像不太大时才调整回原始大小
            output_img = output_img.resize((width, height), Image.LANCZOS)
        
        # 创建特定的输出子目录
        cyclegan_dir = os.path.join(output_dir, f"cyclegan_{direction}")
        os.makedirs(cyclegan_dir, exist_ok=True)
        
        # 保存结果
        img_name = os.path.basename(img_path)
        output_path = os.path.join(cyclegan_dir, img_name)
        output_img.save(output_path)
        
        return output_path
    
    except Exception as e:
        print(f"处理图像 {img_path} 时出错: {str(e)}")
        return None

def process_local_style(model, img_path, output_dir, device, mode='enhanced', 
                        strength=0.8, detail=0.7, enhance_colors=True, smooth=True, direction='photo2monet'):
    """使用局部风格模式处理单个图像"""
    try:
        # 加载和预处理图像
        original_img = Image.open(img_path).convert('RGB')
        width, height = original_img.size
        original_aspect_ratio = width / height
        
        # 调整图像大小以适应模型
        target_size = (256, 256)
        
        # 保持原始宽高比
        if width > height:
            new_width = target_size[0]
            new_height = int(height * (target_size[0] / width))
        else:
            new_height = target_size[1]
            new_width = int(width * (target_size[1] / height))
        
        resized_img = original_img.resize((new_width, new_height), Image.LANCZOS)
        
        # 创建白色背景画布并粘贴调整后的图像
        canvas = Image.new('RGB', target_size, (255, 255, 255))
        offset_x = (target_size[0] - new_width) // 2
        offset_y = (target_size[1] - new_height) // 2
        canvas.paste(resized_img, (offset_x, offset_y))
        
        # 保存原始画布尺寸用于后续处理
        canvas_width, canvas_height = canvas.size
        
        # 准备输入张量
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        input_tensor = transform(canvas).unsqueeze(0).to(device)
        
        # 处理图像
        with torch.no_grad():
            output = model(input_tensor)
        
        # 后处理
        output = (output + 1.0) / 2.0
        output = torch.clamp(output, 0, 1)
        output_np = output.cpu().squeeze(0).permute(1, 2, 0).numpy()
        styled_img = Image.fromarray((output_np * 255).astype(np.uint8))
        
        # 根据模式处理结果
        if mode == 'simple':
            # 简单模式：直接使用样式强度混合
            original_np = np.array(canvas)
            styled_np = np.array(styled_img)
            result_np = original_np * (1 - strength) + styled_np * strength
            result_np = np.clip(result_np, 0, 255).astype(np.uint8)
            output_img = Image.fromarray(result_np)
        
        elif mode == 'enhanced':
            # 增强模式：处理天空区域，保留细节
            original_np = np.array(canvas)
            styled_np = np.array(styled_img)
            
            # 检测天空区域
            has_sky, sky_mask = detect_sky(original_np)
            
            # 创建一个简单的边缘检测掩码
            gray = cv2.cvtColor(original_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_mask = edges > 0
            
            # 创建细节保留掩码（基于边缘和用户设置的细节级别）
            detail_strength = detail  # 存储细节强度值，稍后使用
            detail_mask = gaussian_filter(edge_mask.astype(float), sigma=2) > 0.1
            
            # 组合所有掩码
            weight = np.ones_like(gray, dtype=float) * strength
            
            # 如果检测到天空，则在天空区域应用更强的风格化
            if has_sky:
                weight[sky_mask] = min(strength + 0.2, 1.0)
            
            # 在边缘和细节区域应用较少的风格化
            detail_weight = max(strength - 0.3 * detail_strength, 0.0)
            weight[detail_mask] = detail_weight
            
            # 应用风格转换，按像素混合
            weight = weight[:, :, np.newaxis]  # 扩展维度以适应RGB通道
            result_np = original_np * (1 - weight) + styled_np * weight
            
            # 可选：增强颜色
            if enhance_colors:
                result_np = cv2.convertScaleAbs(result_np, alpha=1.1, beta=5)
            
            # 可选：平滑过渡
            if smooth:
                result_np = smooth_transitions(result_np, detail_mask, radius=3)
            
            result_np = np.clip(result_np, 0, 255).astype(np.uint8)
            output_img = Image.fromarray(result_np)
        
        elif mode == 'advanced':
            # 高级模式：结合增强模式和额外的处理
            original_np = np.array(canvas)
            styled_np = np.array(styled_img)
            
            # 检测天空区域
            has_sky, sky_mask = detect_sky(original_np)
            
            # 创建边缘检测掩码
            gray = cv2.cvtColor(original_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_mask = edges > 0
            
            # 创建细节保留掩码
            detail_strength = detail  # 存储细节强度值，稍后使用
            detail_mask = gaussian_filter(edge_mask.astype(float), sigma=2) > 0.1
            
            # 创建基于色彩分割的区域掩码
            segmented = cv2.pyrMeanShiftFiltering(original_np, sp=20, sr=40)
            
            # 组合所有掩码
            weight = np.ones_like(gray, dtype=float) * strength
            
            # 如果检测到天空，则在天空区域应用更强的风格化
            if has_sky:
                weight[sky_mask] = min(strength + 0.2, 1.0)
            
            # 在边缘和细节区域应用较少的风格化
            detail_weight = max(strength - 0.4 * detail_strength, 0.0)
            weight[detail_mask] = detail_weight
            
            # 应用风格转换，按像素混合
            weight = weight[:, :, np.newaxis]  # 扩展维度以适应RGB通道
            result_np = original_np * (1 - weight) + styled_np * weight
            
            # 自适应颜色增强
            result_yuv = cv2.cvtColor(result_np, cv2.COLOR_RGB2YUV)
            result_yuv[:,:,0] = cv2.equalizeHist(result_yuv[:,:,0])
            result_np = cv2.cvtColor(result_yuv, cv2.COLOR_YUV2RGB)
            
            # 应用引导滤波以平滑色块
            result_np = cv2.ximgproc.guidedFilter(guide=result_np, src=result_np, radius=4, eps=1e-4)
            
            # 强化边缘
            if edge_mask.any():
                edge_enhanced = np.copy(result_np)
                edge_enhanced = cv2.addWeighted(edge_enhanced, 1.5, edge_enhanced, -0.5, 0)
                edge_mask_3d = np.stack([edge_mask] * 3, axis=2)
                result_np = np.where(edge_mask_3d, edge_enhanced, result_np)
            
            result_np = np.clip(result_np, 0, 255).astype(np.uint8)
            output_img = Image.fromarray(result_np)
        
        else:
            # 默认情况：使用样式图像
            output_img = styled_img
        
        # 裁剪回原始比例
        if original_aspect_ratio != 1.0:
            # 计算基于原始长宽比的裁剪尺寸
            if original_aspect_ratio > 1:  # 宽大于高
                crop_width = canvas_width
                crop_height = int(canvas_width / original_aspect_ratio)
            else:  # 高大于宽
                crop_height = canvas_height
                crop_width = int(canvas_height * original_aspect_ratio)
            
            # 确保裁剪尺寸不超过画布
            crop_width = min(crop_width, canvas_width)
            crop_height = min(crop_height, canvas_height)
            
            # 计算裁剪区域的左上角和右下角坐标
            left = (canvas_width - crop_width) // 2
            top = (canvas_height - crop_height) // 2
            right = left + crop_width
            bottom = top + crop_height
            
            # 安全检查，确保裁剪区域是有效的
            if left >= 0 and top >= 0 and right <= canvas_width and bottom <= canvas_height:
                output_img = output_img.crop((left, top, right, bottom))
        
        # 调整回原始大小
        if width * height <= 1024 * 1024:  # 仅当原始图像不太大时才调整回原始大小
            output_img = output_img.resize((width, height), Image.LANCZOS)
        
        # 创建特定的输出子目录
        local_style_dir = os.path.join(output_dir, f"local_style_{mode}_{direction}")
        os.makedirs(local_style_dir, exist_ok=True)
        
        # 保存结果
        img_name = os.path.basename(img_path)
        output_path = os.path.join(local_style_dir, img_name)
        output_img.save(output_path)
        
        return output_path
    
    except Exception as e:
        print(f"处理图像 {img_path} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def batch_process(models, input_dir, output_dir, mode='cyclegan', 
                 local_style_mode='enhanced', direction='photo2monet',
                 strength=0.8, detail=0.7, enhance_colors=True, smooth=True):
    """批量处理指定目录中的所有图像"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 获取输入图像列表
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    print(f"找到 {len(image_files)} 个图像")
    if len(image_files) == 0:
        print(f"错误: 在 {input_dir} 中没有找到图像")
        return
    
    # 根据模式选择处理方法
    if mode == 'cyclegan':
        if direction == 'photo2monet':
            if 'cyclegan_BA' not in models:
                print("错误: 缺少CycleGAN照片到莫奈模型")
                return
            model = models['cyclegan_BA']
        else:  # monet2photo
            if 'cyclegan_AB' not in models:
                print("错误: 缺少CycleGAN莫奈到照片模型")
                return
            model = models['cyclegan_AB']
        
        # 批量处理图像
        print(f"开始使用CycleGAN模式处理 {len(image_files)} 个图像（{direction}）")
        start_time = time.time()
        
        successful = 0
        for img_path in tqdm(image_files):
            result_path = process_cyclegan(model, img_path, output_dir, device, direction)
            if result_path:
                successful += 1
        
        end_time = time.time()
        print(f"处理完成! 成功处理 {successful}/{len(image_files)} 个图像")
        print(f"总耗时: {end_time - start_time:.2f}秒, 平均每张: {(end_time - start_time) / len(image_files):.2f}秒")
    
    elif mode == 'local_style':
        if direction == 'photo2monet':
            if 'enhanced_BA' not in models:
                print("错误: 缺少增强型照片到莫奈模型")
                return
            model = models['enhanced_BA']
        else:  # monet2photo
            if 'enhanced_AB' not in models:
                print("错误: 缺少增强型莫奈到照片模型")
                return
            model = models['enhanced_AB']
        
        # 批量处理图像
        print(f"开始使用局部风格模式处理 {len(image_files)} 个图像（{direction}，{local_style_mode}模式）")
        start_time = time.time()
        
        successful = 0
        for img_path in tqdm(image_files):
            result_path = process_local_style(
                model, img_path, output_dir, device, 
                mode=local_style_mode, strength=strength, 
                detail=detail, enhance_colors=enhance_colors, 
                smooth=smooth, direction=direction
            )
            if result_path:
                successful += 1
        
        end_time = time.time()
        print(f"处理完成! 成功处理 {successful}/{len(image_files)} 个图像")
        print(f"总耗时: {end_time - start_time:.2f}秒, 平均每张: {(end_time - start_time) / len(image_files):.2f}秒")
    
    else:
        print(f"错误: 不支持的处理模式 {mode}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='批量处理图像，应用风格转换')
    parser.add_argument('--input_dir', type=str, default='test_images', help='输入图像目录')
    parser.add_argument('--output_dir', type=str, default='output/batch', help='输出结果目录')
    parser.add_argument('--mode', type=str, choices=['cyclegan', 'local_style'], default='cyclegan', help='处理模式')
    parser.add_argument('--direction', type=str, choices=['photo2monet', 'monet2photo'], default='photo2monet', help='转换方向')
    parser.add_argument('--local_style_mode', type=str, choices=['simple', 'enhanced', 'advanced'], default='enhanced', help='局部风格模式')
    parser.add_argument('--strength', type=float, default=0.8, help='风格强度 (0-1)')
    parser.add_argument('--detail', type=float, default=0.7, help='细节保留水平 (0-1)')
    parser.add_argument('--enhance_colors', action='store_true', default=True, help='是否增强颜色')
    parser.add_argument('--no_enhance_colors', dest='enhance_colors', action='store_false', help='不增强颜色')
    parser.add_argument('--smooth', action='store_true', default=True, help='是否平滑过渡')
    parser.add_argument('--no_smooth', dest='smooth', action='store_false', help='不平滑过渡')
    
    args = parser.parse_args()
    
    # 获取设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print("加载模型...")
    models = load_models(device)
    
    # 检查是否加载了所需的模型
    if args.mode == 'cyclegan':
        if args.direction == 'photo2monet' and 'cyclegan_BA' not in models:
            print("错误: 未能加载CycleGAN照片到莫奈模型")
            return
        if args.direction == 'monet2photo' and 'cyclegan_AB' not in models:
            print("错误: 未能加载CycleGAN莫奈到照片模型")
            return
    
    if args.mode == 'local_style':
        if args.direction == 'photo2monet' and 'enhanced_BA' not in models:
            print("错误: 未能加载增强型照片到莫奈模型")
            return
        if args.direction == 'monet2photo' and 'enhanced_AB' not in models:
            print("错误: 未能加载增强型莫奈到照片模型")
            return
    
    # 执行批处理
    batch_process(
        models,
        args.input_dir,
        args.output_dir,
        mode=args.mode,
        local_style_mode=args.local_style_mode,
        direction=args.direction,
        strength=args.strength,
        detail=args.detail,
        enhance_colors=args.enhance_colors,
        smooth=args.smooth
    )

if __name__ == "__main__":
    main() 