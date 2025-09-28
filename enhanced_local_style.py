import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from enhanced_generator import EnhancedGenerator
import argparse
from skimage.segmentation import slic, felzenszwalb, quickshift
from scipy.ndimage import gaussian_filter

def load_model(model_path, device, channels=16, blocks=1):
    """加载预训练模型"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # 判断模型类型
    if "G_AB_state_dict" in checkpoint:
        state_dict = checkpoint["G_AB_state_dict"]
        model_type = "G_AB"  # 莫奈转照片
    elif "G_BA_state_dict" in checkpoint:
        state_dict = checkpoint["G_BA_state_dict"]
        model_type = "G_BA"  # 照片转莫奈
    else:
        raise ValueError("未知的模型格式")
    
    print(f"模型类型: {model_type}")
    
    # 创建模型
    model = EnhancedGenerator(channels=channels, num_transformer_blocks=blocks)
    
    # 加载权重
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model, model_type

def preprocess_image(img, size=256):
    """预处理图像"""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(img).unsqueeze(0)

def standard_postprocess(tensor):
    """标准后处理"""
    tensor = (tensor.clone() + 1) / 2.0
    tensor = tensor.clamp(0, 1)
    tensor = tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
    return Image.fromarray((tensor * 255).astype('uint8'))

def get_segmentation_mask(img, method='felzenszwalb', n_segments=100, compactness=10):
    """获取图像分割掩码"""
    # 将PIL图像转换为numpy数组
    if isinstance(img, Image.Image):
        img_np = np.array(img)
    else:
        img_np = img
    
    # 应用不同的分割方法
    if method == 'slic':
        segments = slic(img_np, n_segments=n_segments, compactness=compactness)
    elif method == 'felzenszwalb':
        segments = felzenszwalb(img_np, scale=100, sigma=0.5, min_size=50)
    elif method == 'quickshift':
        segments = quickshift(img_np, kernel_size=3, max_dist=6, ratio=0.5)
    else:
        raise ValueError(f"未知的分割方法: {method}")
    
    return segments

def analyze_segments(img, segments):
    """分析图像分割，返回每个区域的特性"""
    # 将PIL图像转换为numpy数组
    if isinstance(img, Image.Image):
        img_np = np.array(img)
    else:
        img_np = img
    
    # 转换为HSV颜色空间，更好地分析颜色特性
    img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    
    segment_stats = {}
    for segment_id in np.unique(segments):
        # 创建当前区域的掩码
        mask = segments == segment_id
        
        # 获取该区域的像素
        region_rgb = img_np[mask]
        region_hsv = img_hsv[mask]
        
        # 计算区域特性
        avg_color_rgb = np.mean(region_rgb, axis=0)
        avg_color_hsv = np.mean(region_hsv, axis=0)
        std_color = np.std(region_rgb, axis=0)
        
        # 计算区域的边缘密度（使用Sobel算子）
        if mask.any():
            mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            region_img = np.zeros_like(img_np)
            region_img[mask_3d] = img_np[mask_3d]
            
            gray = cv2.cvtColor(region_img, cv2.COLOR_RGB2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_density = np.mean(np.sqrt(sobelx**2 + sobely**2))
        else:
            edge_density = 0
        
        # 存储区域统计信息
        segment_stats[segment_id] = {
            'avg_color_rgb': avg_color_rgb,
            'avg_color_hsv': avg_color_hsv,
            'std_color': std_color,
            'edge_density': edge_density,
            'size': np.sum(mask),
            'position': np.mean(np.argwhere(mask), axis=0) if mask.any() else np.array([0, 0])
        }
    
    return segment_stats

def determine_blend_ratios(segment_stats, segments, img_shape):
    """为每个区域确定最佳混合比例"""
    # 创建一个与图像形状相同的混合比例图
    blend_map = np.zeros(img_shape[:2], dtype=np.float32)
    
    # 图像中心点
    center_y, center_x = img_shape[0] // 2, img_shape[1] // 2
    
    # 最大距离（用于距离归一化）
    max_dist = np.sqrt(center_x**2 + center_y**2)
    
    for segment_id, stats in segment_stats.items():
        # 获取区域掩码
        mask = segments == segment_id
        
        # 基本混合比例（生成图像的比例）
        base_ratio = 0.7
        
        # 根据区域特性调整混合比例
        
        # 1. 边缘密度高的区域（细节多）保留更多原始图像
        edge_factor = 0.3 * (stats['edge_density'] / 30)  # 归一化并缩放
        
        # 2. 颜色变化大的区域（纹理多）保留更多原始图像
        color_var_factor = 0.2 * (np.mean(stats['std_color']) / 50)
        
        # 3. 根据区域在图像中的位置调整（边缘区域更多使用生成图像）
        pos_y, pos_x = stats['position']
        dist_to_center = np.sqrt((pos_y - center_y)**2 + (pos_x - center_x)**2)
        dist_factor = 0.1 * (dist_to_center / max_dist)
        
        # 4. 区域大小（小区域更多使用生成图像，避免孤立区域）
        size_factor = -0.1 * (stats['size'] / (img_shape[0] * img_shape[1] / 100))
        
        # 5. 饱和度调整（高饱和度区域保留更多原始图像）
        saturation_factor = 0.2 * (stats['avg_color_hsv'][1] / 255)
        
        # 组合所有因素
        adjustment = edge_factor + color_var_factor - dist_factor + size_factor + saturation_factor
        adjusted_ratio = base_ratio + adjustment
        
        # 限制在合理范围内
        adjusted_ratio = max(0.3, min(0.9, adjusted_ratio))
        
        # 应用到混合图
        blend_map[mask] = adjusted_ratio
    
    # 平滑混合图，避免区域边界锐变
    blend_map = gaussian_filter(blend_map, sigma=3)
    
    return blend_map

def enhanced_local_style_transfer(model, img_path, output_path, device):
    """增强版局部风格转换"""
    # 加载原始图像
    original_img = Image.open(img_path).convert('RGB')
    
    # 调整图像大小并保持宽高比
    width, height = original_img.size
    
    # 确定新的尺寸，长边为256像素
    if width > height:
        new_width = 256
        new_height = int(height * (256 / width))
    else:
        new_height = 256
        new_width = int(width * (256 / height))
    
    # 调整图像大小
    resized_img = original_img.resize((new_width, new_height), Image.LANCZOS)
    
    # 创建一个256x256的画布（黑色背景）
    canvas = Image.new('RGB', (256, 256), (0, 0, 0))
    
    # 将调整大小的图像粘贴到画布中心
    offset_x = (256 - new_width) // 2
    offset_y = (256 - new_height) // 2
    canvas.paste(resized_img, (offset_x, offset_y))
    
    # 预处理
    input_tensor = preprocess_image(canvas).to(device)
    
    # 生成风格转换图像
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # 后处理
    stylized_img = standard_postprocess(output_tensor)
    
    # 将原始图像和风格化图像调整为相同大小
    original_rgb = np.array(canvas)
    stylized_rgb = np.array(stylized_img)
    
    # 获取分割掩码
    print("正在进行图像分割...")
    segments = get_segmentation_mask(canvas, method='felzenszwalb', n_segments=100)
    
    # 分析各个区域
    print("分析图像区域特性...")
    segment_stats = analyze_segments(canvas, segments)
    
    # 确定每个区域的混合比例
    print("确定最佳混合比例...")
    blend_map = determine_blend_ratios(segment_stats, segments, original_rgb.shape)
    
    # 扩展混合图为3通道
    blend_map_3d = np.stack([blend_map] * 3, axis=2)
    
    # 混合原始图像和风格化图像
    print("混合图像...")
    blended = stylized_rgb * blend_map_3d + original_rgb * (1 - blend_map_3d)
    
    # 后处理：增强细节和颜色
    print("增强图像...")
    blended_img = Image.fromarray(blended.astype(np.uint8))
    
    # 增强颜色
    blended_np = np.array(blended_img)
    hsv = cv2.cvtColor(blended_np, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    
    # 增加饱和度
    s = cv2.multiply(s, 1.2)
    s = np.clip(s, 0, 255).astype(np.uint8)
    
    # 增加对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    
    # 合并通道
    hsv = cv2.merge([h, s, v])
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # 锐化
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # 噪声抑制
    final = cv2.bilateralFilter(sharpened, 5, 50, 50)
    
    # 转换为PIL图像
    final_img = Image.fromarray(final)
    
    # 如果原始图像不是正方形，裁剪回原始宽高比
    if width != height:
        new_width = min(256, int(height * (width / height)))
        new_height = min(256, int(width * (height / width)))
        left = (256 - new_width) // 2
        top = (256 - new_height) // 2
        right = left + new_width
        bottom = top + new_height
        final_img = final_img.crop((left, top, right, bottom))
    
    # 调整回原始大小（可选）
    if (width > 256 or height > 256) and (width * height <= 1024 * 1024):  # 限制最大尺寸
        final_img = final_img.resize((width, height), Image.LANCZOS)
    
    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_img.save(output_path)
    print(f"已保存增强图像至: {output_path}")
    
    # 创建对比图
    create_comparison(canvas, stylized_img, blend_map, final_img, 
                     os.path.join(os.path.dirname(output_path), "comparison.jpg"))
    
    return final_img

def create_comparison(original, stylized, blend_map, final, output_path):
    """创建对比图"""
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.title("原始图像")
    plt.imshow(np.array(original))
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.title("风格化图像")
    plt.imshow(np.array(stylized))
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.title("混合比例图")
    plt.imshow(blend_map, cmap='viridis')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.title("最终增强图像")
    plt.imshow(np.array(final))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="增强版局部风格转换")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--model", type=str, required=True, help="模型文件路径")
    parser.add_argument("--output", type=str, default="output/enhanced_local_style.jpg", help="输出图像路径")
    parser.add_argument("--channels", type=int, default=16, help="模型通道数")
    parser.add_argument("--blocks", type=int, default=1, help="Transformer块数量")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    model, model_type = load_model(args.model, device, args.channels, args.blocks)
    
    # 执行增强版局部风格转换
    enhanced_local_style_transfer(model, args.image, args.output, device) 