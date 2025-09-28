import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import os
import argparse

def adaptive_color_correction(img, blocks_detected=None, radius=50):
    """自适应颜色校正，只处理检测到色块的区域"""
    # 转换为numpy数组
    if isinstance(img, Image.Image):
        img_np = np.array(img)
    else:
        img_np = img.copy()
    
    # 如果未提供色块检测结果，尝试自动检测
    if blocks_detected is None:
        blocks_detected = detect_color_blocks(img_np)
    
    # 如果检测到色块
    if blocks_detected.any():
        # 创建结果图像
        result = img_np.copy()
        
        # 对每个检测到色块的区域进行局部处理
        for y in range(blocks_detected.shape[0]):
            for x in range(blocks_detected.shape[1]):
                if blocks_detected[y, x]:
                    # 获取该点周围的区域
                    y_start = max(0, y - radius)
                    y_end = min(img_np.shape[0], y + radius + 1)
                    x_start = max(0, x - radius)
                    x_end = min(img_np.shape[1], x + radius + 1)
                    
                    # 计算该区域的均值
                    region = img_np[y_start:y_end, x_start:x_end]
                    mean_color = np.mean(region, axis=(0, 1))
                    
                    # 平滑过渡
                    weight = 0.5  # 混合权重
                    result[y, x] = weight * img_np[y, x] + (1 - weight) * mean_color
        
        # 返回处理后的图像
        if isinstance(img, Image.Image):
            return Image.fromarray(result)
        return result
    
    # 如果没有检测到色块，返回原图
    return img

def detect_color_blocks(img, threshold=30, kernel_size=11):
    """检测图像中的色块区域"""
    # 确保图像是numpy数组
    if isinstance(img, Image.Image):
        img_np = np.array(img)
    else:
        img_np = img
    
    # 转换为RGB
    if len(img_np.shape) == 2:
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img_np
    
    # 转换为LAB颜色空间
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    
    # 计算局部区域的标准差
    a_channel = img_lab[:, :, 1].astype(np.float32)
    b_channel = img_lab[:, :, 2].astype(np.float32)
    
    # 使用Sobel算子计算梯度
    a_gradient_x = cv2.Sobel(a_channel, cv2.CV_32F, 1, 0, ksize=3)
    a_gradient_y = cv2.Sobel(a_channel, cv2.CV_32F, 0, 1, ksize=3)
    b_gradient_x = cv2.Sobel(b_channel, cv2.CV_32F, 1, 0, ksize=3)
    b_gradient_y = cv2.Sobel(b_channel, cv2.CV_32F, 0, 1, ksize=3)
    
    # 计算梯度幅度
    a_gradient_mag = np.sqrt(a_gradient_x**2 + a_gradient_y**2)
    b_gradient_mag = np.sqrt(b_gradient_x**2 + b_gradient_y**2)
    
    # 结合A和B通道的梯度
    combined_gradient = (a_gradient_mag + b_gradient_mag) / 2
    
    # 使用阈值检测色块边界
    _, block_edges = cv2.threshold(combined_gradient, threshold, 255, cv2.THRESH_BINARY)
    block_edges = block_edges.astype(np.uint8)
    
    # 膨胀边界
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_edges = cv2.dilate(block_edges, kernel, iterations=1)
    
    return dilated_edges > 0

def edge_preserving_smoothing(img, sigma_s=60, sigma_r=0.4):
    """边缘保留平滑，保持图像清晰度"""
    # 转换为numpy数组
    if isinstance(img, Image.Image):
        img_np = np.array(img)
    else:
        img_np = img
    
    # 应用双边滤波
    smoothed = cv2.bilateralFilter(img_np, 0, sigma_r*255, sigma_s)
    
    # 如果原始输入是PIL图像，则返回PIL图像
    if isinstance(img, Image.Image):
        return Image.fromarray(smoothed)
    
    return smoothed

def detail_enhancing_blend(img, original, alpha=0.3, beta=1.5):
    """增强细节的混合"""
    # 确保两张图像大小相同
    if isinstance(img, Image.Image) and isinstance(original, Image.Image):
        if img.size != original.size:
            original = original.resize(img.size, Image.LANCZOS)
        
        # 转换为numpy数组
        img_np = np.array(img).astype(float)
        orig_np = np.array(original).astype(float)
        
        # 提取细节层
        blurred = cv2.GaussianBlur(orig_np, (0, 0), 3)
        detail = orig_np - blurred
        
        # 混合: 基础图像 + 细节增强
        blended = img_np * (1 - alpha) + orig_np * alpha + detail * beta
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        return Image.fromarray(blended)
    
    return img

def fix_color_blocks_improved(input_img_path, original_img_path=None, output_img_path=None):
    """改进的色块修复方法，保持图像清晰度"""
    # 加载生成的图像
    generated_img = Image.open(input_img_path).convert("RGB")
    
    # 1. 检测色块区域
    blocks_detected = detect_color_blocks(generated_img)
    
    # 2. 对检测到的区域应用自适应颜色校正
    color_corrected = adaptive_color_correction(generated_img, blocks_detected)
    
    # 3. 应用边缘保留平滑
    smoothed_img = edge_preserving_smoothing(color_corrected)
    
    # 4. 如果提供了原始图像，则进行细节增强的混合
    if original_img_path and os.path.exists(original_img_path):
        original_img = Image.open(original_img_path).convert("RGB")
        final_img = detail_enhancing_blend(smoothed_img, original_img, alpha=0.1, beta=0.5)
    else:
        final_img = smoothed_img
    
    # 保存结果
    if output_img_path:
        os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
        final_img.save(output_img_path)
        print(f"已保存处理后的图像至: {output_img_path}")
    
    return final_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="改进的色块修复方法")
    parser.add_argument("--input", type=str, required=True, help="输入图像路径")
    parser.add_argument("--original", type=str, help="原始图像路径")
    parser.add_argument("--output", type=str, default="output/fixed_image_improved.jpg", help="输出图像路径")
    
    args = parser.parse_args()
    
    fix_color_blocks_improved(args.input, args.original, args.output) 