import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np
from enhanced_generator import EnhancedGenerator
import argparse
import cv2

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

def generate_with_different_settings(model, image_path, output_dir, device, model_type):
    """使用不同的设置生成图像"""
    os.makedirs(output_dir, exist_ok=True)
    img = Image.open(image_path).convert("RGB")
    
    # 创建不同的预处理方法和后处理方法
    settings = [
        {
            "name": "标准处理",
            "preprocess": transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            "postprocess": lambda x: standard_postprocess(x)
        },
        {
            "name": "提高对比度",
            "preprocess": transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ColorJitter(brightness=0.1, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            "postprocess": lambda x: contrast_enhanced_postprocess(x)
        },
        {
            "name": "多尺度融合",
            "preprocess": transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            "postprocess": lambda x: multi_scale_fusion(x, img)
        },
        {
            "name": "细节增强处理",
            "preprocess": transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            "postprocess": lambda x: detail_enhanced_postprocess(x, img)
        },
        {
            "name": "局部风格应用",
            "preprocess": transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            "postprocess": lambda x: local_style_transfer(x, img)
        },
    ]
    
    results = []
    
    for setting in settings:
        print(f"尝试 {setting['name']}...")
        input_tensor = setting["preprocess"](img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        output_image = setting["postprocess"](output_tensor)
        output_path = os.path.join(output_dir, f"{model_type}_{setting['name'].replace(' ', '_')}.jpg")
        output_image.save(output_path)
        print(f"已保存到 {output_path}")
        
        results.append((setting["name"], output_image))
    
    # 创建对比图
    plt.figure(figsize=(15, 10))
    
    # 显示原图
    plt.subplot(2, 3, 1)
    plt.title("原始图像")
    plt.imshow(img)
    plt.axis('off')
    
    # 显示各种方法的结果
    for i, (name, img) in enumerate(results, 2):
        plt.subplot(2, 3, i)
        plt.title(name)
        plt.imshow(img)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_type}_comparison.jpg"))
    plt.show()

# 标准后处理
def standard_postprocess(tensor):
    tensor = (tensor.clone() + 1) / 2.0
    tensor = tensor.clamp(0, 1)
    tensor = tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
    return Image.fromarray((tensor * 255).astype('uint8'))

# 增强对比度的后处理
def contrast_enhanced_postprocess(tensor):
    # 基本后处理
    img = standard_postprocess(tensor)
    
    # 转换为numpy数组
    img_np = np.array(img)
    
    # 增强对比度
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # 应用CLAHE (对比度受限的自适应直方图均衡化)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # 合并通道
    enhanced_lab = cv2.merge((cl, a, b))
    
    # 转换回RGB
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    # 调整饱和度
    hsv = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.multiply(s, 1.2)  # 增加饱和度
    s = np.clip(s, 0, 255).astype(np.uint8)
    enhanced_hsv = cv2.merge((h, s, v))
    final = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2RGB)
    
    return Image.fromarray(final)

# 多尺度融合
def multi_scale_fusion(tensor, original_img):
    # 基本后处理
    base_output = standard_postprocess(tensor)
    
    # 转换为numpy数组
    output_np = np.array(base_output).astype(np.float32) / 255.0
    
    # 创建不同尺度的输出
    scales = [0.5, 0.75, 1.0]
    scaled_outputs = []
    
    for scale in scales:
        # 调整原始图像大小
        width, height = original_img.size
        new_size = (int(width * scale), int(height * scale))
        scaled_img = original_img.resize(new_size, Image.LANCZOS)
        
        # 预处理
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # 转换
        input_tensor = preprocess(scaled_img).unsqueeze(0).to(tensor.device)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        # 后处理
        scaled_output = standard_postprocess(output)
        scaled_np = np.array(scaled_output).astype(np.float32) / 255.0
        
        scaled_outputs.append(scaled_np)
    
    # 融合不同尺度的输出
    weights = [0.2, 0.3, 0.5]  # 权重
    fused = np.zeros_like(scaled_outputs[0])
    
    for output, weight in zip(scaled_outputs, weights):
        fused += output * weight
    
    # 最终增强
    fused = np.clip(fused * 1.1, 0, 1)  # 稍微增强亮度
    
    return Image.fromarray((fused * 255).astype(np.uint8))

# 细节增强后处理
def detail_enhanced_postprocess(tensor, original_img):
    # 基本后处理
    base_output = standard_postprocess(tensor)
    
    # 转换为numpy数组
    output_np = np.array(base_output)
    
    # 创建原始图像的相同大小的版本
    orig_np = np.array(original_img.resize(base_output.size, Image.LANCZOS))
    
    # 提取原始图像的细节层
    orig_gray = cv2.cvtColor(orig_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(orig_gray, (0, 0), 3)
    detail = orig_gray - blurred
    
    # 增强生成图像上的细节
    output_lab = cv2.cvtColor(output_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(output_lab)
    
    # 添加原始图像的细节到亮度通道
    l = l.astype(np.float32)
    detail = detail.astype(np.float32)
    l = np.clip(l + detail * 0.5, 0, 255).astype(np.uint8)
    
    # 合并回LAB
    enhanced_lab = cv2.merge((l, a, b))
    
    # 转换回RGB
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    # 调整最终图像的色调和饱和度
    hsv = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.multiply(s, 1.2)  # 增加饱和度
    v = cv2.multiply(v, 1.1)  # 增加亮度
    s = np.clip(s, 0, 255).astype(np.uint8)
    v = np.clip(v, 0, 255).astype(np.uint8)
    enhanced_hsv = cv2.merge((h, s, v))
    final = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2RGB)
    
    return Image.fromarray(final)

# 局部风格应用
def local_style_transfer(tensor, original_img):
    # 基本后处理
    base_output = standard_postprocess(tensor)
    
    # 转换为numpy数组
    output_np = np.array(base_output)
    orig_np = np.array(original_img.resize(base_output.size, Image.LANCZOS))
    
    # 使用分水岭算法或其他分割算法分割图像
    # 这里简化为使用K-means聚类
    Z = orig_np.reshape((-1, 3))
    Z = np.float32(Z)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 5  # 分割成5个区域
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    labels = labels.reshape(orig_np.shape[0], orig_np.shape[1])
    
    # 创建结果图像
    result = np.zeros_like(output_np)
    
    # 对每个区域应用不同的混合比例
    for i in range(K):
        mask = (labels == i).astype(np.float32)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        
        # 水面区域使用较多的生成图像
        if i == 0:  # 假设0是水面区域
            blend_ratio = 0.8
        # 花朵区域保留更多原始细节
        elif i == 1:  # 假设1是花朵区域
            blend_ratio = 0.4
        else:
            blend_ratio = 0.6
            
        # 混合
        region = output_np * blend_ratio + orig_np * (1 - blend_ratio)
        result = result + region * mask
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # 最终增强
    hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.multiply(s, 1.2)  # 增加饱和度
    s = np.clip(s, 0, 255).astype(np.uint8)
    enhanced_hsv = cv2.merge((h, s, v))
    final = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2RGB)
    
    return Image.fromarray(final)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="高级图像风格转换")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--model", type=str, required=True, help="模型文件路径")
    parser.add_argument("--output_dir", type=str, default="output/advanced", help="输出目录")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    model, model_type = load_model(args.model, device)
    
    # 生成不同设置下的结果
    generate_with_different_settings(model, args.image, args.output_dir, device, model_type)