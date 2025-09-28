import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
from enhanced_generator import EnhancedGenerator
import argparse

def load_model(model_path, device):
    """加载预训练模型"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # 判断模型类型
    if "G_AB_state_dict" in checkpoint:
        state_dict = checkpoint["G_AB_state_dict"]
        model_type = "G_AB"  # 照片转莫奈
    elif "G_BA_state_dict" in checkpoint:
        state_dict = checkpoint["G_BA_state_dict"]
        model_type = "G_BA"  # 莫奈转照片
    else:
        raise ValueError("未知的模型格式")
    
    # 分析模型结构
    first_key = list(state_dict.keys())[0]  # 第一个键
    if first_key == "initial.0.weight":
        weight_shape = state_dict[first_key].shape
        channels = weight_shape[0]  # 获取通道数
    else:
        channels = 16  # 默认值
    
    print(f"模型类型: {model_type}, 检测到的通道数: {channels}")
    
    # 创建模型
    model = EnhancedGenerator(channels=channels, num_transformer_blocks=1)
    
    # 加载权重
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model, model_type

def transform_image(model, image_path, output_path, device, input_size=256):
    """转换单张图像"""
    # 读取图像
    img = Image.open(image_path).convert("RGB")
    original_size = img.size
    print(f"原始图像尺寸: {original_size}")
    
    # 创建预处理函数
    preprocess = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 预处理图像
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    
    # 运行模型
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # 后处理
    output_tensor = (output_tensor.clone() + 1) / 2.0
    output_tensor = output_tensor.clamp(0, 1)
    output_tensor = output_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
    
    # 转换为PIL图像
    output_img = Image.fromarray((output_tensor * 255).astype('uint8'))
    
    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_img.save(output_path)
    print(f"结果已保存到: {output_path}")
    
    # 返回生成的图像
    return output_img

def test_different_sizes(model, image_path, device, model_type, output_dir="output"):
    """测试不同输入尺寸"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 测试不同的输入尺寸
    sizes = [128, 256, 512, 768]
    results = []
    
    for size in sizes:
        print(f"\n测试输入尺寸: {size}x{size}")
        output_path = f"{output_dir}/{model_type}_size_{size}.jpg"
        
        try:
            output_img = transform_image(model, image_path, output_path, device, size)
            results.append((size, output_img))
        except Exception as e:
            print(f"尺寸 {size} 处理失败: {str(e)}")
    
    # 绘制对比图
    if results:
        plt.figure(figsize=(15, 5))
        
        # 显示原图
        plt.subplot(1, len(results) + 1, 1)
        plt.title("原始图像")
        plt.imshow(Image.open(image_path))
        plt.axis('off')
        
        # 显示各种尺寸的结果
        for i, (size, img) in enumerate(results, 2):
            plt.subplot(1, len(results) + 1, i)
            plt.title(f"尺寸: {size}x{size}")
            plt.imshow(img)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_type}_size_comparison.jpg")
        plt.show()

def try_skip_connections(model, image_path, device, model_type, output_dir="output"):
    """尝试手动添加跳过连接"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取图像
    img = Image.open(image_path).convert("RGB")
    
    # 预处理
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 转换为tensor
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    
    # 保存原始输入tensor
    original_input = input_tensor.clone()
    
    # 运行模型并尝试添加跳过连接
    def post_process(tensor):
        tensor = (tensor.clone() + 1) / 2.0
        tensor = tensor.clamp(0, 1)
        tensor = tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
        return Image.fromarray((tensor * 255).astype('uint8'))
    
    with torch.no_grad():
        # 正常输出
        normal_output = model(input_tensor)
        normal_img = post_process(normal_output)
        normal_img.save(f"{output_dir}/{model_type}_normal.jpg")
        
        # 尝试添加输入残差连接（混合原始输入与输出）
        # 方法1: 50%原始 + 50%生成
        mixed_output_1 = normal_output * 0.5 + (original_input * 2 - 1) * 0.5
        mixed_img_1 = post_process(mixed_output_1)
        mixed_img_1.save(f"{output_dir}/{model_type}_mixed_50.jpg")
        
        # 方法2: 30%原始 + 70%生成
        mixed_output_2 = normal_output * 0.7 + (original_input * 2 - 1) * 0.3
        mixed_img_2 = post_process(mixed_output_2)
        mixed_img_2.save(f"{output_dir}/{model_type}_mixed_30.jpg")
        
        # 方法3: 10%原始 + 90%生成
        mixed_output_3 = normal_output * 0.9 + (original_input * 2 - 1) * 0.1
        mixed_img_3 = post_process(mixed_output_3)
        mixed_img_3.save(f"{output_dir}/{model_type}_mixed_10.jpg")
    
    # 绘制对比图
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.title("原始图像")
    plt.imshow(img)
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.title("正常输出")
    plt.imshow(normal_img)
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.title("混合 (50%原始)")
    plt.imshow(mixed_img_1)
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.title("混合 (30%原始)")
    plt.imshow(mixed_img_2)
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.title("混合 (10%原始)")
    plt.imshow(mixed_img_3)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_type}_skip_connection_test.jpg")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="直接图像风格转换")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--model", type=str, required=True, help="模型文件路径")
    parser.add_argument("--output", type=str, default="output/result.jpg", help="输出图像路径")
    parser.add_argument("--test_sizes", action="store_true", help="测试不同输入尺寸")
    parser.add_argument("--test_skip", action="store_true", help="测试跳过连接")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    model, model_type = load_model(args.model, device)
    
    # 根据参数执行不同的操作
    if args.test_sizes:
        test_different_sizes(model, args.image, device, model_type)
    elif args.test_skip:
        try_skip_connections(model, args.image, device, model_type)
    else:
        transform_image(model, args.image, args.output, device) 