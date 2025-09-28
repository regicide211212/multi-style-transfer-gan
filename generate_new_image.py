import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from enhanced_generator import EnhancedGenerator

def convert_monet_to_photo(input_image_path, output_image_path, model_path=None):
    """
    将一张新的输入图像转换为莫奈风格
    
    参数:
        input_image_path: 输入图像路径
        output_image_path: 输出图像保存路径
        model_path: 预训练模型路径
    """
    # 1. 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化生成器模型
    model = EnhancedGenerator(channels=64, num_transformer_blocks=3)
    
    # 加载预训练权重
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"已加载预训练模型: {model_path}")
    else:
        print("警告: 未找到预训练模型，将使用未训练的模型")
    
    model = model.to(device)
    model.eval()  # 设置为评估模式
    
    # 2. 预处理输入图像
    def preprocess_image(img_path, target_size=(256, 256)):
        img = Image.open(img_path).convert('RGB')
        
        # 确保图像尺寸为4的倍数
        w, h = img.size
        new_w = (w // 4) * 4
        new_h = (h // 4) * 4
        
        if new_w != w or new_h != h:
            img = img.crop((0, 0, new_w, new_h))
        
        # 如果需要调整到特定尺寸
        if target_size:
            img = img.resize(target_size, Image.LANCZOS)
        
        # 转换为tensor并归一化到[-1,1]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        return transform(img).unsqueeze(0)  # 添加批次维度
    
    # 处理输入图像
    input_tensor = preprocess_image(input_image_path)
    input_tensor = input_tensor.to(device)
    
    # 3. 使用模型生成转换后的图像
    with torch.no_grad():  # 不计算梯度
        output_tensor = model(input_tensor)
    
    # 4. 后处理生成的图像
    def postprocess_image(tensor):
        # 将[-1,1]范围转换回[0,1]
        tensor = (tensor + 1) / 2.0
        tensor = tensor.clamp(0, 1)
        
        # 转换为PIL图像
        tensor = tensor.cpu().squeeze(0)
        tensor = tensor.permute(1, 2, 0).numpy()
        
        return Image.fromarray((tensor * 255).astype('uint8'))
    
    # 处理输出图像
    output_image = postprocess_image(output_tensor)
    
    # 5. 保存结果
    output_image.save(output_image_path)
    print(f"生成的图像已保存至: {output_image_path}")
    
    # 6. 可视化对比
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("原始图像")
    plt.imshow(Image.open(input_image_path))
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("风格转换后")
    plt.imshow(output_image)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return output_image

# 使用示例
if __name__ == "__main__":
    input_image = "input/test_image.jpg"  # 替换为您的新图像路径
    output_image = "output/generated_image.jpg"  # 替换为输出保存路径
    model_checkpoint = "checkpoints/generator_latest.pth"  # 替换为您的模型权重路径
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_image), exist_ok=True)
    
    # 执行风格转换
    convert_monet_to_photo(input_image, output_image, model_checkpoint) 