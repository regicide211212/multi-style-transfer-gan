import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from enhanced_generator import EnhancedGenerator
import os

def load_and_test_model(model_path, image_path, output_path):
    """加载模型并测试单张图像"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model = EnhancedGenerator(channels=16, num_transformer_blocks=1)
    
    checkpoint = torch.load(model_path, map_location=device)
    if "G_AB_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["G_AB_state_dict"])
    elif "G_BA_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["G_BA_state_dict"])
    else:
        raise ValueError("未找到模型状态字典")
    
    model = model.to(device)
    model.eval()
    print("模型加载成功")
    
    # 记录模型参数形状
    print("模型参数形状:")
    for name, param in model.named_parameters():
        if name.startswith("initial") or "weight" in name:
            print(f"{name}: {param.shape}")
    
    # 加载图像
    img = Image.open(image_path).convert("RGB")
    
    # 预处理图像 (不同方法供测试)
    transform1 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform2 = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 尝试两种预处理方法
    input_tensor1 = transform1(img).unsqueeze(0).to(device)
    input_tensor2 = transform2(img).unsqueeze(0).to(device)
    
    print(f"输入图像尺寸1: {input_tensor1.shape}")
    print(f"输入图像尺寸2: {input_tensor2.shape}")
    
    # 生成图像
    with torch.no_grad():
        output_tensor1 = model(input_tensor1)
        output_tensor2 = model(input_tensor2)
    
    # 后处理
    def postprocess(tensor):
        tensor = (tensor.clone() + 1) / 2.0
        tensor = tensor.clamp(0, 1)
        tensor = tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
        return Image.fromarray((tensor * 255).astype('uint8'))
    
    output_image1 = postprocess(output_tensor1)
    output_image2 = postprocess(output_tensor2)
    
    # 保存图像
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_image1.save(output_path.replace('.jpg', '_method1.jpg'))
    output_image2.save(output_path.replace('.jpg', '_method2.jpg'))
    
    # 显示图像对比
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.title("原始图像")
    plt.imshow(img)
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.title("方法1 (Resize+CenterCrop)")
    plt.imshow(output_image1)
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.title("方法2 (直接Resize)")
    plt.imshow(output_image2)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path.replace('.jpg', '_comparison.jpg'))
    plt.show()

if __name__ == "__main__":
    # 设置路径
    model_path = "models/G_AB_epoch_200.pth"  # 修改为您的模型路径
    image_path = "input/test_image.jpg"       # 修改为您的测试图像路径
    output_path = "output/debug_output.jpg"   # 修改为您想保存的输出路径
    
    # 执行测试
    load_and_test_model(model_path, image_path, output_path) 