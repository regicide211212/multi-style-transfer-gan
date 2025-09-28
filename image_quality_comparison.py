import os
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob

def calculate_metrics(img1, img2):
    """计算两张图像间的MSE、PSNR和SSIM指标"""
    # 确保两张图像尺寸一致
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # 转换为浮点型并归一化到0-1范围
    img1 = img1.astype(float) / 255.0
    img2 = img2.astype(float) / 255.0
    
    # 计算MSE
    mse_value = np.mean((img1 - img2) ** 2)
    
    # 计算PSNR (使用skimage，更可靠)
    psnr_value = psnr(img1, img2, data_range=1.0)
    
    # 计算SSIM (使用skimage，更可靠)
    ssim_value = ssim(img1, img2, channel_axis=2, data_range=1.0)
    
    return {
        'mse': mse_value,
        'psnr': psnr_value,
        'ssim': ssim_value
    }

def compare_folders(folder1, folder2, output_excel=None, output_chart=None):
    """比较两个文件夹中图像的质量指标"""
    print(f"比较文件夹: \n  {folder1} \n  {folder2}")
    
    # 获取第一个文件夹中的所有图像文件
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    folder1_images = []
    for ext in extensions:
        folder1_images.extend(glob.glob(os.path.join(folder1, f"*{ext}")))
        folder1_images.extend(glob.glob(os.path.join(folder1, f"*{ext.upper()}")))
    
    folder1_image_names = [os.path.basename(img) for img in folder1_images]
    print(f"在 {folder1} 中找到 {len(folder1_images)} 张图像")
    
    # 获取第二个文件夹中的所有图像文件
    folder2_images = []
    for ext in extensions:
        folder2_images.extend(glob.glob(os.path.join(folder2, f"*{ext}")))
        folder2_images.extend(glob.glob(os.path.join(folder2, f"*{ext.upper()}")))
    
    folder2_image_names = [os.path.basename(img) for img in folder2_images]
    print(f"在 {folder2} 中找到 {len(folder2_images)} 张图像")
    
    # 找到两个文件夹中共有的图像
    common_images = []
    for i, name1 in enumerate(folder1_image_names):
        for j, name2 in enumerate(folder2_image_names):
            # 检查文件名相似性 (忽略前缀，比如cyclegan_和local_style_)
            if name1 == name2 or name1 in name2 or name2 in name1:
                common_images.append((folder1_images[i], folder2_images[j]))
                break
    
    print(f"找到 {len(common_images)} 对可比较的图像")
    
    if len(common_images) == 0:
        print("没有可比较的图像，退出")
        return
    
    # 比较每对图像
    results = []
    folder1_name = os.path.basename(folder1)
    folder2_name = os.path.basename(folder2)
    
    for img1_path, img2_path in common_images:
        print(f"比较: \n  {os.path.basename(img1_path)} \n  {os.path.basename(img2_path)}")
        
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            print(f"  无法读取图像，跳过")
            continue
        
        metrics = calculate_metrics(img1, img2)
        
        results.append({
            'image1': os.path.basename(img1_path),
            'image2': os.path.basename(img2_path),
            'mse': metrics['mse'],
            'psnr': metrics['psnr'],
            'ssim': metrics['ssim']
        })
        
        print(f"  MSE: {metrics['mse']:.6f}, PSNR: {metrics['psnr']:.2f}dB, SSIM: {metrics['ssim']:.4f}")
    
    # 计算平均值
    avg_mse = sum(r['mse'] for r in results) / len(results)
    avg_psnr = sum(r['psnr'] for r in results) / len(results)
    avg_ssim = sum(r['ssim'] for r in results) / len(results)
    
    print(f"\n平均指标:")
    print(f"  MSE: {avg_mse:.6f}")
    print(f"  PSNR: {avg_psnr:.2f}dB")
    print(f"  SSIM: {avg_ssim:.4f}")
    
    # 创建摘要数据
    summary_data = [{
        '文件夹': folder1_name,
        '图像数量': len(folder1_images)
    }, {
        '文件夹': folder2_name,
        '图像数量': len(folder2_images)
    }, {
        '文件夹': f"比较结果 ({len(common_images)}对图像)",
        '平均MSE': avg_mse,
        '平均PSNR': avg_psnr,
        '平均SSIM': avg_ssim
    }]
    
    # 创建详细数据
    detail_data = []
    for r in results:
        detail_data.append({
            '图像1': r['image1'],
            '图像2': r['image2'],
            'MSE': r['mse'],
            'PSNR': r['psnr'],
            'SSIM': r['ssim']
        })
    
    # 保存到Excel
    if output_excel:
        with pd.ExcelWriter(output_excel) as writer:
            # 摘要表
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='摘要', index=False)
            
            # 详细结果表
            detail_df = pd.DataFrame(detail_data)
            detail_df.to_excel(writer, sheet_name='详细结果', index=False)
            
        print(f"结果已保存到 {output_excel}")
    
    # 创建图表
    if output_chart:
        plt.figure(figsize=(15, 5))
        
        # SSIM图表
        plt.subplot(1, 3, 1)
        plt.bar(['比较结果'], [avg_ssim], color='blue')
        plt.title('平均SSIM (越高越好)')
        plt.ylim(0, 1)
        plt.text(0, avg_ssim + 0.02, f'{avg_ssim:.4f}', ha='center')
        
        # PSNR图表
        plt.subplot(1, 3, 2)
        plt.bar(['比较结果'], [avg_psnr], color='green')
        plt.title('平均PSNR (越高越好)')
        plt.text(0, avg_psnr + 0.5, f'{avg_psnr:.2f}', ha='center')
        
        # MSE图表
        plt.subplot(1, 3, 3)
        plt.bar(['比较结果'], [avg_mse], color='red')
        plt.title('平均MSE (越低越好)')
        plt.text(0, avg_mse + 0.001, f'{avg_mse:.6f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(output_chart)
        print(f"图表已保存到 {output_chart}")
    
    return summary_data, detail_data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='比较两个文件夹中图像的质量指标')
    parser.add_argument('--folder1', type=str, required=True, help='第一个图像文件夹路径')
    parser.add_argument('--folder2', type=str, required=True, help='第二个图像文件夹路径')
    parser.add_argument('--output_excel', type=str, default='image_comparison_results.xlsx', help='输出Excel文件路径')
    parser.add_argument('--output_chart', type=str, default='image_comparison_chart.png', help='输出图表文件路径')
    
    args = parser.parse_args()
    
    compare_folders(args.folder1, args.folder2, args.output_excel, args.output_chart)