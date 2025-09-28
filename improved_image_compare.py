import os
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
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
    
    # 计算PSNR
    psnr_value = psnr(img1, img2, data_range=1.0)
    
    # 计算SSIM
    ssim_value = ssim(img1, img2, channel_axis=2, data_range=1.0)
    
    return mse_value, psnr_value, ssim_value

def get_file_ext(path):
    """获取文件扩展名"""
    return os.path.splitext(path)[1].lower()

def compare_folders_with_original():
    # 文件夹路径
    original_folder = "E:\\desktop\\graduate\\gan_proj_0217_copy\\monet2photo_uvcgan2\\datasets\\testB"  # 原始照片文件夹
    cyclegan_folder = "E:\\desktop\\graduate\\gan_proj_0217_copy\\monet2photo_uvcgan2\\results\\cyclegan_monet2photo"
    local_style_folder = "E:\\desktop\\graduate\\gan_proj_0217_copy\\monet2photo_uvcgan2\\results\\local_style_enhanced_monet2photo"
    
    # 获取所有图像文件
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    
    original_images = []
    for ext in extensions:
        original_images.extend(glob.glob(os.path.join(original_folder, ext)))
    
    cyclegan_images = []
    for ext in extensions:
        cyclegan_images.extend(glob.glob(os.path.join(cyclegan_folder, ext)))
    
    local_style_images = []
    for ext in extensions:
        local_style_images.extend(glob.glob(os.path.join(local_style_folder, ext)))
    
    print(f"原始照片: 找到 {len(original_images)} 张图像")
    print(f"CycleGAN处理: 找到 {len(cyclegan_images)} 张图像")
    print(f"局部风格增强: 找到 {len(local_style_images)} 张图像")
    
    # 获取文件基础名（不包含路径和扩展名）
    original_names = [os.path.splitext(os.path.basename(img))[0] for img in original_images]
    cyclegan_names = [os.path.splitext(os.path.basename(img))[0] for img in cyclegan_images]
    local_style_names = [os.path.splitext(os.path.basename(img))[0] for img in local_style_images]
    
    # 找出三个文件夹中都存在的图像名称
    common_names = set()
    for name in original_names:
        for c_name in cyclegan_names:
            if name in c_name or c_name in name:
                for l_name in local_style_names:
                    if name in l_name or l_name in name:
                        common_names.add(name)
                        break
                break
    
    print(f"找到 {len(common_names)} 个在所有文件夹中都有对应版本的图像")
    
    # 为每个图像找到对应的完整路径
    matching_sets = []
    for name in common_names:
        original_match = None
        cyclegan_match = None
        local_style_match = None
        
        # 匹配原始图像
        for img in original_images:
            base_name = os.path.splitext(os.path.basename(img))[0]
            if name == base_name or name in base_name or base_name in name:
                original_match = img
                break
                
        # 匹配CycleGAN图像
        for img in cyclegan_images:
            base_name = os.path.splitext(os.path.basename(img))[0]
            if name == base_name or name in base_name or base_name in name:
                cyclegan_match = img
                break
                
        # 匹配局部风格图像
        for img in local_style_images:
            base_name = os.path.splitext(os.path.basename(img))[0]
            if name == base_name or name in base_name or base_name in name:
                local_style_match = img
                break
        
        if original_match and cyclegan_match and local_style_match:
            matching_sets.append((original_match, cyclegan_match, local_style_match))
    
    if not matching_sets:
        print("无法找到三个文件夹中的匹配图像，请检查文件名格式")
        return
    
    print(f"\n找到 {len(matching_sets)} 组完整匹配")
    
    # 计算图像质量指标
    cyclegan_metrics = {"mse": [], "psnr": [], "ssim": []}
    local_style_metrics = {"mse": [], "psnr": [], "ssim": []}
    
    print("\n=== 质量评估结果 ===")
    
    for original_path, cyclegan_path, local_style_path in matching_sets:
        original_img = cv2.imread(original_path)
        cyclegan_img = cv2.imread(cyclegan_path)
        local_style_img = cv2.imread(local_style_path)
        
        if original_img is None or cyclegan_img is None or local_style_img is None:
            print(f"无法读取图像组: {os.path.basename(original_path)}")
            continue
        
        # 计算CycleGAN与原始图像的指标
        cy_mse, cy_psnr, cy_ssim = calculate_metrics(original_img, cyclegan_img)
        cyclegan_metrics["mse"].append(cy_mse)
        cyclegan_metrics["psnr"].append(cy_psnr)
        cyclegan_metrics["ssim"].append(cy_ssim)
        
        # 计算局部风格与原始图像的指标
        ls_mse, ls_psnr, ls_ssim = calculate_metrics(original_img, local_style_img)
        local_style_metrics["mse"].append(ls_mse)
        local_style_metrics["psnr"].append(ls_psnr)
        local_style_metrics["ssim"].append(ls_ssim)
        
        print(f"\n图像: {os.path.basename(original_path)}")
        print(f"  CycleGAN ({os.path.basename(cyclegan_path)}):")
        print(f"    MSE: {cy_mse:.6f}, PSNR: {cy_psnr:.2f}dB, SSIM: {cy_ssim:.4f}")
        print(f"  局部风格增强 ({os.path.basename(local_style_path)}):")
        print(f"    MSE: {ls_mse:.6f}, PSNR: {ls_psnr:.2f}dB, SSIM: {ls_ssim:.4f}")
        print(f"  对比 (局部风格相对于CycleGAN):")
        print(f"    MSE: {'更好' if ls_mse < cy_mse else '更差'}, "
              f"PSNR: {'更好' if ls_psnr > cy_psnr else '更差'}, "
              f"SSIM: {'更好' if ls_ssim > cy_ssim else '更差'}")
    
    # 计算平均指标
    avg_cy_mse = np.mean(cyclegan_metrics["mse"])
    avg_cy_psnr = np.mean(cyclegan_metrics["psnr"])
    avg_cy_ssim = np.mean(cyclegan_metrics["ssim"])
    
    avg_ls_mse = np.mean(local_style_metrics["mse"])
    avg_ls_psnr = np.mean(local_style_metrics["psnr"])
    avg_ls_ssim = np.mean(local_style_metrics["ssim"])
    
    print("\n" + "=" * 50)
    print("平均指标对比:")
    print(f"  CycleGAN:")
    print(f"    MSE: {avg_cy_mse:.6f}")
    print(f"    PSNR: {avg_cy_psnr:.2f}dB")
    print(f"    SSIM: {avg_cy_ssim:.4f}")
    print(f"  局部风格增强:")
    print(f"    MSE: {avg_ls_mse:.6f}")
    print(f"    PSNR: {avg_ls_psnr:.2f}dB")
    print(f"    SSIM: {avg_ls_ssim:.4f}")
    print(f"  对比结果 (局部风格相对于CycleGAN):")
    print(f"    MSE: {'更好' if avg_ls_mse < avg_cy_mse else '更差'} ({(avg_cy_mse - avg_ls_mse) / avg_cy_mse * 100:.2f}%)")
    print(f"    PSNR: {'更好' if avg_ls_psnr > avg_cy_psnr else '更差'} ({(avg_ls_psnr - avg_cy_psnr) / avg_cy_psnr * 100:.2f}%)")
    print(f"    SSIM: {'更好' if avg_ls_ssim > avg_cy_ssim else '更差'} ({(avg_ls_ssim - avg_cy_ssim) / avg_cy_ssim * 100:.2f}%)")
    print("=" * 50)
    
    print("\n结果解释:")
    print("- MSE (均方误差): 越低表示图像越相似")
    print("- PSNR (峰值信噪比): 越高表示图像质量越好") 
    print("- SSIM (结构相似性): 越接近1表示结构相似度越高")
    
    # 总结结果
    better_count = {
        "mse": sum(1 for ls, cy in zip(local_style_metrics["mse"], cyclegan_metrics["mse"]) if ls < cy),
        "psnr": sum(1 for ls, cy in zip(local_style_metrics["psnr"], cyclegan_metrics["psnr"]) if ls > cy),
        "ssim": sum(1 for ls, cy in zip(local_style_metrics["ssim"], cyclegan_metrics["ssim"]) if ls > cy)
    }
    
    total = len(matching_sets)
    print(f"\n在 {total} 张图像中:")
    print(f"- MSE: 局部风格增强在 {better_count['mse']} 张图像上优于CycleGAN ({better_count['mse']/total*100:.1f}%)")
    print(f"- PSNR: 局部风格增强在 {better_count['psnr']} 张图像上优于CycleGAN ({better_count['psnr']/total*100:.1f}%)")
    print(f"- SSIM: 局部风格增强在 {better_count['ssim']} 张图像上优于CycleGAN ({better_count['ssim']/total*100:.1f}%)")

if __name__ == "__main__":
    compare_folders_with_original()