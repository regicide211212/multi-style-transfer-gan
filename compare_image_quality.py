import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import argparse
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def calculate_mse(img1, img2):
    """计算均方误差 (MSE)"""
    mse_value = np.mean((img1 - img2) ** 2)
    print(f"  计算MSE: {mse_value:.8f}")
    return mse_value

def calculate_ssim(img1, img2):
    """计算结构相似性 (SSIM)"""
    # 注意：SSIM需要相同大小的图像并且数据类型为float
    # 添加data_range参数，因为图像已经归一化为0-1范围
    ssim_value = ssim(img1, img2, channel_axis=2, data_range=1.0)
    print(f"  计算SSIM: {ssim_value:.8f}")
    return ssim_value

def calculate_psnr(img1, img2):
    """计算峰值信噪比 (PSNR)"""
    # 添加data_range参数，因为图像已经归一化为0-1范围
    psnr_value = psnr(img1, img2, data_range=1.0)
    print(f"  计算PSNR: {psnr_value:.8f}")
    return psnr_value

def find_matching_images(base_folder, comparison_folder):
    """查找两个文件夹中具有相同名称的图像"""
    # 使用Path对象处理路径，兼容Windows和Unix
    base_path = Path(base_folder)
    comparison_path = Path(comparison_folder)
    
    # 获取图像文件列表
    base_images = set()
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        base_images.update([f.name for f in base_path.glob(ext)])
    
    comparison_images = set()
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        comparison_images.update([f.name for f in comparison_path.glob(ext)])
    
    # 找到两个集合的交集
    common_images = base_images.intersection(comparison_images)
    
    print(f"基准文件夹图像数量: {len(base_images)}")
    print(f"比较文件夹图像数量: {len(comparison_images)}")
    print(f"共同图像数量: {len(common_images)}")
    if len(common_images) > 0:
        print(f"共同图像示例(最多3个): {list(common_images)[:3]}")
    
    return common_images

def compare_with_test_images(local_style_folder, cyclegan_folder, test_images_folder, output_file=None):
    """比较局部风格和CycleGAN结果与原始测试图像的质量"""
    test_path = Path(test_images_folder)
    local_path = Path(local_style_folder)
    cyclegan_path = Path(cyclegan_folder)
    
    # 获取所有测试图像
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        test_images.extend([f.name for f in test_path.glob(ext)])
    
    print(f"测试图像文件夹: {test_images_folder}")
    print(f"找到 {len(test_images)} 张测试图像")
    
    # 检查处理后的图像
    local_style_results = {}
    cyclegan_results = {}
    
    comparison_results = []
    
    # 查找对应的处理结果
    for test_img in tqdm(test_images, desc="查找处理结果"):
        # 检查名称匹配的处理结果
        local_style_file = local_path / test_img
        cyclegan_file = cyclegan_path / test_img
        
        if local_style_file.exists() and cyclegan_file.exists():
            print(f"\n处理图像对: {test_img}")
            
            # 读取图像
            test_img_data = cv2.imread(str(test_path / test_img))
            local_img_data = cv2.imread(str(local_style_file))
            cyclegan_img_data = cv2.imread(str(cyclegan_file))
            
            if test_img_data is None or local_img_data is None or cyclegan_img_data is None:
                print(f"  无法读取一个或多个图像，跳过")
                continue
            
            # 调整大小
            if test_img_data.shape != local_img_data.shape:
                print(f"  调整局部风格图像大小从 {local_img_data.shape} 到 {test_img_data.shape}")
                local_img_data = cv2.resize(local_img_data, (test_img_data.shape[1], test_img_data.shape[0]))
            
            if test_img_data.shape != cyclegan_img_data.shape:
                print(f"  调整CycleGAN图像大小从 {cyclegan_img_data.shape} 到 {test_img_data.shape}")
                cyclegan_img_data = cv2.resize(cyclegan_img_data, (test_img_data.shape[1], test_img_data.shape[0]))
            
            # 转换为RGB
            test_img_rgb = cv2.cvtColor(test_img_data, cv2.COLOR_BGR2RGB)
            local_img_rgb = cv2.cvtColor(local_img_data, cv2.COLOR_BGR2RGB)
            cyclegan_img_rgb = cv2.cvtColor(cyclegan_img_data, cv2.COLOR_BGR2RGB)
            
            # 归一化
            test_img_norm = test_img_rgb.astype(float) / 255.0
            local_img_norm = local_img_rgb.astype(float) / 255.0
            cyclegan_img_norm = cyclegan_img_rgb.astype(float) / 255.0
            
            try:
                # 计算局部风格指标
                print(f"  计算局部风格vs原始指标:")
                local_ssim = calculate_ssim(test_img_norm, local_img_norm)
                local_psnr = calculate_psnr(test_img_norm, local_img_norm)
                local_mse = calculate_mse(test_img_norm, local_img_norm)
                
                # 计算CycleGAN指标
                print(f"  计算CycleGAN vs原始指标:")
                cyclegan_ssim = calculate_ssim(test_img_norm, cyclegan_img_norm)
                cyclegan_psnr = calculate_psnr(test_img_norm, cyclegan_img_norm)
                cyclegan_mse = calculate_mse(test_img_norm, cyclegan_img_norm)
                
                # 保存结果
                comparison_results.append({
                    'image': test_img,
                    'local_style_ssim': local_ssim,
                    'local_style_psnr': local_psnr,
                    'local_style_mse': local_mse,
                    'cyclegan_ssim': cyclegan_ssim,
                    'cyclegan_psnr': cyclegan_psnr,
                    'cyclegan_mse': cyclegan_mse
                })
                
                print(f"  处理完成: 局部风格 SSIM={local_ssim:.4f}, PSNR={local_psnr:.4f}dB, MSE={local_mse:.8f}")
                print(f"  处理完成: CycleGAN SSIM={cyclegan_ssim:.4f}, PSNR={cyclegan_psnr:.4f}dB, MSE={cyclegan_mse:.8f}")
                
            except Exception as e:
                print(f"  计算图像 {test_img} 指标时出错: {str(e)}")
    
    # 计算平均值
    if comparison_results:
        print("\n成功处理的图像数量:", len(comparison_results))
        
        avg_local_ssim = sum(r['local_style_ssim'] for r in comparison_results) / len(comparison_results)
        avg_local_psnr = sum(r['local_style_psnr'] for r in comparison_results) / len(comparison_results)
        avg_local_mse = sum(r['local_style_mse'] for r in comparison_results) / len(comparison_results)
        
        avg_cyclegan_ssim = sum(r['cyclegan_ssim'] for r in comparison_results) / len(comparison_results)
        avg_cyclegan_psnr = sum(r['cyclegan_psnr'] for r in comparison_results) / len(comparison_results)
        avg_cyclegan_mse = sum(r['cyclegan_mse'] for r in comparison_results) / len(comparison_results)
        
        print(f"局部风格平均 SSIM: {avg_local_ssim:.4f}")
        print(f"局部风格平均 PSNR: {avg_local_psnr:.4f} dB")
        print(f"局部风格平均 MSE: {avg_local_mse:.8f}")
        
        print(f"CycleGAN平均 SSIM: {avg_cyclegan_ssim:.4f}")
        print(f"CycleGAN平均 PSNR: {avg_cyclegan_psnr:.4f} dB")
        print(f"CycleGAN平均 MSE: {avg_cyclegan_mse:.8f}")
        
        # 保存结果到文件
        if output_file:
            # 创建DataFrame
            df = pd.DataFrame({
                '图像': [r['image'] for r in comparison_results],
                '局部风格_SSIM': [r['local_style_ssim'] for r in comparison_results],
                '局部风格_PSNR': [r['local_style_psnr'] for r in comparison_results],
                '局部风格_MSE': [r['local_style_mse'] for r in comparison_results],
                'CycleGAN_SSIM': [r['cyclegan_ssim'] for r in comparison_results],
                'CycleGAN_PSNR': [r['cyclegan_psnr'] for r in comparison_results],
                'CycleGAN_MSE': [r['cyclegan_mse'] for r in comparison_results]
            })
            
            # 添加平均值行
            avg_df = pd.DataFrame({
                '图像': ['平均值'],
                '局部风格_SSIM': [avg_local_ssim],
                '局部风格_PSNR': [avg_local_psnr],
                '局部风格_MSE': [avg_local_mse],
                'CycleGAN_SSIM': [avg_cyclegan_ssim],
                'CycleGAN_PSNR': [avg_cyclegan_psnr],
                'CycleGAN_MSE': [avg_cyclegan_mse]
            })
            
            df = pd.concat([df, avg_df], ignore_index=True)
            
            # 保存到Excel
            print(f"\n保存结果到 {output_file}")
            df.to_excel(output_file, index=False)
            
            # 创建图表
            create_comparison_charts_with_test(avg_local_ssim, avg_local_psnr, avg_local_mse,
                                             avg_cyclegan_ssim, avg_cyclegan_psnr, avg_cyclegan_mse,
                                             os.path.splitext(output_file)[0] + "_charts.png")
            print(f"已创建图表: {os.path.splitext(output_file)[0] + '_charts.png'}")
        
        return {
            'avg_local_ssim': avg_local_ssim,
            'avg_local_psnr': avg_local_psnr,
            'avg_local_mse': avg_local_mse,
            'avg_cyclegan_ssim': avg_cyclegan_ssim,
            'avg_cyclegan_psnr': avg_cyclegan_psnr,
            'avg_cyclegan_mse': avg_cyclegan_mse,
            'details': comparison_results
        }
    else:
        print("警告: 没有找到可比较的图像")
        return None

def compare_image_folders(base_folder, comparison_folders, output_file=None):
    """比较基准文件夹与多个比较文件夹中的图像"""
    # 保存结果的数据结构
    results = []
    
    print(f"基准文件夹: {base_folder}")
    
    # 确保基准文件夹有图像
    base_path = Path(base_folder)
    base_image_count = len(list(base_path.glob("*.jpg"))) + len(list(base_path.glob("*.jpeg"))) + len(list(base_path.glob("*.png")))
    print(f"基准文件夹中包含 {base_image_count} 张图像")
    
    if base_image_count == 0:
        print(f"错误: 基准文件夹 {base_folder} 中没有图像!")
        return results
    
    # 对每个比较文件夹进行处理
    for comparison_folder in comparison_folders:
        print(f"\n比较文件夹: {comparison_folder}")
        
        # 跳过如果是同一个文件夹
        if os.path.normpath(base_folder) == os.path.normpath(comparison_folder):
            print(f"  跳过：无法将文件夹与自身比较")
            continue
        
        # 确保比较文件夹有图像
        comp_path = Path(comparison_folder)
        comp_image_count = len(list(comp_path.glob("*.jpg"))) + len(list(comp_path.glob("*.jpeg"))) + len(list(comp_path.glob("*.png")))
        print(f"比较文件夹中包含 {comp_image_count} 张图像")
        
        if comp_image_count == 0:
            print(f"错误: 比较文件夹 {comparison_folder} 中没有图像!")
            continue
        
        # 查找共有的图像
        common_images = find_matching_images(base_folder, comparison_folder)
        if not common_images:
            print(f"  未找到匹配的图像!")
            continue
        
        print(f"  找到 {len(common_images)} 张匹配的图像")
        
        # 计算每张图像的指标
        folder_results = {
            'folder': os.path.basename(comparison_folder),
            'images_count': len(common_images),
            'avg_ssim': 0.0,
            'avg_psnr': 0.0,
            'avg_mse': 0.0,
            'details': []
        }
        
        # 用于计算平均值的累计器
        total_ssim = 0.0
        total_psnr = 0.0
        total_mse = 0.0
        processed_count = 0
        
        # 处理每张图像
        for img_name in tqdm(common_images, desc="  处理图像"):
            base_img_path = os.path.join(base_folder, img_name)
            comparison_img_path = os.path.join(comparison_folder, img_name)
            
            print(f"\n处理图像对: \n  {base_img_path} \n  {comparison_img_path}")
            
            try:
                # 读取图像
                base_img = cv2.imread(base_img_path)
                comparison_img = cv2.imread(comparison_img_path)
                
                if base_img is None:
                    print(f"  无法读取基准图像: {base_img_path}")
                    continue
                    
                if comparison_img is None:
                    print(f"  无法读取比较图像: {comparison_img_path}")
                    continue
                
                print(f"  基准图像尺寸: {base_img.shape}")
                print(f"  比较图像尺寸: {comparison_img.shape}")
                
                # 确保两个图像大小相同
                if base_img.shape != comparison_img.shape:
                    print(f"  图像尺寸不同，正在调整大小...")
                    comparison_img = cv2.resize(comparison_img, (base_img.shape[1], base_img.shape[0]))
                    print(f"  调整后比较图像尺寸: {comparison_img.shape}")
                
                # 转换为RGB（OpenCV默认是BGR）
                base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
                comparison_img = cv2.cvtColor(comparison_img, cv2.COLOR_BGR2RGB)
                
                # 检查图像是否相同
                if np.array_equal(base_img, comparison_img):
                    print(f"  警告: 两个图像完全相同!")
                    continue
                
                # 归一化像素值到0-1范围用于SSIM
                base_img_norm = base_img.astype(float) / 255.0
                comparison_img_norm = comparison_img.astype(float) / 255.0
                
                # 计算指标
                ssim_value = calculate_ssim(base_img_norm, comparison_img_norm)
                psnr_value = calculate_psnr(base_img_norm, comparison_img_norm)
                mse_value = calculate_mse(base_img_norm, comparison_img_norm)
                
                # 检查得到的值是否有效
                if np.isnan(ssim_value) or np.isinf(ssim_value) or ssim_value == 0:
                    print(f"  警告: SSIM值无效 ({ssim_value})，跳过此图像")
                    continue
                    
                if np.isnan(psnr_value) or np.isinf(psnr_value) or psnr_value == 0:
                    print(f"  警告: PSNR值无效 ({psnr_value})，跳过此图像")
                    continue
                
                # 累加计算总和
                total_ssim += ssim_value
                total_psnr += psnr_value
                total_mse += mse_value
                processed_count += 1
                
                # 保存每张图片的结果
                folder_results['details'].append({
                    'image': img_name,
                    'ssim': ssim_value,
                    'psnr': psnr_value,
                    'mse': mse_value
                })
                
                print(f"  处理完成: SSIM={ssim_value:.4f}, PSNR={psnr_value:.4f}dB, MSE={mse_value:.8f}")
                
            except Exception as e:
                print(f"  处理图像 {img_name} 时出错: {str(e)}")
        
        # 计算平均值
        print(f"\n成功处理的图像数量: {processed_count}")
        if processed_count > 0:
            folder_results['avg_ssim'] = total_ssim / processed_count
            folder_results['avg_psnr'] = total_psnr / processed_count
            folder_results['avg_mse'] = total_mse / processed_count
            
            print(f"  平均 SSIM: {folder_results['avg_ssim']:.4f}")
            print(f"  平均 PSNR: {folder_results['avg_psnr']:.4f} dB")
            print(f"  平均 MSE: {folder_results['avg_mse']:.8f}")
        else:
            print(f"  警告: 没有成功处理任何图像，无法计算平均值")
        
        results.append(folder_results)
    
    # 保存结果到文件
    if output_file and results and any(len(r['details']) > 0 for r in results):
        # 创建一个数据表格
        summary_data = []
        for result in results:
            summary_data.append({
                '文件夹': result['folder'],
                '图像数量': result['images_count'],
                '平均SSIM': result['avg_ssim'],
                '平均PSNR': result['avg_psnr'],
                '平均MSE': result['avg_mse']
            })
        
        # 转换为DataFrame
        df = pd.DataFrame(summary_data)
        
        # 保存到Excel文件
        print(f"\n保存结果到 {output_file}")
        df.to_excel(output_file, index=False)
        
        # 创建可视化图表
        chart_file = os.path.splitext(output_file)[0] + "_charts.png"
        create_comparison_charts(results, chart_file)
        print(f"已创建图表: {chart_file}")
    else:
        print(f"\n警告: 没有有效的结果，不创建输出文件")
        
    return results

def create_comparison_charts_with_test(local_ssim, local_psnr, local_mse, 
                                      cyclegan_ssim, cyclegan_psnr, cyclegan_mse,
                                      output_file):
    """创建两种模型与原始图像比较的图表"""
    # 创建图表
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    models = ['局部风格增强', 'CycleGAN']
    ssim_values = [local_ssim, cyclegan_ssim]
    psnr_values = [local_psnr, cyclegan_psnr]
    mse_values = [local_mse, cyclegan_mse]
    
    # SSIM图表
    axes[0].bar(models, ssim_values, color=['blue', 'orange'])
    axes[0].set_title('平均SSIM (越高越好)')
    axes[0].set_ylim([0, 1])
    for i, v in enumerate(ssim_values):
        axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center')
    
    # PSNR图表
    axes[1].bar(models, psnr_values, color=['blue', 'orange'])
    axes[1].set_title('平均PSNR (越高越好)')
    for i, v in enumerate(psnr_values):
        axes[1].text(i, v + 0.5, f'{v:.2f}', ha='center')
    
    # MSE图表
    axes[2].bar(models, mse_values, color=['blue', 'orange'])
    axes[2].set_title('平均MSE (越低越好)')
    for i, v in enumerate(mse_values):
        axes[2].text(i, v + 0.001, f'{v:.6f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"图表已保存到: {output_file}")

def create_comparison_charts(results, output_file):
    """创建比较图表"""
    if not results or all(r['avg_ssim'] == 0 for r in results):
        print("警告: 没有有效数据来创建图表")
        return
    
    # 提取数据
    folders = [r['folder'] for r in results]
    ssim_values = [r['avg_ssim'] for r in results]
    psnr_values = [r['avg_psnr'] for r in results]
    mse_values = [r['avg_mse'] for r in results]
    
    # 创建图表
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    # SSIM图表
    axes[0].bar(folders, ssim_values, color='blue')
    axes[0].set_title('平均SSIM (越高越好)')
    axes[0].set_ylim([0, 1])
    for i, v in enumerate(ssim_values):
        axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center')
    
    # PSNR图表
    axes[1].bar(folders, psnr_values, color='green')
    axes[1].set_title('平均PSNR (越高越好)')
    for i, v in enumerate(psnr_values):
        axes[1].text(i, v + 0.5, f'{v:.2f}', ha='center')
    
    # MSE图表
    axes[2].bar(folders, mse_values, color='red')
    axes[2].set_title('平均MSE (越低越好)')
    for i, v in enumerate(mse_values):
        axes[2].text(i, v + 0.001, f'{v:.6f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"图表已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='比较不同文件夹中图像的质量指标')
    parser.add_argument('--base_folder', type=str, default='output/batch/local_style_enhanced_photo2monet',
                        help='基准文件夹路径')
    parser.add_argument('--comparison_folders', type=str, nargs='+',
                        help='比较文件夹路径(可以指定多个)')
    parser.add_argument('--output', type=str, default='image_comparison_results.xlsx',
                        help='输出结果文件名')
    parser.add_argument('--auto_detect', action='store_true',
                        help='自动检测output/batch下的所有相关文件夹进行比较')
    parser.add_argument('--with_test_images', action='store_true',
                        help='与原始测试图像进行比较')
    parser.add_argument('--test_images_folder', type=str, default='test_images',
                        help='测试图像文件夹路径')
    parser.add_argument('--debug', action='store_true',
                        help='启用详细调试信息')
    
    args = parser.parse_args()
    
    # 与测试图像比较模式
    if args.with_test_images:
        print("使用与原始测试图像比较模式")
        if not os.path.exists(args.test_images_folder):
            print(f"错误: 测试图像文件夹 {args.test_images_folder} 不存在!")
            return
        
        local_style_folder = args.base_folder
        cyclegan_folder = "output/batch/cyclegan_photo2monet"
        
        if args.comparison_folders and len(args.comparison_folders) > 0:
            cyclegan_folder = args.comparison_folders[0]
        
        if not os.path.exists(local_style_folder):
            print(f"错误: 局部风格文件夹 {local_style_folder} 不存在!")
            return
            
        if not os.path.exists(cyclegan_folder):
            print(f"错误: CycleGAN文件夹 {cyclegan_folder} 不存在!")
            return
        
        start_time = time.time()
        compare_with_test_images(local_style_folder, cyclegan_folder, args.test_images_folder, args.output)
        end_time = time.time()
        
        print(f"\n比较完成! 总耗时: {end_time - start_time:.2f}秒")
        return
    
    # 如果开启自动检测
    if args.auto_detect:
        base_folder = args.base_folder
        # 使用Path对象处理路径
        base_path = Path(base_folder)
        output_batch_dir = str(base_path.parent)
        
        # 使用Path.glob查找所有文件夹
        all_folders = [str(f) for f in Path(output_batch_dir).glob("*") if f.is_dir()]
        
        # 排除基准文件夹本身
        comparison_folders = [f for f in all_folders if f != base_folder]
        
        print(f"自动检测到以下比较文件夹:")
        for folder in comparison_folders:
            print(f"  - {folder}")
    else:
        base_folder = args.base_folder
        comparison_folders = args.comparison_folders
    
    if not os.path.exists(base_folder):
        print(f"错误: 基准文件夹 {base_folder} 不存在!")
        return
    
    if not comparison_folders:
        print("错误: 未指定比较文件夹!")
        return
    
    # 比较文件夹
    start_time = time.time()
    compare_image_folders(base_folder, comparison_folders, args.output)
    end_time = time.time()
    
    print(f"\n比较完成! 总耗时: {end_time - start_time:.2f}秒")

if __name__ == "__main__":
    main() 