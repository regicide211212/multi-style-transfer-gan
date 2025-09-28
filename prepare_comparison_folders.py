import os
import shutil
from pathlib import Path
import argparse

def prepare_folders(batch_dir, cyclegan_prefix='cyclegan_photo2monet_'):
    """
    整理批处理文件夹，将带有特定前缀的文件移动到对应子文件夹中
    
    参数:
    batch_dir: 批处理输出目录
    cyclegan_prefix: CycleGAN结果文件的前缀
    """
    batch_path = Path(batch_dir)
    
    if not batch_path.exists():
        print(f"错误: 目录 {batch_dir} 不存在!")
        return False
    
    # 创建CycleGAN输出目录
    cyclegan_dir = batch_path / "cyclegan_photo2monet"
    cyclegan_dir.mkdir(exist_ok=True)
    print(f"创建目录: {cyclegan_dir}")
    
    # 寻找所有cyclegan_photo2monet_前缀的文件
    count = 0
    for file_path in batch_path.glob(f"{cyclegan_prefix}*.jpg"):
        # 提取原始文件名（去掉前缀）
        original_name = file_path.name.replace(cyclegan_prefix, "")
        target_path = cyclegan_dir / original_name
        
        # 复制文件
        shutil.copy2(file_path, target_path)
        count += 1
    
    print(f"已复制 {count} 个文件到 {cyclegan_dir} 目录")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='准备图像比较所需的文件夹结构')
    parser.add_argument('--batch_dir', type=str, default='output/batch',
                        help='批处理输出目录')
    parser.add_argument('--cyclegan_prefix', type=str, default='cyclegan_photo2monet_',
                        help='CycleGAN输出文件前缀')
    
    args = parser.parse_args()
    
    prepare_folders(args.batch_dir, args.cyclegan_prefix)
    
    print("\n准备完成! 现在可以运行比较脚本:")
    print("python compare_image_quality.py --auto_detect")

if __name__ == "__main__":
    main() 