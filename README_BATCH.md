# 批量图像风格转换工具

这个批处理工具可以批量处理图像，应用CycleGAN或局部风格转换效果。

## 功能特点

- 支持CycleGAN和局部风格两种转换模式
- 局部风格模式支持简单、增强和高级三种处理方式
- 支持照片到莫奈风格和莫奈风格到照片两种转换方向
- 可以调整风格强度、细节保留级别等参数
- 自动批量处理指定目录下的所有图片

## 使用方法

### 基本用法

```bash
python batch_process_images.py
```

默认使用CycleGAN模式，处理test_images目录下的所有图片，结果保存到output/batch目录。

### 参数说明

```bash
python batch_process_images.py --help
```

可查看所有可用参数：

| 参数 | 说明 | 可选值 | 默认值 |
|------|------|--------|--------|
| --input_dir | 输入图像目录 | 任意有效路径 | test_images |
| --output_dir | 输出结果目录 | 任意有效路径 | output/batch |
| --mode | 处理模式 | cyclegan, local_style | cyclegan |
| --direction | 转换方向 | photo2monet, monet2photo | photo2monet |
| --local_style_mode | 局部风格模式 | simple, enhanced, advanced | enhanced |
| --strength | 风格强度 | 0-1之间的浮点数 | 0.8 |
| --detail | 细节保留水平 | 0-1之间的浮点数 | 0.7 |
| --enhance_colors | 是否增强颜色 | 无需值 | 默认启用 |
| --no_enhance_colors | 不增强颜色 | 无需值 | - |
| --smooth | 是否平滑过渡 | 无需值 | 默认启用 |
| --no_smooth | 不平滑过渡 | 无需值 | - |

### 示例

1. 使用CycleGAN将照片转为莫奈风格：

```bash
python batch_process_images.py --mode cyclegan --direction photo2monet
```

2. 使用局部风格高级模式将照片转为莫奈风格：

```bash
python batch_process_images.py --mode local_style --local_style_mode advanced --direction photo2monet
```

3. 使用局部风格简单模式，降低风格强度，提高细节保留：

```bash
python batch_process_images.py --mode local_style --local_style_mode simple --strength 0.6 --detail 0.9
```

4. 处理自定义目录的图像：

```bash
python batch_process_images.py --input_dir my_photos --output_dir my_results
```

## 处理模式说明

### CycleGAN模式

使用标准CycleGAN模型进行风格转换，适合完整的风格迁移。不支持参数调整，转换效果取决于模型训练情况。

### 局部风格模式

提供三种处理方式：

1. **简单模式（simple）**：直接线性混合原始图像和风格化图像，简单快速。
2. **增强模式（enhanced）**：智能识别图像不同区域，保留细节，特别处理天空区域，提供更自然的过渡效果。
3. **高级模式（advanced）**：使用多种高级算法，包括自适应颜色校正、边缘增强和色块平滑，提供最佳视觉效果。

## 注意事项

- 处理大量或大尺寸图片可能需要较长时间
- 确保models目录中有所需的模型文件
- 如果使用GPU处理，请确保有足够的显存 