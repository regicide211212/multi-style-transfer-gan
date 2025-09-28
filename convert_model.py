import torch
import argparse
import os

def convert_model(input_path, output_path):
    """将复杂模型文件转换为简单状态字典"""
    try:
        # 加载模型文件
        checkpoint = torch.load(input_path, map_location='cpu')
        
        # 检查模型格式
        if "G_AB_state_dict" in checkpoint:
            state_dict = checkpoint["G_AB_state_dict"]
        elif "G_BA_state_dict" in checkpoint:
            state_dict = checkpoint["G_BA_state_dict"]
        elif isinstance(checkpoint, dict) and "epoch" in checkpoint:
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                # 尝试删除不匹配的键
                state_dict = {}
                for k, v in checkpoint.items():
                    if k != "epoch" and not k.startswith("G_"):
                        state_dict[k] = v
        else:
            # 假设直接是状态字典
            state_dict = checkpoint
        
        # 保存简化后的模型
        torch.save(state_dict, output_path)
        print(f"模型已转换并保存至 {output_path}")
        return True
    except Exception as e:
        print(f"转换失败: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="转换复杂模型文件为简单状态字典")
    parser.add_argument("--input", type=str, required=True, help="输入模型文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出模型文件路径")
    
    args = parser.parse_args()
    convert_model(args.input, args.output) 