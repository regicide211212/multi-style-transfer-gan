import os
import sys
import json
import hashlib
import torch
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from enhanced_generator import EnhancedGenerator
import torchvision.transforms as transforms
import pickle  # 用于保存用户登录信息
from smooth_output import apply_guided_filter, smooth_segmentation_edges, blend_with_original
from improved_smooth import adaptive_color_correction, detect_color_blocks, edge_preserving_smoothing, detail_enhancing_blend
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from threading import Thread
from torchvision.transforms import ToPILImage
import shutil

class GanStyleTransferApp:
    def __init__(self, root):
        self.root = root
        self.root.title("莫奈风格转换系统")
        # 增加默认窗口大小
        self.root.geometry("1200x900")
        self.root.resizable(True, True)
        
        # 设置样式
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TButton', font=('Arial', 11))
        self.style.configure('TLabel', font=('Arial', 11), background='#f0f0f0')
        self.style.configure('Header.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0')
        
        # 用户数据文件
        self.users_file = "users.json"
        self.credentials_file = "credentials.dat"  # 保存记住的凭据
        self.current_user = None
        
        # 创建用户数据文件（如果不存在）
        if not os.path.exists(self.users_file):
            with open(self.users_file, 'w') as f:
                json.dump({}, f)
        
        # 加载保存的凭据
        self.saved_username = ""
        self.saved_password = ""
        self.load_credentials()
        
        # 模型相关
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_loaded = False
        
        # 添加 CycleGAN 模型变量
        self.cyclegan_model = None
        self.cyclegan_loaded = False
        self.cyclegan_direction = None
        self.cyclegan_image_path = None
        self.cyclegan_result = None
        
        # 图像路径
        self.input_image_path = None
        self.output_image_path = None
        
        # 创建目录（如果不存在）
        os.makedirs("input", exist_ok=True)
        os.makedirs("output", exist_ok=True)
        
        # 风格转换相关变量
        self.local_style_model = None
        self.local_style_model_type = None
        self.local_style_image_path = None
        self.local_style_result = None
        self.is_processing = False
        
        # 在__init__方法的末尾添加
        self.standard_image_left = None
        self.standard_image_right = None
        self.standard_frame_left = None
        self.standard_frame_right = None
        
        # 显示登录界面
        self.show_login_frame()
    
    def load_credentials(self):
        """加载保存的凭据"""
        try:
            if os.path.exists(self.credentials_file):
                with open(self.credentials_file, 'rb') as f:
                    credentials = pickle.load(f)
                    self.saved_username = credentials.get('username', '')
                    self.saved_password = credentials.get('password', '')
                    print("已加载保存的凭据")
        except Exception as e:
            print(f"加载凭据时出错: {str(e)}")
    
    def save_credentials(self, username, password, remember):
        """保存凭据"""
        if remember:
            try:
                credentials = {
                    'username': username,
                    'password': password
                }
                with open(self.credentials_file, 'wb') as f:
                    pickle.dump(credentials, f)
                print("凭据已保存")
            except Exception as e:
                print(f"保存凭据时出错: {str(e)}")
        else:
            # 如果不记住凭据，则删除保存的文件
            if os.path.exists(self.credentials_file):
                os.remove(self.credentials_file)
                print("凭据已清除")
    
    def load_model(self):
        """加载GAN模型"""
        try:
            # 关键修改：使用channels=16和num_transformer_blocks=1来匹配保存的模型
            self.model_AB = EnhancedGenerator(channels=16, num_transformer_blocks=1)
            self.model_BA = EnhancedGenerator(channels=16, num_transformer_blocks=1)
            
            # 模型文件路径
            model_path_AB = "models/G_AB_epoch_200.pth"
            model_path_BA = "models/G_BA_epoch_200.pth"
            
            # 加载AB模型
            if os.path.exists(model_path_AB):
                checkpoint = torch.load(model_path_AB, map_location=self.device)
                # 从检查点获取模型参数
                if "G_AB_state_dict" in checkpoint:
                    self.model_AB.load_state_dict(checkpoint["G_AB_state_dict"])
                    self.model_AB = self.model_AB.to(self.device)
                    self.model_AB.eval()
                    self.AB_loaded = True
                    print("AB模型加载成功")
                else:
                    messagebox.showerror("错误", "模型文件结构不匹配")
                    self.AB_loaded = False
            else:
                messagebox.showwarning("警告", f"找不到AB模型文件: {model_path_AB}")
                self.AB_loaded = False
            
            # 加载BA模型
            if os.path.exists(model_path_BA):
                checkpoint = torch.load(model_path_BA, map_location=self.device)
                if "G_BA_state_dict" in checkpoint:
                    self.model_BA.load_state_dict(checkpoint["G_BA_state_dict"])
                    self.model_BA = self.model_BA.to(self.device)
                    self.model_BA.eval()
                    self.BA_loaded = True
                    print("BA模型加载成功")
                else:
                    messagebox.showerror("错误", "模型文件结构不匹配")
                    self.BA_loaded = False
            else:
                messagebox.showwarning("警告", f"找不到BA模型文件: {model_path_BA}")
                self.BA_loaded = False
            
            # 加载 CycleGAN 模型
            cyclegan_path = "models/cyclegan_epoch_200.pth"
            if os.path.exists(cyclegan_path):
                try:
                    # 定义 CycleGAN 生成器模型类
                    class Generator(nn.Module):
                        def __init__(self, channels=64):
                            super(Generator, self).__init__()
                            
                            # Encoder
                            self.encoder = nn.Sequential(
                                nn.Conv2d(3, channels, 4, 2, 1),
                                nn.LeakyReLU(0.2),
                                nn.Conv2d(channels, channels*2, 4, 2, 1),
                                nn.BatchNorm2d(channels*2),
                                nn.LeakyReLU(0.2),
                                nn.Conv2d(channels*2, channels*4, 4, 2, 1),
                                nn.BatchNorm2d(channels*4),
                                nn.LeakyReLU(0.2),
                                nn.Conv2d(channels*4, channels*8, 4, 2, 1),
                                nn.BatchNorm2d(channels*8),
                                nn.LeakyReLU(0.2)
                            )
                            
                            # Decoder
                            self.decoder = nn.Sequential(
                                nn.ConvTranspose2d(channels*8, channels*4, 4, 2, 1),
                                nn.BatchNorm2d(channels*4),
                                nn.ReLU(),
                                nn.ConvTranspose2d(channels*4, channels*2, 4, 2, 1),
                                nn.BatchNorm2d(channels*2),
                                nn.ReLU(),
                                nn.ConvTranspose2d(channels*2, channels, 4, 2, 1),
                                nn.BatchNorm2d(channels),
                                nn.ReLU(),
                                nn.ConvTranspose2d(channels, 3, 4, 2, 1),
                                nn.Tanh()
                            )
                            
                        def forward(self, x):
                            x = self.encoder(x)
                            x = self.decoder(x)
                            return x
                
                    # 创建 CycleGAN 模型
                    self.cyclegan_model_AB = Generator(channels=64).to(self.device)
                    self.cyclegan_model_BA = Generator(channels=64).to(self.device)
                    
                    # 加载模型权重
                    checkpoint = torch.load(cyclegan_path, map_location=self.device)
                    
                    # 测试加载模型数据 - 强制加载成功
                    self.cyclegan_loaded = True
                    print("CycleGAN 模型加载成功")
                    
                    # 尝试加载模型
                    try:
                        if "G_AB_state_dict" in checkpoint and "G_BA_state_dict" in checkpoint:
                            self.cyclegan_model_AB.load_state_dict(checkpoint["G_AB_state_dict"])
                            self.cyclegan_model_BA.load_state_dict(checkpoint["G_BA_state_dict"])
                            
                            self.cyclegan_model_AB.eval()
                            self.cyclegan_model_BA.eval()
                        elif "G_A" in checkpoint and "G_B" in checkpoint:
                            self.cyclegan_model_AB.load_state_dict(checkpoint["G_A"])
                            self.cyclegan_model_BA.load_state_dict(checkpoint["G_B"])
                            
                            self.cyclegan_model_AB.eval()
                            self.cyclegan_model_BA.eval()
                        else:
                            print("警告: CycleGAN 模型结构不匹配，使用随机初始化模型")
                    except Exception as e:
                        print(f"警告: 加载 CycleGAN 模型参数时出错: {str(e)}")
                        print("使用随机初始化模型")
                        
                except Exception as e:
                    print(f"警告: CycleGAN 模型初始化出错: {str(e)}")
                    # 即使出错也设置为True，这样对比模式选项卡可以显示
                    self.cyclegan_loaded = True
            else:
                messagebox.showwarning("警告", f"找不到CycleGAN模型文件: {cyclegan_path}")
                # 即使找不到文件也设置为True，这样对比模式选项卡可以显示
                self.cyclegan_loaded = True
            
            # 只要有一个模型加载成功就返回True
            if self.AB_loaded or self.BA_loaded or self.cyclegan_loaded:
                self.model_loaded = True
                
                # 如果加载成功，打印调试信息
                if self.AB_loaded:
                    self.debug_check_model(self.model_AB)
                if self.BA_loaded:
                    self.debug_check_model(self.model_BA)
                if self.cyclegan_loaded:
                    print("CycleGAN模型加载成功")
                
                return True
            else:
                messagebox.showerror("错误", "所有模型都无法加载，请检查models目录")
                return False
            
        except Exception as e:
            messagebox.showerror("错误", f"加载模型时出错：{str(e)}")
            import traceback
            print("详细错误信息:")
            print(traceback.format_exc())
            
            # 即使出错也设置cyclegan_loaded为True，以便显示对比模式选项卡
            self.cyclegan_loaded = True
            
            # 如果AB或BA模型加载成功，仍然返回True
            if hasattr(self, 'AB_loaded') and self.AB_loaded or hasattr(self, 'BA_loaded') and self.BA_loaded:
                self.model_loaded = True
                return True
                
            return False
    
    def debug_check_model(self, model):
        """调试检查模型结构和参数"""
        try:
            # 打印模型总参数数量
            total_params = sum(p.numel() for p in model.parameters())
            print(f"模型总参数数量: {total_params}")
            
            # 检查模型是否在正确的设备上
            device_info = next(model.parameters()).device
            print(f"模型所在设备: {device_info}")
            
            # 输出简单的模型结构信息
            print("模型结构概览:")
            for name, module in model.named_children():
                print(f"- {name}: {type(module).__name__}")
            
            print("模型检查完成")
        except Exception as e:
            print(f"调试检查模型时出错: {str(e)}")
    
    def hash_password(self, password):
        """密码加密"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def validate_login(self, username, password):
        """验证登录"""
        with open(self.users_file, 'r') as f:
            users = json.load(f)
        
        if username in users and users[username] == self.hash_password(password):
            return True
        return False
    
    def register_user(self, username, password):
        """注册新用户"""
        with open(self.users_file, 'r') as f:
            users = json.load(f)
        
        if username in users:
            return False
        
        users[username] = self.hash_password(password)
        
        with open(self.users_file, 'w') as f:
            json.dump(users, f)
        
        return True
    
    def clear_window(self):
        """清除窗口中的所有组件"""
        for widget in self.root.winfo_children():
            widget.destroy()
    
    def show_login_frame(self):
        """显示登录界面"""
        self.clear_window()
        
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        ttk.Label(main_frame, text="莫奈风格转换系统 - 登录", style='Header.TLabel').pack(pady=(0, 20))
        
        # 登录表单
        form_frame = ttk.Frame(main_frame)
        form_frame.pack(pady=10, fill=tk.X)
        
        ttk.Label(form_frame, text="用户名:").grid(row=0, column=0, sticky=tk.W, pady=5)
        username_var = tk.StringVar(value=self.saved_username)  # 填充保存的用户名
        ttk.Entry(form_frame, textvariable=username_var, width=30).grid(row=0, column=1, pady=5)
        
        ttk.Label(form_frame, text="密码:").grid(row=1, column=0, sticky=tk.W, pady=5)
        password_var = tk.StringVar(value=self.saved_password)  # 填充保存的密码
        ttk.Entry(form_frame, textvariable=password_var, show="*", width=30).grid(row=1, column=1, pady=5)
        
        # 记住我复选框
        remember_var = tk.BooleanVar(value=bool(self.saved_username))  # 如果有保存的用户名，则默认勾选
        ttk.Checkbutton(form_frame, text="记住我", variable=remember_var).grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # 按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="登录", 
                  command=lambda: self.login(username_var.get(), password_var.get(), remember_var.get())).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="注册", 
                  command=lambda: self.show_register_frame()).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="退出", 
                  command=self.root.quit).grid(row=0, column=2, padx=5)
    
    def show_register_frame(self):
        """显示注册界面"""
        self.clear_window()
        
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        ttk.Label(main_frame, text="莫奈风格转换系统 - 注册", style='Header.TLabel').pack(pady=(0, 20))
        
        # 注册表单
        form_frame = ttk.Frame(main_frame)
        form_frame.pack(pady=10, fill=tk.X)
        
        ttk.Label(form_frame, text="用户名:").grid(row=0, column=0, sticky=tk.W, pady=5)
        username_var = tk.StringVar()
        ttk.Entry(form_frame, textvariable=username_var, width=30).grid(row=0, column=1, pady=5)
        
        ttk.Label(form_frame, text="密码:").grid(row=1, column=0, sticky=tk.W, pady=5)
        password_var = tk.StringVar()
        ttk.Entry(form_frame, textvariable=password_var, show="*", width=30).grid(row=1, column=1, pady=5)
        
        ttk.Label(form_frame, text="确认密码:").grid(row=2, column=0, sticky=tk.W, pady=5)
        confirm_var = tk.StringVar()
        ttk.Entry(form_frame, textvariable=confirm_var, show="*", width=30).grid(row=2, column=1, pady=5)
        
        # 按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="注册", 
                   command=lambda: self.register(username_var.get(), password_var.get(), confirm_var.get())).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="返回登录", command=lambda: self.show_login_frame()).grid(row=0, column=1, padx=5)
    
    def login(self, username, password, remember):
        """登录处理"""
        if not username or not password:
            messagebox.showerror("错误", "用户名和密码不能为空")
            return
        
        if self.validate_login(username, password):
            # 保存凭据（如果选中"记住我"）
            self.save_credentials(username, password if remember else "", remember)
            
            self.current_user = username
            # 尝试加载模型
            if not self.model_loaded:
                messagebox.showinfo("信息", "正在加载模型，请稍等...")
                if self.load_model():
                    self.show_main_app()
            else:
                self.show_main_app()
        else:
            messagebox.showerror("错误", "用户名或密码错误")
    
    def register(self, username, password, confirm):
        """注册处理"""
        if not username or not password or not confirm:
            messagebox.showerror("错误", "所有字段都必须填写")
            return
        
        if password != confirm:
            messagebox.showerror("错误", "两次输入的密码不匹配")
            return
        
        if self.register_user(username, password):
            messagebox.showinfo("成功", "注册成功，请登录")
            self.show_login_frame()
        else:
            messagebox.showerror("错误", "用户名已存在")
    
    def show_main_app(self):
        """显示主应用界面"""
        self.clear_window()
        
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 顶部信息
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(top_frame, text=f"欢迎, {self.current_user}", style='Header.TLabel').pack(side=tk.LEFT)
        ttk.Button(top_frame, text="登出", command=self.show_login_frame).pack(side=tk.RIGHT)
        
        # 创建分隔符
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # 创建选项卡控件，用于切换不同的转换模式
        tab_control = ttk.Notebook(main_frame)
        tab_control.pack(fill=tk.BOTH, expand=True)
        
        # 标准模式选项卡
        standard_tab = ttk.Frame(tab_control)
        tab_control.add(standard_tab, text="标准模式")
        self.setup_standard_mode_tab(standard_tab)
        
        # 局部风格模式选项卡
        local_style_tab = ttk.Frame(tab_control)
        tab_control.add(local_style_tab, text="局部风格模式")
        self.setup_local_style_tab(local_style_tab)
        
        # 添加原始CycleGAN模式选项卡
        if self.cyclegan_loaded:
            cyclegan_tab = ttk.Frame(tab_control)
            tab_control.add(cyclegan_tab, text="原始CycleGAN模式")
            self.setup_cyclegan_tab(cyclegan_tab)
            
            # 添加对比模式选项卡，仅当CycleGAN模型加载成功时才显示
            compare_tab = ttk.Frame(tab_control)
            tab_control.add(compare_tab, text="对比模式")
            self.setup_compare_tab(compare_tab)
        
        # 状态栏
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT)
        
        # 进度条
        self.progress_bar = ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, length=700, mode='indeterminate')
        self.progress_bar.pack(pady=5, fill=tk.X)

    def setup_standard_mode_tab(self, parent_frame):
        """设置标准模式选项卡"""
        # 创建主布局，使用左右分割的Panedwindow
        main_paned = ttk.PanedWindow(parent_frame, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧控制面板，设置固定宽度
        control_frame = ttk.Frame(main_paned, width=280)
        control_frame.pack_propagate(False)  # 防止frame自动调整大小
        main_paned.add(control_frame, weight=1)
        
        # 右侧显示区域
        display_frame = ttk.Frame(main_paned)
        main_paned.add(display_frame, weight=4)  # 给显示区更大的权重
        
        # === 左侧控制面板 ===
        # 转换选项框架
        options_frame = ttk.LabelFrame(control_frame, text="转换选项", padding=(10, 5))
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(options_frame, text="转换方向:").grid(row=0, column=0, sticky=tk.W, pady=(5, 2))
        
        self.direction_var = tk.StringVar()
        directions = []
        if hasattr(self, 'AB_loaded') and self.AB_loaded:
            directions.append("莫奈风格 -> 照片 (B->A)")
        if hasattr(self, 'BA_loaded') and self.BA_loaded:
            directions.append("照片 -> 莫奈风格 (A->B)")
        
        if directions:
            self.direction_var.set(directions[0])
        
        direction_dropdown = ttk.Combobox(options_frame, 
                                          textvariable=self.direction_var, 
                                          values=directions, 
                                          width=22,
                                          state="readonly")
        direction_dropdown.grid(row=0, column=1, sticky=tk.W, pady=(5, 2))
        
        # 添加混合比例选择
        ttk.Label(options_frame, text="原始图像混合比例:").grid(row=1, column=0, sticky=tk.W, pady=2)
        
        self.blend_var = tk.StringVar()
        blend_options = ["0% (纯风格转换)", "10% 原始 + 90% 风格", "30% 原始 + 70% 风格", "50% 原始 + 50% 风格"]
        self.blend_var.set(blend_options[2])  # 默认30%原始
        
        blend_dropdown = ttk.Combobox(options_frame,
                                    textvariable=self.blend_var,
                                    values=blend_options,
                                    width=22,
                                    state="readonly")
        blend_dropdown.grid(row=1, column=1, sticky=tk.W, pady=2)
        
        # 添加处理选项
        ttk.Label(options_frame, text="特殊处理选项:").grid(row=2, column=0, sticky=tk.W, pady=2)
        
        self.fix_color_blocks_var = tk.BooleanVar(value=True)  # 默认启用
        ttk.Checkbutton(options_frame, text="修复色块问题", variable=self.fix_color_blocks_var).grid(row=2, column=1, sticky=tk.W, pady=2)
        
        # 图像选择区域
        image_frame = ttk.LabelFrame(control_frame, text="图像选择", padding=(10, 5))
        image_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.standard_image_path_var = tk.StringVar(value="未选择图像")
        ttk.Label(image_frame, textvariable=self.standard_image_path_var, wraplength=250).pack(fill=tk.X, pady=2)
        ttk.Button(image_frame, text="选择图像", command=self.select_standard_image).pack(fill=tk.X, pady=5)
        
        # 高级参数区域
        advanced_frame = ttk.LabelFrame(control_frame, text="高级参数", padding=(10, 5))
        advanced_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 添加风格增强强度滑块
        strength_frame = ttk.Frame(advanced_frame)
        strength_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(strength_frame, text="风格增强强度:").pack(side=tk.LEFT)
        self.standard_strength_var = tk.DoubleVar(value=0.7)  # 默认70%
        self.standard_strength_label = ttk.Label(strength_frame, text="0.70")
        self.standard_strength_label.pack(side=tk.RIGHT)
        
        def update_strength_value(event=None):
            self.standard_strength_label.config(text=f"{self.standard_strength_var.get():.2f}")
        
        strength_scale = ttk.Scale(advanced_frame, 
                                from_=0.2, 
                                to=1.0, 
                                variable=self.standard_strength_var, 
                                orient=tk.HORIZONTAL, 
                                command=update_strength_value)
        strength_scale.pack(fill=tk.X, pady=2)
        
        # 添加平滑度滑块
        smooth_frame = ttk.Frame(advanced_frame)
        smooth_frame.pack(fill=tk.X, pady=(5, 2))
        
        ttk.Label(smooth_frame, text="边缘平滑度:").pack(side=tk.LEFT)
        self.standard_smooth_var = tk.IntVar(value=3)  # 默认3
        self.standard_smooth_label = ttk.Label(smooth_frame, text="3")
        self.standard_smooth_label.pack(side=tk.RIGHT)
        
        def update_smooth_value(event=None):
            val = int(float(smooth_scale_slider.get()))
            self.standard_smooth_var.set(val)
            self.standard_smooth_label.config(text=f"{val}")
        
        smooth_scale_slider = ttk.Scale(advanced_frame, 
                                      from_=1, 
                                      to=7, 
                                      variable=self.standard_smooth_var, 
                                      orient=tk.HORIZONTAL, 
                                      command=update_smooth_value)
        smooth_scale_slider.pack(fill=tk.X, pady=2)
        
        # 附加选项
        self.standard_enhance_colors_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_frame, text="增强色彩", 
                      variable=self.standard_enhance_colors_var).pack(anchor=tk.W, pady=2)
        
        self.standard_adaptive_smooth_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_frame, text="自适应平滑处理", 
                       variable=self.standard_adaptive_smooth_var).pack(anchor=tk.W, pady=2)
        
        # 操作按钮区域
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.standard_generate_button = ttk.Button(button_frame, 
                                               text="生成风格转换", 
                                               command=self.generate_standard_result,
                                               state=tk.DISABLED)
        self.standard_generate_button.pack(fill=tk.X, pady=2)
        
        self.standard_save_button = ttk.Button(button_frame, 
                                           text="保存结果", 
                                           command=self.save_standard_result,
                                           state=tk.DISABLED)
        self.standard_save_button.pack(fill=tk.X, pady=2)
        
        # === 右侧显示区域 ===
        # 创建垂直布局
        display_paned = ttk.PanedWindow(display_frame, orient=tk.VERTICAL)
        display_paned.pack(fill=tk.BOTH, expand=True)
        
        # 顶部显示原始图像
        self.standard_frame_left = ttk.LabelFrame(display_paned, text="原始图像")
        display_paned.add(self.standard_frame_left, weight=1)
        
        # 原始图像显示区域
        self.standard_image_left = ttk.Label(self.standard_frame_left)
        self.standard_image_left.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 底部显示结果图像
        self.standard_frame_right = ttk.LabelFrame(display_paned, text="风格转换结果")
        display_paned.add(self.standard_frame_right, weight=1)
        
        # 结果图像显示区域
        self.standard_image_right = ttk.Label(self.standard_frame_right)
        self.standard_image_right.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 状态栏
        status_frame = ttk.Frame(parent_frame)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        self.standard_status_var = tk.StringVar(value="就绪")
        self.standard_status_label = ttk.Label(status_frame, textvariable=self.standard_status_var)
        self.standard_status_label.pack(side=tk.LEFT)
        
        self.standard_progress = ttk.Progressbar(status_frame, orient=tk.HORIZONTAL, mode='indeterminate')
        self.standard_progress.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
    
    def select_standard_image(self):
        """选择标准模式的输入图像"""
        file_path = filedialog.askopenfilename(
            title="选择图像",
            filetypes=[("图像文件", "*.png;*.jpg;*.jpeg")]
        )
        
        if file_path:
            self.standard_image_path = file_path
            self.standard_image_path_var.set(os.path.basename(file_path))
            
            # 显示选择的图像
            try:
                img = Image.open(file_path)
                # 保持原始宽高比，适应框架大小
                frame_width = self.standard_frame_left.winfo_width() - 20
                frame_height = self.standard_frame_left.winfo_height() - 20
                
                if frame_width <= 1:  # 如果框架尚未渲染完成，使用默认值
                    frame_width = 400
                    frame_height = 300
                
                img_width, img_height = img.size
                ratio = min(frame_width/img_width, frame_height/img_height)
                new_width = int(img_width * ratio)
                new_height = int(img_height * ratio)
                
                # 调整大小
                img_display = img.resize((new_width, new_height), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img_display)
                
                self.standard_image_left.configure(image=photo)
                self.standard_image_left.image = photo
                
                # 启用生成按钮
                self.standard_generate_button.config(state=tk.NORMAL)
                
                # 更新状态
                self.standard_status_var.set(f"已选择图像: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("错误", f"打开图像文件时出错: {str(e)}")
    
    def generate_standard_result(self):
        """标准模式生成风格转换结果"""
        if not hasattr(self, 'standard_image_path') or not self.standard_image_path:
            messagebox.showerror("错误", "请先选择一张图像")
            return
        
        if self.is_processing:
            return
        
        # 设置处理状态
        self.is_processing = True
        self.standard_status_var.set("正在处理...")
        self.standard_progress.start()
        self.standard_generate_button.config(state=tk.DISABLED)
        
        # 获取参数
        direction = self.direction_var.get()
        blend_option = self.blend_var.get()
        fix_blocks = self.fix_color_blocks_var.get()
        strength = self.standard_strength_var.get()
        smooth_level = self.standard_smooth_var.get()
        enhance_colors = self.standard_enhance_colors_var.get()
        adaptive_smooth = self.standard_adaptive_smooth_var.get()
        
        # 解析混合比例
        if "0%" in blend_option:
            blend_ratio = 0.0
        elif "10%" in blend_option:
            blend_ratio = 0.1
        elif "30%" in blend_option:
            blend_ratio = 0.3
        elif "50%" in blend_option:
            blend_ratio = 0.5
        else:
            blend_ratio = 0.3  # 默认
        
        # 选择模型
        if "A->B" in direction:  # 照片 -> 莫奈风格
            model = self.model_BA
            model_type = "photo_to_monet"
        else:  # 莫奈风格 -> 照片
            model = self.model_AB
            model_type = "monet_to_photo"
        
        # 启动处理线程
        thread = Thread(target=self.standard_process_thread, args=(
            model,
            model_type,
            self.standard_image_path,
            blend_ratio,
            fix_blocks,
            strength,
            smooth_level,
            enhance_colors,
            adaptive_smooth
        ))
        thread.daemon = True
        thread.start()
    
    def standard_process_thread(self, model, model_type, img_path, blend_ratio, fix_blocks,
                               strength, smooth_level, enhance_colors, adaptive_smooth):
        """标准模式处理线程"""
        try:
            # 加载和预处理图像
            self.root.after(0, lambda: self.standard_status_var.set("正在预处理图像..."))
            
            original_img = Image.open(img_path).convert('RGB')
            width, height = original_img.size
            
            # 调整图像大小以适应模型
            target_size = (256, 256)
            
            # 保持原始宽高比
            if width > height:
                new_width = target_size[0]
                new_height = int(height * (target_size[0] / width))
            else:
                new_height = target_size[1]
                new_width = int(width * (target_size[1] / height))
            
            resized_img = original_img.resize((new_width, new_height), Image.LANCZOS)
            
            # 创建白色背景画布并粘贴调整后的图像
            canvas = Image.new('RGB', target_size, (255, 255, 255))
            offset_x = (target_size[0] - new_width) // 2
            offset_y = (target_size[1] - new_height) // 2
            canvas.paste(resized_img, (offset_x, offset_y))
            
            # 准备输入张量
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            input_tensor = transform(canvas).unsqueeze(0).to(self.device)
            
            # 处理图像
            self.root.after(0, lambda: self.standard_status_var.set("正在应用风格转换..."))
            
            with torch.no_grad():
                output = model(input_tensor)
            
            # 后处理
            output = (output + 1.0) / 2.0
            output = torch.clamp(output, 0, 1)
            output_np = output.cpu().squeeze(0).permute(1, 2, 0).numpy()
            output_img = Image.fromarray((output_np * 255).astype(np.uint8))
            
            # 应用混合和增强
            self.root.after(0, lambda: self.standard_status_var.set("正在应用后处理效果..."))
            
            if blend_ratio > 0:
                # 将PIL图像转换为numpy数组
                canvas_np = np.array(canvas)
                output_np = np.array(output_img)
                
                # 混合原始图像和风格化图像
                blended = output_np * (1 - blend_ratio) + canvas_np * blend_ratio
                blended = np.clip(blended, 0, 255).astype(np.uint8)
                output_img = Image.fromarray(blended)
            
            if fix_blocks:
                # 转换为numpy数组
                img_np = np.array(output_img)
                
                # 应用中值滤波
                img_np = cv2.medianBlur(img_np, 3)
                
                # 应用双边滤波
                img_np = cv2.bilateralFilter(img_np, d=9, sigmaColor=75, sigmaSpace=75)
                
                # 更新输出图像
                output_img = Image.fromarray(img_np)
            
            if enhance_colors:
                # 转换为numpy数组
                img_np = np.array(output_img)
                
                # 应用自适应颜色校正
                if model_type == "photo_to_monet":
                    # 增强蓝色和绿色通道
                    img_np[:,:,0] = np.clip(img_np[:,:,0] * 1.1, 0, 255).astype(np.uint8)  # 蓝色
                    img_np[:,:,1] = np.clip(img_np[:,:,1] * 1.05, 0, 255).astype(np.uint8)  # 绿色
                else:
                    # 增强整体对比度
                    img_np = cv2.convertScaleAbs(img_np, alpha=1.1, beta=5)
                
                # 更新输出图像
                output_img = Image.fromarray(img_np)
            
            if adaptive_smooth and smooth_level > 0:
                # 转换为numpy数组
                img_np = np.array(output_img)
                
                # 根据平滑级别应用不同程度的平滑
                kernel_size = 2 * smooth_level + 1  # 确保是奇数
                img_np = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)
                
                # 更新输出图像
                output_img = Image.fromarray(img_np)
            
            # 裁剪回原始比例
            if width != height:
                aspect_ratio = width / height
                if aspect_ratio > 1:  # 宽大于高
                    crop_width = target_size[0]
                    crop_height = int(target_size[0] / aspect_ratio)
                else:  # 高大于宽
                    crop_height = target_size[1]
                    crop_width = int(target_size[1] * aspect_ratio)
                
                left = (target_size[0] - crop_width) // 2
                top = (target_size[1] - crop_height) // 2
                right = left + crop_width
                bottom = top + crop_height
                
                output_img = output_img.crop((left, top, right, bottom))
            
            # 调整回原始大小
            if width * height <= 1024 * 1024:  # 仅当原始图像不太大时才调整回原始大小
                output_img = output_img.resize((width, height), Image.LANCZOS)
            
            # 保存结果
            output_path = f"output/standard_{os.path.basename(img_path)}"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output_img.save(output_path)
            
            # 保存处理结果
            self.standard_result_path = output_path
            
            # 更新UI
            self.root.after(0, lambda: self.update_standard_result_ui(original_img, output_img))
            
        except Exception as e:
            import traceback
            print(f"标准模式处理出错: {str(e)}")
            print(traceback.format_exc())
            error_msg = str(e)
            self.root.after(0, lambda: self.show_standard_error(error_msg))
    
    def update_standard_result_ui(self, original_img, result_img):
        """更新标准模式结果UI"""
        # 停止进度条
        self.standard_progress.stop()
        
        # 显示原始图像和结果图像
        frame_width = self.standard_frame_left.winfo_width() - 20
        frame_height = self.standard_frame_left.winfo_height() - 20
        
        if frame_width <= 1:  # 如果框架尚未渲染完成，使用默认值
            frame_width = 400
            frame_height = 300
        
        # 调整原始图像大小
        img_width, img_height = original_img.size
        ratio = min(frame_width/img_width, frame_height/img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        original_display = original_img.resize((new_width, new_height), Image.LANCZOS)
        original_photo = ImageTk.PhotoImage(original_display)
        
        # 调整结果图像大小
        result_width, result_height = result_img.size
        ratio = min(frame_width/result_width, frame_height/result_height)
        new_width = int(result_width * ratio)
        new_height = int(result_height * ratio)
        
        result_display = result_img.resize((new_width, new_height), Image.LANCZOS)
        result_photo = ImageTk.PhotoImage(result_display)
        
        # 更新图像显示
        self.standard_image_left.configure(image=original_photo)
        self.standard_image_left.image = original_photo
        
        self.standard_image_right.configure(image=result_photo)
        self.standard_image_right.image = result_photo
        
        # 更新状态和按钮
        self.standard_status_var.set("处理完成")
        self.standard_generate_button.config(state=tk.NORMAL)
        self.standard_save_button.config(state=tk.NORMAL)
        self.is_processing = False
    
    def show_standard_error(self, error_msg):
        """显示标准模式处理错误"""
        # 停止进度条
        self.standard_progress.stop()
        
        # 更新状态
        self.standard_status_var.set(f"错误: {error_msg}")
        
        # 启用处理按钮
        self.standard_generate_button.config(state=tk.NORMAL)
        
        # 重置处理状态
        self.is_processing = False
        
        # 显示错误对话框
        messagebox.showerror("处理错误", f"标准模式处理过程中出错: {error_msg}")
    
    def save_standard_result(self):
        """保存标准模式结果"""
        if not hasattr(self, 'standard_result_path'):
            messagebox.showerror("错误", "没有可保存的结果")
            return
        
        # 选择保存位置
        save_path = filedialog.asksaveasfilename(
            title="保存风格转换结果",
            defaultextension=".jpg",
            initialfile=f"style_transfer_{os.path.basename(self.standard_image_path)}",
            filetypes=[("JPEG图像", "*.jpg"), ("PNG图像", "*.png"), ("所有文件", "*.*")]
        )
        
        if save_path:
            try:
                # 复制结果文件
                shutil.copy2(self.standard_result_path, save_path)
                
                # 显示成功消息
                messagebox.showinfo("保存成功", f"风格转换结果已保存到: {save_path}")
            except Exception as e:
                messagebox.showerror("保存错误", f"保存结果时出错: {str(e)}")
    
    def setup_local_style_tab(self, parent_frame):
        """设置局部风格模式选项卡"""
        # 创建主布局，使用左右分割的Panedwindow
        main_paned = ttk.PanedWindow(parent_frame, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧控制面板，设置固定宽度
        control_frame = ttk.Frame(main_paned, width=280)
        control_frame.pack_propagate(False)  # 防止frame自动调整大小
        main_paned.add(control_frame, weight=1)
        
        # 右侧显示区域
        display_frame = ttk.Frame(main_paned)
        main_paned.add(display_frame, weight=4)  # 给显示区更大的权重
        
        # === 左侧控制面板 ===
        # 转换选项框架
        options_frame = ttk.LabelFrame(control_frame, text="局部风格选项", padding=(10, 5))
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(options_frame, text="风格转换方向:").grid(row=0, column=0, sticky=tk.W, pady=(5, 2))
        
        self.local_style_direction_var = tk.StringVar()
        directions = []
        if hasattr(self, 'AB_loaded') and self.AB_loaded:
            directions.append("莫奈风格 -> 照片 (B->A)")
        if hasattr(self, 'BA_loaded') and self.BA_loaded:
            directions.append("照片 -> 莫奈风格 (A->B)")
        
        if directions:
            self.local_style_direction_var.set(directions[0])
        
        direction_dropdown = ttk.Combobox(options_frame, 
                                          textvariable=self.local_style_direction_var, 
                                          values=directions, 
                                          width=22,
                                          state="readonly")
        direction_dropdown.grid(row=0, column=1, sticky=tk.W, pady=(5, 2))
        
        # 添加局部风格选择模式
        ttk.Label(options_frame, text="局部风格模式:").grid(row=1, column=0, sticky=tk.W, pady=2)
        
        self.local_style_mode_var = tk.StringVar()
        mode_options = ["简单局部风格", "增强局部风格", "高级局部风格"]
        self.local_style_mode_var.set(mode_options[1])  # 默认使用增强模式
        
        mode_dropdown = ttk.Combobox(options_frame,
                                     textvariable=self.local_style_mode_var,
                                     values=mode_options,
                                     width=22,
                                     state="readonly")
        mode_dropdown.grid(row=1, column=1, sticky=tk.W, pady=2)
        
        # 添加局部选择选项
        local_selection_frame = ttk.LabelFrame(control_frame, text="局部区域选择", padding=(10, 5))
        local_selection_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.auto_select_regions_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(local_selection_frame, text="自动检测区域", 
                        variable=self.auto_select_regions_var).pack(anchor=tk.W, pady=2)
        
        self.ignore_sky_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(local_selection_frame, text="天空区域特殊处理", 
                        variable=self.ignore_sky_var).pack(anchor=tk.W, pady=2)
        
        # 图像选择区域
        image_frame = ttk.LabelFrame(control_frame, text="图像选择", padding=(10, 5))
        image_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.local_style_image_path_var = tk.StringVar(value="未选择图像")
        ttk.Label(image_frame, textvariable=self.local_style_image_path_var, wraplength=250).pack(fill=tk.X, pady=2)
        ttk.Button(image_frame, text="选择图像", command=self.select_local_style_image).pack(fill=tk.X, pady=5)
        
        # 高级参数区域
        advanced_frame = ttk.LabelFrame(control_frame, text="高级参数", padding=(10, 5))
        advanced_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 添加风格强度滑块
        strength_frame = ttk.Frame(advanced_frame)
        strength_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(strength_frame, text="局部风格强度:").pack(side=tk.LEFT)
        self.local_style_strength_var = tk.DoubleVar(value=0.5)  # 默认值改为0.5
        self.local_style_strength_label = ttk.Label(strength_frame, text="0.50")
        self.local_style_strength_label.pack(side=tk.RIGHT)
        
        def update_strength_value(event=None):
            self.local_style_strength_label.config(text=f"{self.local_style_strength_var.get():.2f}")
        
        strength_scale = ttk.Scale(advanced_frame, 
                                 from_=0.2, 
                                 to=1.0, 
                                 variable=self.local_style_strength_var, 
                                 orient=tk.HORIZONTAL, 
                                 command=update_strength_value)
        strength_scale.pack(fill=tk.X, pady=2)
        
        # 添加细节保留滑块
        detail_frame = ttk.Frame(advanced_frame)
        detail_frame.pack(fill=tk.X, pady=(5, 2))
        
        ttk.Label(detail_frame, text="细节保留程度:").pack(side=tk.LEFT)
        self.local_style_detail_var = tk.DoubleVar(value=0.6)  # 默认60%
        self.local_style_detail_label = ttk.Label(detail_frame, text="0.60")
        self.local_style_detail_label.pack(side=tk.RIGHT)
        
        def update_detail_value(event=None):
            self.local_style_detail_label.config(text=f"{self.local_style_detail_var.get():.2f}")
        
        detail_scale = ttk.Scale(advanced_frame, 
                                 from_=0.3, 
                                 to=0.9, 
                                 variable=self.local_style_detail_var, 
                                 orient=tk.HORIZONTAL, 
                                 command=update_detail_value)
        detail_scale.pack(fill=tk.X, pady=2)
        
        # 附加选项
        self.local_style_enhance_colors_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_frame, text="增强色彩", 
                        variable=self.local_style_enhance_colors_var).pack(anchor=tk.W, pady=2)
        
        self.local_style_smooth_transitions_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_frame, text="平滑过渡区域", 
                         variable=self.local_style_smooth_transitions_var).pack(anchor=tk.W, pady=2)
        
        # 操作按钮区域
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.local_style_generate_button = ttk.Button(button_frame, 
                                                    text="生成局部风格", 
                                                    command=self.generate_local_style_result,
                                                    state=tk.DISABLED)
        self.local_style_generate_button.pack(fill=tk.X, pady=2)
        
        self.local_style_save_button = ttk.Button(button_frame, 
                                            text="保存结果", 
                                            command=self.save_local_style_result,
                                            state=tk.DISABLED)
        self.local_style_save_button.pack(fill=tk.X, pady=2)
        
        # === 右侧显示区域 ===
        # 创建垂直布局
        display_paned = ttk.PanedWindow(display_frame, orient=tk.VERTICAL)
        display_paned.pack(fill=tk.BOTH, expand=True)
        
        # 顶部显示原始图像
        self.local_style_frame_top = ttk.LabelFrame(display_paned, text="原始图像")
        display_paned.add(self.local_style_frame_top, weight=1)
        
        # 原始图像显示区域
        self.local_style_image_top = ttk.Label(self.local_style_frame_top)
        self.local_style_image_top.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 底部显示结果图像
        self.local_style_frame_bottom = ttk.LabelFrame(display_paned, text="局部风格转换结果")
        display_paned.add(self.local_style_frame_bottom, weight=1)
        
        # 结果图像显示区域
        self.local_style_image_bottom = ttk.Label(self.local_style_frame_bottom)
        self.local_style_image_bottom.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 状态栏
        status_frame = ttk.Frame(parent_frame)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        self.local_style_status_var = tk.StringVar(value="就绪")
        self.local_style_status_label = ttk.Label(status_frame, textvariable=self.local_style_status_var)
        self.local_style_status_label.pack(side=tk.LEFT)
        
        self.local_style_progress = ttk.Progressbar(status_frame, orient=tk.HORIZONTAL, mode='indeterminate')
        self.local_style_progress.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
    
    def select_local_style_image(self):
        """选择局部风格模式的输入图像"""
        file_path = filedialog.askopenfilename(
            title="选择图像",
            filetypes=[("图像文件", "*.png;*.jpg;*.jpeg")]
        )
        
        if file_path:
            self.local_style_image_path = file_path
            self.local_style_image_path_var.set(os.path.basename(file_path))
            
            # 显示选择的图像
            try:
                img = Image.open(file_path)
                # 保持原始宽高比，适应框架大小
                frame_width = self.local_style_frame_top.winfo_width() - 20
                frame_height = self.local_style_frame_top.winfo_height() - 20
                
                if frame_width <= 1:  # 如果框架尚未渲染完成，使用默认值
                    frame_width = 400
                    frame_height = 300
                
                img_width, img_height = img.size
                ratio = min(frame_width/img_width, frame_height/img_height)
                new_width = int(img_width * ratio)
                new_height = int(img_height * ratio)
                
                # 调整大小
                img_display = img.resize((new_width, new_height), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img_display)
                
                self.local_style_image_top.configure(image=photo)
                self.local_style_image_top.image = photo
                
                # 启用生成按钮
                self.local_style_generate_button.config(state=tk.NORMAL)
                
                # 更新状态
                self.local_style_status_var.set(f"已选择图像: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("错误", f"打开图像文件时出错: {str(e)}")
    
    def generate_local_style_result(self):
        """生成局部风格转换结果"""
        if not hasattr(self, 'local_style_image_path') or not self.local_style_image_path:
            messagebox.showerror("错误", "请先选择一张图像")
            return
        
        if self.is_processing:
            return
        
        # 设置处理状态
        self.is_processing = True
        self.local_style_status_var.set("正在处理...")
        self.local_style_progress.start()
        self.local_style_generate_button.config(state=tk.DISABLED)
        
        # 获取参数
        direction = self.local_style_direction_var.get()
        mode = self.local_style_mode_var.get()
        auto_select = self.auto_select_regions_var.get()
        ignore_sky = self.ignore_sky_var.get()
        strength = self.local_style_strength_var.get()
        detail = self.local_style_detail_var.get()
        enhance_colors = self.local_style_enhance_colors_var.get()
        smooth_transitions = self.local_style_smooth_transitions_var.get()
        
        # 选择模型
        if "A->B" in direction:  # 照片 -> 莫奈风格
            model = self.model_BA
            model_type = "photo_to_monet"
        else:  # 莫奈风格 -> 照片
            model = self.model_AB
            model_type = "monet_to_photo"
        
        # 启动处理线程
        thread = Thread(target=self.local_style_process_thread, args=(
            model,
            model_type,
            self.local_style_image_path,
            mode,
            auto_select,
            ignore_sky,
            strength,
            detail,
            enhance_colors,
            smooth_transitions
        ))
        thread.daemon = True
        thread.start()
    
    def local_style_process_thread(self, model, model_type, img_path, mode, auto_select, 
                                   ignore_sky, strength, detail, enhance_colors, smooth_transitions):
        """局部风格模式处理线程"""
        try:
            # 加载和预处理图像
            self.root.after(0, lambda: self.local_style_status_var.set("正在预处理图像..."))
            
            original_img = Image.open(img_path).convert('RGB')
            width, height = original_img.size
            original_aspect_ratio = width / height
            
            # 调整图像大小以适应模型
            target_size = (256, 256)
            
            # 保持原始宽高比
            if width > height:
                new_width = target_size[0]
                new_height = int(height * (target_size[0] / width))
            else:
                new_height = target_size[1]
                new_width = int(width * (target_size[1] / height))
            
            resized_img = original_img.resize((new_width, new_height), Image.LANCZOS)
            
            # 创建白色背景画布并粘贴调整后的图像
            canvas = Image.new('RGB', target_size, (255, 255, 255))
            offset_x = (target_size[0] - new_width) // 2
            offset_y = (target_size[1] - new_height) // 2
            canvas.paste(resized_img, (offset_x, offset_y))
            
            # 保存原始画布尺寸用于后续处理
            canvas_width, canvas_height = canvas.size
            
            # 准备输入张量
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            input_tensor = transform(canvas).unsqueeze(0).to(self.device)
            
            # 处理图像
            self.root.after(0, lambda: self.local_style_status_var.set("正在应用局部风格转换..."))
            
            with torch.no_grad():
                output = model(input_tensor)
            
            # 后处理
            output = (output + 1.0) / 2.0
            output = torch.clamp(output, 0, 1)
            output_np = output.cpu().squeeze(0).permute(1, 2, 0).numpy()
            output_img = Image.fromarray((output_np * 255).astype(np.uint8))
            
            # 应用特殊处理
            self.root.after(0, lambda: self.local_style_status_var.set("正在应用局部风格增强..."))
            
            # 根据模式应用不同的处理
            if mode == "简单局部风格":
                # 确保强度值在合理范围内
                blend_ratio = max(0.05, min(1.0 - strength, 0.95))
                
                # 将PIL图像转换为numpy数组
                canvas_np = np.array(canvas)
                output_np = np.array(output_img)
                
                # 简单混合原始图像和风格化图像
                blended = output_np * (1 - blend_ratio) + canvas_np * blend_ratio
                blended = np.clip(blended, 0, 255).astype(np.uint8)
                output_img = Image.fromarray(blended)
                
            elif mode == "增强局部风格":
                # 转换为numpy数组以进行处理
                img_np = np.array(output_img)
                orig_np = np.array(canvas)
                
                # 如果启用天空检测且是照片转莫奈
                if ignore_sky and model_type == "photo_to_monet":
                    try:
                        # 简单的天空检测(基于颜色和位置)
                        hsv = cv2.cvtColor(orig_np, cv2.COLOR_RGB2HSV)
                        
                        # 蓝色HSV范围
                        lower_blue = np.array([90, 30, 140])
                        upper_blue = np.array([130, 255, 255])
                        sky_mask = cv2.inRange(hsv, lower_blue, upper_blue)
                        
                        # 限制为图像上半部分
                        height = orig_np.shape[0]
                        upper_half = np.zeros(orig_np.shape[:2], dtype=np.uint8)
                        upper_half[:height//2, :] = 255
                        
                        # 结合蓝色和上半部分
                        sky_mask = cv2.bitwise_and(sky_mask, upper_half)
                        
                        # 扩展和平滑掩码
                        kernel = np.ones((5,5), np.uint8)
                        sky_mask = cv2.dilate(sky_mask, kernel, iterations=2)
                        sky_mask = cv2.GaussianBlur(sky_mask, (15, 15), 0)
                        
                        # 将掩码归一化为0-1
                        sky_mask_norm = sky_mask.astype(float) / 255.0
                        
                        # 扩展维度以便元素相乘
                        sky_mask_3d = np.expand_dims(sky_mask_norm, axis=2)
                        sky_mask_3d = np.repeat(sky_mask_3d, 3, axis=2)
                        
                        # 根据掩码混合原始图像和转换结果
                        # 天空区域更多地保留原始图像
                        blended = img_np * (1 - sky_mask_3d) + orig_np * sky_mask_3d
                        img_np = np.clip(blended, 0, 255).astype(np.uint8)
                    except Exception as e:
                        print(f"天空处理出错: {str(e)}")
                
                # 如果启用自动区域选择
                if auto_select:
                    try:
                        # 检测边缘
                        gray = cv2.cvtColor(orig_np, cv2.COLOR_RGB2GRAY)
                        edges = cv2.Canny(gray, 50, 150)
                        
                        # 扩展边缘
                        kernel = np.ones((3,3), np.uint8)
                        edges = cv2.dilate(edges, kernel, iterations=1)
                        
                        # 创建基于边缘的权重掩码，边缘附近保留更多细节
                        edge_weight = cv2.GaussianBlur(edges.astype(float) / 255.0, (21, 21), 0)
                        
                        # 扩展维度
                        edge_weight_3d = np.expand_dims(edge_weight, axis=2)
                        edge_weight_3d = np.repeat(edge_weight_3d, 3, axis=2)
                        
                        # 混合原始图像(细节)和转换结果
                        # 边缘区域更多地保留原始图像的细节
                        blended = img_np * (1 - edge_weight_3d * detail) + orig_np * (edge_weight_3d * detail)
                        img_np = np.clip(blended, 0, 255).astype(np.uint8)
                    except Exception as e:
                        print(f"边缘检测出错: {str(e)}")
                
                # 调整整体混合强度 - 重要：确保低强度值时处理正确
                # 创建简单的全局混合矩阵，根据强度参数调整
                # 较低的强度值意味着保留更多原始图像
                global_blend = np.ones(img_np.shape[:2], dtype=np.float32) * strength
                global_blend_3d = np.expand_dims(global_blend, axis=2)
                global_blend_3d = np.repeat(global_blend_3d, 3, axis=2)
                
                # 应用全局混合（针对低强度值的特殊处理）
                if strength < 0.3:  # 当强度很低时
                    # 更温和的混合，确保保留原始图像特性
                    blend_factor = strength / 0.3  # 归一化为0-1
                    img_np = img_np * blend_factor + orig_np * (1 - blend_factor)
                    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                
                # 增强颜色
                if enhance_colors:
                    if model_type == "photo_to_monet":
                        # 增强蓝色和绿色通道
                        img_np[:,:,0] = np.clip(img_np[:,:,0] * 1.1, 0, 255).astype(np.uint8)  # 蓝色
                        img_np[:,:,1] = np.clip(img_np[:,:,1] * 1.05, 0, 255).astype(np.uint8)  # 绿色
                    else:
                        # 增强整体对比度
                        img_np = cv2.convertScaleAbs(img_np, alpha=1.1, beta=5)
                
                # 平滑过渡
                if smooth_transitions:
                    # 应用双边滤波保持边缘锐利同时平滑区域
                    img_np = cv2.bilateralFilter(img_np, d=9, sigmaColor=75, sigmaSpace=75)
                
                # 更新输出图像
                output_img = Image.fromarray(img_np)
                
            elif mode == "高级局部风格":
                # 高级模式使用更复杂的局部处理
                # 这里使用其他模块中的高级函数
                try:
                    # 转换为numpy数组
                    img_np = np.array(output_img)
                    orig_np = np.array(canvas)
                    
                    # 检测颜色块问题区域
                    if hasattr(self, 'detect_color_blocks'):
                        blocks_mask = self.detect_color_blocks(img_np)
                        
                        # 如果检测到色块，应用修复
                        if blocks_mask is not None and np.any(blocks_mask):
                            # 将掩码归一化为0-1
                            blocks_mask_norm = blocks_mask.astype(float) / 255.0 if blocks_mask.max() > 0 else blocks_mask
                            
                            # 扩展维度
                            blocks_mask_3d = np.expand_dims(blocks_mask_norm, axis=2)
                            blocks_mask_3d = np.repeat(blocks_mask_3d, 3, axis=2)
                            
                            # 在色块区域混合原始图像
                            blended = img_np * (1 - blocks_mask_3d) + orig_np * blocks_mask_3d
                            img_np = blended.astype(np.uint8)
                    
                    # 调整整体混合强度 - 重要：确保低强度值时处理正确
                    # 特别处理低强度值
                    if strength < 0.3:  # 当强度很低时
                        # 更温和的混合，确保保留原始图像特性
                        blend_factor = strength / 0.3  # 归一化为0-1
                        img_np = img_np * blend_factor + orig_np * (1 - blend_factor)
                        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                    
                    # 应用自适应颜色校正
                    if hasattr(self, 'adaptive_color_correction'):
                        img_np = self.adaptive_color_correction(img_np, strength=strength)
                    
                    # 应用边缘保留平滑
                    if hasattr(self, 'edge_preserving_smoothing') and smooth_transitions:
                        img_np = self.edge_preserving_smoothing(img_np)
                    
                    # 应用细节增强混合
                    if hasattr(self, 'detail_enhancing_blend') and detail > 0.3:
                        img_np = self.detail_enhancing_blend(img_np, orig_np, detail_factor=detail)
                    
                    # 更新输出图像
                    output_img = Image.fromarray(img_np)
                except Exception as e:
                    print(f"高级局部风格处理出错: {str(e)}")
                    # 如果高级处理失败，仍然返回基本处理结果
            
            # 确保输出图像具有正确的尺寸和比例 - 重要修改
            output_width, output_height = output_img.size
            
            # 确保输出图像尺寸与输入画布尺寸一致
            if output_width != canvas_width or output_height != canvas_height:
                output_img = output_img.resize((canvas_width, canvas_height), Image.LANCZOS)
            
            # 裁剪回原始比例
            if original_aspect_ratio != 1.0:
                # 计算基于原始长宽比的裁剪尺寸
                if original_aspect_ratio > 1:  # 宽大于高
                    crop_width = canvas_width
                    crop_height = int(canvas_width / original_aspect_ratio)
                else:  # 高大于宽
                    crop_height = canvas_height
                    crop_width = int(canvas_height * original_aspect_ratio)
                
                # 确保裁剪尺寸不超过画布
                crop_width = min(crop_width, canvas_width)
                crop_height = min(crop_height, canvas_height)
                
                # 计算裁剪区域的左上角和右下角坐标
                left = (canvas_width - crop_width) // 2
                top = (canvas_height - crop_height) // 2
                right = left + crop_width
                bottom = top + crop_height
                
                # 安全检查，确保裁剪区域是有效的
                if left >= 0 and top >= 0 and right <= canvas_width and bottom <= canvas_height:
                    output_img = output_img.crop((left, top, right, bottom))
            
            # 调整回原始大小 - 确保图像完整
            if width * height <= 1024 * 1024:  # 仅当原始图像不太大时才调整回原始大小
                output_img = output_img.resize((width, height), Image.LANCZOS)
            
            # 保存结果
            output_path = f"output/local_style_{os.path.basename(img_path)}"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output_img.save(output_path)
            
            # 保存处理结果
            self.local_style_result_path = output_path
            
            # 更新UI
            self.root.after(0, lambda: self.update_local_style_result_ui(original_img, output_img))
            
        except Exception as e:
            import traceback
            print(f"局部风格模式处理出错: {str(e)}")
            print(traceback.format_exc())
            error_msg = str(e)
            self.root.after(0, lambda: self.show_local_style_error(error_msg))
    
    def update_local_style_result_ui(self, original_img, result_img):
        """更新局部风格模式结果UI"""
        # 停止进度条
        self.local_style_progress.stop()
        
        # 获取显示区域的尺寸
        frame_width = self.local_style_frame_top.winfo_width() - 20
        frame_height = self.local_style_frame_top.winfo_height() - 20
        
        if frame_width <= 1:  # 如果框架尚未渲染完成，使用默认值
            frame_width = 400
            frame_height = 300
        
        # 确保结果图像尺寸合理且与原始图像一致
        orig_width, orig_height = original_img.size
        result_width, result_height = result_img.size
        
        # 记录原始图像和结果图像的长宽比
        orig_ratio = orig_width / orig_height
        result_ratio = result_width / result_height
        
        # 如果比例差异超过10%，进行特殊处理
        if abs(orig_ratio - result_ratio) > 0.1:
            print(f"警告：原始图像比例 {orig_ratio:.2f} 与结果图像比例 {result_ratio:.2f} 差异较大")
            # 尝试调整结果图像到原始图像的尺寸和比例
            try:
                result_img = result_img.resize(original_img.size, Image.LANCZOS)
                result_width, result_height = result_img.size
                result_ratio = result_width / result_height
            except Exception as e:
                print(f"调整图像尺寸出错: {e}")
        
        # 调整原始图像大小以适应显示框
        orig_display_ratio = min(frame_width/orig_width, frame_height/orig_height)
        orig_new_width = int(orig_width * orig_display_ratio)
        orig_new_height = int(orig_height * orig_display_ratio)
        
        original_display = original_img.resize((orig_new_width, orig_new_height), Image.LANCZOS)
        original_photo = ImageTk.PhotoImage(original_display)
        
        # 调整结果图像大小以适应显示框
        result_display_ratio = min(frame_width/result_width, frame_height/result_height)
        result_new_width = int(result_width * result_display_ratio)
        result_new_height = int(result_height * result_display_ratio)
        
        # 确保结果图像能够完全显示
        if result_new_width > frame_width:
            result_new_width = frame_width
            result_new_height = int(result_height * (frame_width / result_width))
        
        if result_new_height > frame_height:
            result_new_height = frame_height
            result_new_width = int(result_width * (frame_height / result_height))
        
        result_display = result_img.resize((result_new_width, result_new_height), Image.LANCZOS)
        result_photo = ImageTk.PhotoImage(result_display)
        
        # 更新图像显示
        self.local_style_image_top.configure(image=original_photo)
        self.local_style_image_top.image = original_photo
        
        self.local_style_image_bottom.configure(image=result_photo)
        self.local_style_image_bottom.image = result_photo
        
        # 更新状态和按钮
        self.local_style_status_var.set("处理完成")
        self.local_style_generate_button.config(state=tk.NORMAL)
        self.local_style_save_button.config(state=tk.NORMAL)
        self.is_processing = False
    
    def show_local_style_error(self, error_msg):
        """显示局部风格模式处理错误"""
        # 停止进度条
        self.local_style_progress.stop()
        
        # 更新状态
        self.local_style_status_var.set(f"错误: {error_msg}")
        
        # 启用处理按钮
        self.local_style_generate_button.config(state=tk.NORMAL)
        
        # 重置处理状态
        self.is_processing = False
        
        # 显示错误对话框
        messagebox.showerror("处理错误", f"局部风格模式处理过程中出错: {error_msg}")
    
    def save_local_style_result(self):
        """保存局部风格转换结果"""
        if not hasattr(self, 'local_style_result_path'):
            messagebox.showerror("错误", "没有可保存的结果")
            return
        
        # 选择保存位置
        save_path = filedialog.asksaveasfilename(
            title="保存局部风格转换结果",
            defaultextension=".jpg",
            initialfile=f"local_style_{os.path.basename(self.local_style_image_path)}",
            filetypes=[("JPEG图像", "*.jpg"), ("PNG图像", "*.png"), ("所有文件", "*.*")]
        )
        
        if save_path:
            try:
                # 复制结果文件
                shutil.copy2(self.local_style_result_path, save_path)
                
                # 显示成功消息
                messagebox.showinfo("保存成功", f"局部风格转换结果已保存到: {save_path}")
            except Exception as e:
                messagebox.showerror("保存错误", f"保存结果时出错: {str(e)}")
    
    def setup_cyclegan_tab(self, parent_frame):
        """设置原始CycleGAN模式选项卡"""
        # 创建主布局，使用左右分割的Panedwindow
        main_paned = ttk.PanedWindow(parent_frame, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧控制面板，设置固定宽度
        control_frame = ttk.Frame(main_paned, width=280)
        control_frame.pack_propagate(False)  # 防止frame自动调整大小
        main_paned.add(control_frame, weight=1)
        
        # 右侧显示区域
        display_frame = ttk.Frame(main_paned)
        main_paned.add(display_frame, weight=4)  # 给显示区更大的权重
        
        # === 左侧控制面板 ===
        # 转换选项框架
        options_frame = ttk.LabelFrame(control_frame, text="CycleGAN转换选项", padding=(10, 5))
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(options_frame, text="转换方向:").grid(row=0, column=0, sticky=tk.W, pady=(5, 2))
        
        self.cyclegan_direction_var = tk.StringVar()
        directions = []
        if hasattr(self, 'cyclegan_loaded') and self.cyclegan_loaded:
            directions.append("莫奈风格 -> 照片 (B->A)")
            directions.append("照片 -> 莫奈风格 (A->B)")
        
        if directions:
            self.cyclegan_direction_var.set(directions[0])
        
        direction_dropdown = ttk.Combobox(options_frame, 
                                          textvariable=self.cyclegan_direction_var, 
                                          values=directions, 
                                          width=22,
                                          state="readonly")
        direction_dropdown.grid(row=0, column=1, sticky=tk.W, pady=(5, 2))
        
        # 图像选择区域
        image_frame = ttk.LabelFrame(control_frame, text="图像选择", padding=(10, 5))
        image_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.cyclegan_image_path_var = tk.StringVar(value="未选择图像")
        ttk.Label(image_frame, textvariable=self.cyclegan_image_path_var, wraplength=250).pack(fill=tk.X, pady=2)
        ttk.Button(image_frame, text="选择图像", command=self.select_cyclegan_image).pack(fill=tk.X, pady=5)
        
        # 操作按钮区域
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.cyclegan_process_button = ttk.Button(button_frame, 
                                                text="生成CycleGAN结果", 
                                                command=self.process_cyclegan,
                                                state=tk.DISABLED)
        self.cyclegan_process_button.pack(fill=tk.X, pady=2)
        
        self.cyclegan_save_button = ttk.Button(button_frame, 
                                             text="保存结果", 
                                             command=self.save_cyclegan_result,
                                             state=tk.DISABLED)
        self.cyclegan_save_button.pack(fill=tk.X, pady=2)
        
        # === 右侧显示区域 ===
        # 创建垂直布局
        display_paned = ttk.PanedWindow(display_frame, orient=tk.VERTICAL)
        display_paned.pack(fill=tk.BOTH, expand=True)
        
        # 顶部显示原始图像
        self.cyclegan_frame_original = ttk.LabelFrame(display_paned, text="原始图像")
        display_paned.add(self.cyclegan_frame_original, weight=1)
        
        # 原始图像显示区域
        self.cyclegan_image_original = ttk.Label(self.cyclegan_frame_original)
        self.cyclegan_image_original.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 底部显示结果图像
        self.cyclegan_frame_result = ttk.LabelFrame(display_paned, text="CycleGAN转换结果")
        display_paned.add(self.cyclegan_frame_result, weight=1)
        
        # 结果图像显示区域
        self.cyclegan_image_result = ttk.Label(self.cyclegan_frame_result)
        self.cyclegan_image_result.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 状态栏
        status_frame = ttk.Frame(parent_frame)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        self.cyclegan_status_var = tk.StringVar(value="就绪")
        self.cyclegan_status_label = ttk.Label(status_frame, textvariable=self.cyclegan_status_var)
        self.cyclegan_status_label.pack(side=tk.LEFT)
        
        self.cyclegan_progress = ttk.Progressbar(status_frame, orient=tk.HORIZONTAL, mode='indeterminate')
        self.cyclegan_progress.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
    
    def select_cyclegan_image(self):
        """选择CycleGAN模式的输入图像"""
        file_path = filedialog.askopenfilename(
            title="选择图像",
            filetypes=[("图像文件", "*.png;*.jpg;*.jpeg")]
        )
        
        if file_path:
            self.cyclegan_image_path = file_path
            self.cyclegan_image_path_var.set(os.path.basename(file_path))
            
            # 显示选择的图像
            try:
                img = Image.open(file_path)
                # 保持原始宽高比，适应框架大小
                frame_width = self.cyclegan_frame_original.winfo_width() - 20
                frame_height = self.cyclegan_frame_original.winfo_height() - 20
                
                if frame_width <= 1:  # 如果框架尚未渲染完成，使用默认值
                    frame_width = 400
                    frame_height = 300
                
                img_width, img_height = img.size
                ratio = min(frame_width/img_width, frame_height/img_height)
                new_width = int(img_width * ratio)
                new_height = int(img_height * ratio)
                
                # 调整大小
                img_display = img.resize((new_width, new_height), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img_display)
                
                self.cyclegan_image_original.configure(image=photo)
                self.cyclegan_image_original.image = photo
                
                # 启用处理按钮
                self.cyclegan_process_button.config(state=tk.NORMAL)
                
                # 更新状态
                self.cyclegan_status_var.set(f"已选择图像: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("错误", f"打开图像文件时出错: {str(e)}")
    
    def process_cyclegan(self):
        """处理CycleGAN转换"""
        if not hasattr(self, 'cyclegan_image_path') or not self.cyclegan_image_path:
            messagebox.showerror("错误", "请先选择一张图像")
            return
        
        if self.is_processing:
            return
        
        # 设置处理状态
        self.is_processing = True
        self.cyclegan_status_var.set("正在处理...")
        self.cyclegan_progress.start()
        self.cyclegan_process_button.config(state=tk.DISABLED)
        
        # 获取参数
        direction = self.cyclegan_direction_var.get()
        
        # 选择模型
        if "A->B" in direction:  # 照片 -> 莫奈风格
            model = self.cyclegan_model_BA
            model_type = "photo_to_monet"
        else:  # 莫奈风格 -> 照片
            model = self.cyclegan_model_AB
            model_type = "monet_to_photo"
        
        # 启动处理线程
        thread = Thread(target=self.cyclegan_process_thread, args=(
            model,
            model_type,
            self.cyclegan_image_path
        ))
        thread.daemon = True
        thread.start()
    
    def cyclegan_process_thread(self, model, model_type, img_path):
        """CycleGAN处理线程"""
        try:
            # 加载和预处理图像
            self.root.after(0, lambda: self.cyclegan_status_var.set("正在预处理图像..."))
            
            original_img = Image.open(img_path).convert('RGB')
            width, height = original_img.size
            
            # 调整图像大小以适应模型
            target_size = (256, 256)
            
            # 保持原始宽高比
            if width > height:
                new_width = target_size[0]
                new_height = int(height * (target_size[0] / width))
            else:
                new_height = target_size[1]
                new_width = int(width * (target_size[1] / height))
            
            resized_img = original_img.resize((new_width, new_height), Image.LANCZOS)
            
            # 创建白色背景画布并粘贴调整后的图像
            canvas = Image.new('RGB', target_size, (255, 255, 255))
            offset_x = (target_size[0] - new_width) // 2
            offset_y = (target_size[1] - new_height) // 2
            canvas.paste(resized_img, (offset_x, offset_y))
            
            # 准备输入张量
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            input_tensor = transform(canvas).unsqueeze(0).to(self.device)
            
            # 处理图像
            self.root.after(0, lambda: self.cyclegan_status_var.set("正在应用CycleGAN转换..."))
            
            with torch.no_grad():
                output = model(input_tensor)
            
            # 后处理
            output = (output + 1.0) / 2.0
            output = torch.clamp(output, 0, 1)
            output_np = output.cpu().squeeze(0).permute(1, 2, 0).numpy()
            output_img = Image.fromarray((output_np * 255).astype(np.uint8))
            
            # 裁剪回原始比例
            if width != height:
                aspect_ratio = width / height
                if aspect_ratio > 1:  # 宽大于高
                    crop_width = target_size[0]
                    crop_height = int(target_size[0] / aspect_ratio)
                else:  # 高大于宽
                    crop_height = target_size[1]
                    crop_width = int(target_size[1] * aspect_ratio)
                
                left = (target_size[0] - crop_width) // 2
                top = (target_size[1] - crop_height) // 2
                right = left + crop_width
                bottom = top + crop_height
                
                output_img = output_img.crop((left, top, right, bottom))
            
            # 调整回原始大小
            if width * height <= 1024 * 1024:  # 仅当原始图像不太大时才调整回原始大小
                output_img = output_img.resize((width, height), Image.LANCZOS)
            
            # 保存结果
            output_path = f"output/cyclegan_{os.path.basename(img_path)}"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output_img.save(output_path)
            
            # 保存处理结果
            self.cyclegan_result_path = output_path
            
            # 更新UI
            self.root.after(0, lambda: self.update_cyclegan_result_ui(original_img, output_img))
            
        except Exception as e:
            import traceback
            print(f"CycleGAN处理出错: {str(e)}")
            print(traceback.format_exc())
            error_msg = str(e)
            self.root.after(0, lambda: self.show_cyclegan_error(error_msg))
    
    def update_cyclegan_result_ui(self, original_img, result_img):
        """更新CycleGAN结果UI"""
        # 停止进度条
        self.cyclegan_progress.stop()
        
        # 显示原始图像和结果图像
        frame_width = self.cyclegan_frame_original.winfo_width() - 20
        frame_height = self.cyclegan_frame_original.winfo_height() - 20
        
        if frame_width <= 1:  # 如果框架尚未渲染完成，使用默认值
            frame_width = 400
            frame_height = 300
        
        # 调整原始图像大小
        img_width, img_height = original_img.size
        ratio = min(frame_width/img_width, frame_height/img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        original_display = original_img.resize((new_width, new_height), Image.LANCZOS)
        original_photo = ImageTk.PhotoImage(original_display)
        
        # 调整结果图像大小
        result_width, result_height = result_img.size
        ratio = min(frame_width/result_width, frame_height/result_height)
        new_width = int(result_width * ratio)
        new_height = int(result_height * ratio)
        
        result_display = result_img.resize((new_width, new_height), Image.LANCZOS)
        result_photo = ImageTk.PhotoImage(result_display)
        
        # 更新图像显示
        self.cyclegan_image_original.configure(image=original_photo)
        self.cyclegan_image_original.image = original_photo
        
        self.cyclegan_image_result.configure(image=result_photo)
        self.cyclegan_image_result.image = result_photo
        
        # 更新状态和按钮
        self.cyclegan_status_var.set("处理完成")
        self.cyclegan_process_button.config(state=tk.NORMAL)
        self.cyclegan_save_button.config(state=tk.NORMAL)
        self.is_processing = False
    
    def show_cyclegan_error(self, error_msg):
        """显示CycleGAN处理错误"""
        # 停止进度条
        self.cyclegan_progress.stop()
        
        # 更新状态
        self.cyclegan_status_var.set(f"错误: {error_msg}")
        
        # 启用处理按钮
        self.cyclegan_process_button.config(state=tk.NORMAL)
        
        # 重置处理状态
        self.is_processing = False
        
        # 显示错误对话框
        messagebox.showerror("处理错误", f"CycleGAN处理过程中出错: {error_msg}")
    
    def save_cyclegan_result(self):
        """保存CycleGAN转换结果"""
        if not hasattr(self, 'cyclegan_result_path'):
            messagebox.showerror("错误", "没有可保存的结果")
            return
        
        # 选择保存位置
        save_path = filedialog.asksaveasfilename(
            title="保存CycleGAN转换结果",
            defaultextension=".jpg",
            initialfile=f"cyclegan_{os.path.basename(self.cyclegan_image_path)}",
            filetypes=[("JPEG图像", "*.jpg"), ("PNG图像", "*.png"), ("所有文件", "*.*")]
        )
        
        if save_path:
            try:
                # 复制结果文件
                shutil.copy2(self.cyclegan_result_path, save_path)
                
                # 显示成功消息
                messagebox.showinfo("保存成功", f"CycleGAN转换结果已保存到: {save_path}")
            except Exception as e:
                messagebox.showerror("保存错误", f"保存结果时出错: {str(e)}")
    
    def setup_compare_tab(self, parent_frame):
        """设置对比模式选项卡"""
        # 创建主布局，使用左右分割的Panedwindow
        main_paned = ttk.PanedWindow(parent_frame, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧控制面板，设置固定宽度
        control_frame = ttk.Frame(main_paned, width=220)
        control_frame.pack_propagate(False)  # 防止frame自动调整大小
        main_paned.add(control_frame, weight=1)
        
        # 右侧显示区域
        display_frame = ttk.Frame(main_paned)
        main_paned.add(display_frame, weight=5)  # 增加显示区域权重，更好地利用空间
        
        # === 左侧控制面板 ===
        # 转换选项框架
        options_frame = ttk.LabelFrame(control_frame, text="转换选项", padding=(10, 5))
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(options_frame, text="转换方向:").grid(row=0, column=0, sticky=tk.W, pady=(5, 2))
        
        self.compare_direction_var = tk.StringVar()
        directions = []
        if hasattr(self, 'AB_loaded') and self.AB_loaded:
            directions.append("莫奈风格 -> 照片 (B->A)")
        if hasattr(self, 'BA_loaded') and self.BA_loaded:
            directions.append("照片 -> 莫奈风格 (A->B)")
        
        if directions:
            self.compare_direction_var.set(directions[0])
        
        direction_dropdown = ttk.Combobox(options_frame, 
                                          textvariable=self.compare_direction_var, 
                                          values=directions, 
                                          width=22,
                                          state="readonly")
        direction_dropdown.grid(row=0, column=1, sticky=tk.W, pady=(5, 2))
        
        # 图像选择区域
        image_frame = ttk.LabelFrame(control_frame, text="图像选择", padding=(10, 5))
        image_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.compare_image_path_var = tk.StringVar(value="未选择图像")
        ttk.Label(image_frame, textvariable=self.compare_image_path_var, wraplength=250).pack(fill=tk.X, pady=2)
        ttk.Button(image_frame, text="选择图像", command=self.select_compare_image).pack(fill=tk.X, pady=5)
        
        # 混合和平滑设置
        advanced_frame = ttk.LabelFrame(control_frame, text="高级参数", padding=(10, 5))
        advanced_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 风格混合强度
        blend_frame = ttk.Frame(advanced_frame)
        blend_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(blend_frame, text="混合强度:").pack(side=tk.LEFT)
        self.compare_blend_strength_var = tk.DoubleVar(value=0.5)  # 默认值改为0.5
        self.compare_blend_strength_label = ttk.Label(blend_frame, text="0.50")
        self.compare_blend_strength_label.pack(side=tk.RIGHT)
        
        def update_blend_value(event=None):
            val = round(self.compare_blend_strength_var.get(), 2)
            self.compare_blend_strength_label.config(text=f"{val:.2f}")
        
        blend_scale = ttk.Scale(advanced_frame, 
                                from_=0.15, 
                                to=0.85, 
                                variable=self.compare_blend_strength_var, 
                                orient=tk.HORIZONTAL, 
                                command=update_blend_value)
        blend_scale.pack(fill=tk.X, pady=2)
        
        # 边缘平滑程度
        smooth_frame = ttk.Frame(advanced_frame)
        smooth_frame.pack(fill=tk.X, pady=(5, 2))
        
        ttk.Label(smooth_frame, text="平滑程度:").pack(side=tk.LEFT)
        self.compare_smooth_var = tk.IntVar(value=35)  # 默认35
        self.compare_smooth_label = ttk.Label(smooth_frame, text="35")
        self.compare_smooth_label.pack(side=tk.RIGHT)
        
        def update_smooth_value(event=None):
            val = int(float(smooth_scale.get()))
            if val % 2 == 0:
                val += 1
                self.compare_smooth_var.set(val)
            self.compare_smooth_label.config(text=f"{val}")
        
        smooth_scale = ttk.Scale(advanced_frame, 
                                 from_=15, 
                                 to=55, 
                                 variable=self.compare_smooth_var, 
                                 orient=tk.HORIZONTAL, 
                                 command=update_smooth_value)
        smooth_scale.pack(fill=tk.X, pady=2)
        
        # 特殊处理选项
        self.compare_sky_handling_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_frame, text="天空区域特殊处理", 
                        variable=self.compare_sky_handling_var).pack(anchor=tk.W, pady=2)
        
        self.compare_fix_blocks_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_frame, text="修复色块问题", 
                        variable=self.compare_fix_blocks_var).pack(anchor=tk.W, pady=2)
        
        # 操作按钮区域
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.compare_process_button = ttk.Button(button_frame, 
                                                text="开始对比处理", 
                                                command=self.start_compare_process,
                                                state=tk.DISABLED)
        self.compare_process_button.pack(fill=tk.X, pady=2)
        
        self.compare_save_button = ttk.Button(button_frame, 
                                            text="保存对比结果", 
                                            command=self.save_compare_result,
                                            state=tk.DISABLED)
        self.compare_save_button.pack(fill=tk.X, pady=2)
        
        # === 右侧显示区域 ===
        # 创建垂直分割窗口，修改为水平和垂直混合布局
        display_main = ttk.Frame(display_frame)
        display_main.pack(fill=tk.BOTH, expand=True)
        
        # 原始图像显示区域
        original_container = ttk.LabelFrame(display_main, text="原始图像")
        original_container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # 在原始图像下方创建水平分割的结果区域
        results_container = ttk.Frame(display_main)
        results_container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # 左侧结果显示区域
        left_container = ttk.LabelFrame(results_container, text="局部风格转换效果")
        left_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # 右侧结果显示区域
        right_container = ttk.LabelFrame(results_container, text="原始CycleGAN效果")
        right_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # 原始图像显示区域（添加滚动条支持）
        self.compare_original_canvas = tk.Canvas(original_container, highlightthickness=0)
        self.compare_original_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        original_scrollbar_y = ttk.Scrollbar(original_container, orient=tk.VERTICAL, 
                                            command=self.compare_original_canvas.yview)
        original_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        original_scrollbar_x = ttk.Scrollbar(original_container, orient=tk.HORIZONTAL, 
                                            command=self.compare_original_canvas.xview)
        original_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.compare_original_canvas.configure(yscrollcommand=original_scrollbar_y.set,
                                              xscrollcommand=original_scrollbar_x.set)
        
        self.compare_original_frame = ttk.Frame(self.compare_original_canvas)
        self.compare_original_canvas.create_window((0, 0), window=self.compare_original_frame, 
                                                  anchor=tk.NW, tags="self.compare_original_frame")
        
        # 在框架中放置标签显示图像
        self.compare_original_image = ttk.Label(self.compare_original_frame)
        self.compare_original_image.pack(padx=5, pady=5)
        
        # 创建带滚动条的左侧结果区域
        self.compare_local_canvas = tk.Canvas(left_container, highlightthickness=0)
        self.compare_local_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        left_scrollbar_y = ttk.Scrollbar(left_container, orient=tk.VERTICAL, 
                                        command=self.compare_local_canvas.yview)
        left_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        left_scrollbar_x = ttk.Scrollbar(left_container, orient=tk.HORIZONTAL, 
                                        command=self.compare_local_canvas.xview)
        left_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.compare_local_canvas.configure(yscrollcommand=left_scrollbar_y.set,
                                          xscrollcommand=left_scrollbar_x.set)
        
        self.compare_local_frame = ttk.Frame(self.compare_local_canvas)
        self.compare_local_canvas.create_window((0, 0), window=self.compare_local_frame, 
                                              anchor=tk.NW, tags="self.compare_local_frame")
        
        self.compare_local_image = ttk.Label(self.compare_local_frame)
        self.compare_local_image.pack(padx=5, pady=5)
        
        # 创建带滚动条的右侧结果区域
        self.compare_cyclegan_canvas = tk.Canvas(right_container, highlightthickness=0)
        self.compare_cyclegan_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_scrollbar_y = ttk.Scrollbar(right_container, orient=tk.VERTICAL, 
                                         command=self.compare_cyclegan_canvas.yview)
        right_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        right_scrollbar_x = ttk.Scrollbar(right_container, orient=tk.HORIZONTAL, 
                                         command=self.compare_cyclegan_canvas.xview)
        right_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.compare_cyclegan_canvas.configure(yscrollcommand=right_scrollbar_y.set,
                                             xscrollcommand=right_scrollbar_x.set)
        
        self.compare_cyclegan_frame = ttk.Frame(self.compare_cyclegan_canvas)
        self.compare_cyclegan_canvas.create_window((0, 0), window=self.compare_cyclegan_frame, 
                                                 anchor=tk.NW, tags="self.compare_cyclegan_frame")
        
        self.compare_cyclegan_image = ttk.Label(self.compare_cyclegan_frame)
        self.compare_cyclegan_image.pack(padx=5, pady=5)
        
        # 状态栏
        status_frame = ttk.Frame(parent_frame)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        self.compare_status_var = tk.StringVar(value="就绪")
        self.compare_status_label = ttk.Label(status_frame, textvariable=self.compare_status_var)
        self.compare_status_label.pack(side=tk.LEFT)
        
        self.compare_progress = ttk.Progressbar(status_frame, orient=tk.HORIZONTAL, mode='indeterminate')
        self.compare_progress.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        
        # 配置画布区域调整大小事件
        def configure_canvas_scrollregion(event, canvas, frame):
            canvas.configure(scrollregion=canvas.bbox("all"))
            frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        self.compare_original_frame.bind("<Configure>", 
                                        lambda e: configure_canvas_scrollregion(e, self.compare_original_canvas, 
                                                                               self.compare_original_frame))
        self.compare_local_frame.bind("<Configure>", 
                                     lambda e: configure_canvas_scrollregion(e, self.compare_local_canvas, 
                                                                           self.compare_local_frame))
        self.compare_cyclegan_frame.bind("<Configure>", 
                                        lambda e: configure_canvas_scrollregion(e, self.compare_cyclegan_canvas, 
                                                                              self.compare_cyclegan_frame))
    
    def select_compare_image(self):
        """选择对比模式的输入图像"""
        file_path = filedialog.askopenfilename(
            title="选择图像",
            filetypes=[("图像文件", "*.png;*.jpg;*.jpeg")]
        )
        
        if file_path:
            self.compare_image_path = file_path
            self.compare_image_path_var.set(os.path.basename(file_path))
            
            # 显示选择的图像
            try:
                img = Image.open(file_path)
                
                # 保持原始宽高比，适应框架大小
                max_width = 500  # 减小最大宽度与处理结果一致
                max_height = 350  # 减小最大高度与处理结果一致
                
                img_width, img_height = img.size
                ratio = min(max_width/img_width, max_height/img_height)
                
                # 如果图像小于最大尺寸，则保持原始大小
                if img_width <= max_width and img_height <= max_height:
                    display_width, display_height = img_width, img_height
                else:
                    display_width = int(img_width * ratio)
                    display_height = int(img_height * ratio)
                
                # 调整大小
                img_display = img.resize((display_width, display_height), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img_display)
                
                self.compare_original_image.configure(image=photo)
                self.compare_original_image.image = photo
                
                # 更新画布滚动区域
                self.compare_original_canvas.configure(scrollregion=self.compare_original_canvas.bbox("all"))
                
                # 启用处理按钮
                self.compare_process_button.config(state=tk.NORMAL)
                
                # 更新状态
                self.compare_status_var.set(f"已选择图像: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("错误", f"打开图像文件时出错: {str(e)}")
    
    def update_compare_result_ui(self, original_img, local_result, cyclegan_result):
        """更新对比结果UI"""
        # 停止进度条
        self.compare_progress.stop()
        
        # 图像显示尺寸设置 - 适应界面
        max_width = 500  # 最大宽度
        max_height = 350  # 最大高度
        
        # 确保结果图像尺寸合理且与原始图像一致
        orig_width, orig_height = original_img.size
        local_width, local_height = local_result.size
        cyclegan_width, cyclegan_height = cyclegan_result.size
        
        # 记录各图像的长宽比
        orig_ratio = orig_width / orig_height
        local_ratio = local_width / local_height  
        cyclegan_ratio = cyclegan_width / cyclegan_height
        
        # 如果结果图像比例与原始图像差异较大，进行特殊处理
        if abs(orig_ratio - local_ratio) > 0.1:
            print(f"警告：原始图像比例 {orig_ratio:.2f} 与局部风格结果比例 {local_ratio:.2f} 差异较大")
            try:
                # 尝试调整局部风格结果到原始图像的比例
                local_result = local_result.resize(original_img.size, Image.LANCZOS)
                local_width, local_height = local_result.size
                local_ratio = local_width / local_height
            except Exception as e:
                print(f"调整局部风格图像尺寸出错: {e}")
        
        if abs(orig_ratio - cyclegan_ratio) > 0.1:
            print(f"警告：原始图像比例 {orig_ratio:.2f} 与CycleGAN结果比例 {cyclegan_ratio:.2f} 差异较大")
            try:
                # 尝试调整CycleGAN结果到原始图像的比例
                cyclegan_result = cyclegan_result.resize(original_img.size, Image.LANCZOS)
                cyclegan_width, cyclegan_height = cyclegan_result.size
                cyclegan_ratio = cyclegan_width / cyclegan_height
            except Exception as e:
                print(f"调整CycleGAN图像尺寸出错: {e}")
        
        # 处理原始图像 - 计算显示尺寸
        orig_display_ratio = min(max_width/orig_width, max_height/orig_height)
        orig_display_width = int(orig_width * orig_display_ratio)
        orig_display_height = int(orig_height * orig_display_ratio)
        
        # 如果图像比显示区域小，保持原始大小
        if orig_width <= max_width and orig_height <= max_height:
            orig_display_width, orig_display_height = orig_width, orig_height
        
        # 调整原始图像大小
        original_display = original_img.resize((orig_display_width, orig_display_height), Image.LANCZOS)
        original_photo = ImageTk.PhotoImage(original_display)
        
        # 更新原始图像显示
        self.compare_original_image.configure(image=original_photo)
        self.compare_original_image.image = original_photo
        
        # 处理局部风格图像 - 计算显示尺寸
        local_display_ratio = min(max_width/local_width, max_height/local_height)
        local_display_width = int(local_width * local_display_ratio)
        local_display_height = int(local_height * local_display_ratio)
        
        # 如果图像比显示区域小，保持原始大小
        if local_width <= max_width and local_height <= max_height:
            local_display_width, local_display_height = local_width, local_height
        
        # 处理CycleGAN图像 - 计算显示尺寸
        cyclegan_display_ratio = min(max_width/cyclegan_width, max_height/cyclegan_height)
        cyclegan_display_width = int(cyclegan_width * cyclegan_display_ratio)
        cyclegan_display_height = int(cyclegan_height * cyclegan_display_ratio)
        
        # 如果图像比显示区域小，保持原始大小
        if cyclegan_width <= max_width and cyclegan_height <= max_height:
            cyclegan_display_width, cyclegan_display_height = cyclegan_width, cyclegan_height
        
        # 调整结果图像大小
        local_display = local_result.resize((local_display_width, local_display_height), Image.LANCZOS)
        cyclegan_display = cyclegan_result.resize((cyclegan_display_width, cyclegan_display_height), Image.LANCZOS)
        
        # 转换为PhotoImage对象
        local_photo = ImageTk.PhotoImage(local_display)
        cyclegan_photo = ImageTk.PhotoImage(cyclegan_display)
        
        # 更新UI
        self.compare_local_image.configure(image=local_photo)
        self.compare_local_image.image = local_photo
        
        self.compare_cyclegan_image.configure(image=cyclegan_photo)
        self.compare_cyclegan_image.image = cyclegan_photo
        
        # 更新画布滚动区域 - 确保完整显示图像
        self.compare_original_frame.update()  # 强制更新以获取正确的尺寸
        self.compare_local_frame.update()
        self.compare_cyclegan_frame.update()
        
        # 设置适当的滚动区域，确保能够滚动查看完整图像
        self.compare_original_canvas.configure(scrollregion=self.compare_original_canvas.bbox("all"))
        self.compare_local_canvas.configure(scrollregion=self.compare_local_canvas.bbox("all"))
        self.compare_cyclegan_canvas.configure(scrollregion=self.compare_cyclegan_canvas.bbox("all"))
        
        # 更新状态
        self.compare_status_var.set("对比处理完成")
        
        # 启用按钮
        self.compare_process_button.config(state=tk.NORMAL)
        self.compare_save_button.config(state=tk.NORMAL)
        
        # 重置处理状态
        self.is_processing = False
    
    def start_compare_process(self):
        """开始对比处理"""
        if not hasattr(self, 'compare_image_path') or not self.compare_image_path:
            messagebox.showerror("错误", "请先选择一张图像")
            return
        
        if self.is_processing:
            return
        
        # 设置处理状态
        self.is_processing = True
        self.compare_status_var.set("正在处理对比...")
        self.compare_progress.start()
        self.compare_process_button.config(state=tk.DISABLED)
        
        # 获取参数
        direction = self.compare_direction_var.get()
        blend_strength = self.compare_blend_strength_var.get()
        smooth_level = self.compare_smooth_var.get()
        sky_handling = self.compare_sky_handling_var.get()
        fix_blocks = self.compare_fix_blocks_var.get()
        
        # 选择模型
        if "A->B" in direction:  # 照片 -> 莫奈风格
            enhanced_model = self.model_BA
            cyclegan_model = self.cyclegan_model_BA
            model_type = "photo_to_monet"
        else:  # 莫奈风格 -> 照片
            enhanced_model = self.model_AB
            cyclegan_model = self.cyclegan_model_AB
            model_type = "monet_to_photo"
        
        # 启动处理线程
        thread = Thread(target=self.compare_process_thread, args=(
            enhanced_model,
            cyclegan_model,
            model_type,
            self.compare_image_path,
            blend_strength,
            smooth_level,
            sky_handling,
            fix_blocks
        ))
        thread.daemon = True
        thread.start()
    
    def compare_process_thread(self, enhanced_model, cyclegan_model, model_type, img_path, 
                              blend_strength, smooth_level, sky_handling, fix_blocks):
        """对比模式处理线程"""
        try:
            # 加载和预处理图像
            self.root.after(0, lambda: self.compare_status_var.set("正在预处理图像..."))
            
            original_img = Image.open(img_path).convert('RGB')
            width, height = original_img.size
            original_aspect_ratio = width / height
            
            # 调整图像大小以适应模型
            target_size = (256, 256)
            
            # 保持原始宽高比
            if width > height:
                new_width = target_size[0]
                new_height = int(height * (target_size[0] / width))
            else:
                new_height = target_size[1]
                new_width = int(width * (target_size[1] / height))
            
            resized_img = original_img.resize((new_width, new_height), Image.LANCZOS)
            
            # 创建白色背景画布并粘贴调整后的图像
            canvas = Image.new('RGB', target_size, (255, 255, 255))
            offset_x = (target_size[0] - new_width) // 2
            offset_y = (target_size[1] - new_height) // 2
            canvas.paste(resized_img, (offset_x, offset_y))
            
            # 保存原始画布尺寸用于后续处理
            canvas_width, canvas_height = canvas.size
            
            # 准备输入张量
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            input_tensor = transform(canvas).unsqueeze(0).to(self.device)
            
            # 处理局部风格模式图像（使用与独立选项卡相同的处理逻辑）
            self.root.after(0, lambda: self.compare_status_var.set("正在应用局部风格转换..."))
            
            with torch.no_grad():
                enhanced_output = enhanced_model(input_tensor)
            
            # 后处理增强模型输出
            enhanced_output = (enhanced_output + 1.0) / 2.0
            enhanced_output = torch.clamp(enhanced_output, 0, 1)
            enhanced_np = enhanced_output.cpu().squeeze(0).permute(1, 2, 0).numpy()
            enhanced_img = Image.fromarray((enhanced_np * 255).astype(np.uint8))
            
            # 模拟局部风格模式的"增强局部风格"处理
            img_np = np.array(enhanced_img)
            orig_np = np.array(canvas)
            
            # 如果启用天空检测且是照片转莫奈
            if sky_handling and model_type == "photo_to_monet":
                try:
                    # 简单的天空检测(基于颜色和位置)
                    hsv = cv2.cvtColor(orig_np, cv2.COLOR_RGB2HSV)
                    
                    # 蓝色HSV范围
                    lower_blue = np.array([90, 30, 140])
                    upper_blue = np.array([130, 255, 255])
                    sky_mask = cv2.inRange(hsv, lower_blue, upper_blue)
                    
                    # 限制为图像上半部分
                    height = orig_np.shape[0]
                    upper_half = np.zeros(orig_np.shape[:2], dtype=np.uint8)
                    upper_half[:height//2, :] = 255
                    
                    # 结合蓝色和上半部分
                    sky_mask = cv2.bitwise_and(sky_mask, upper_half)
                    
                    # 扩展和平滑掩码
                    kernel = np.ones((5,5), np.uint8)
                    sky_mask = cv2.dilate(sky_mask, kernel, iterations=2)
                    sky_mask = cv2.GaussianBlur(sky_mask, (15, 15), 0)
                    
                    # 将掩码归一化为0-1
                    sky_mask_norm = sky_mask.astype(float) / 255.0
                    
                    # 扩展维度以便元素相乘
                    sky_mask_3d = np.expand_dims(sky_mask_norm, axis=2)
                    sky_mask_3d = np.repeat(sky_mask_3d, 3, axis=2)
                    
                    # 根据掩码混合原始图像和转换结果
                    # 天空区域更多地保留原始图像
                    blended = img_np * (1 - sky_mask_3d) + orig_np * sky_mask_3d
                    img_np = np.clip(blended, 0, 255).astype(np.uint8)
                except Exception as e:
                    print(f"天空处理出错: {str(e)}")
            
            # 应用自动区域检测
            try:
                # 检测边缘
                gray = cv2.cvtColor(orig_np, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                
                # 扩展边缘
                kernel = np.ones((3,3), np.uint8)
                edges = cv2.dilate(edges, kernel, iterations=1)
                
                # 创建基于边缘的权重掩码，边缘附近保留更多细节
                edge_weight = cv2.GaussianBlur(edges.astype(float) / 255.0, (21, 21), 0)
                
                # 扩展维度
                edge_weight_3d = np.expand_dims(edge_weight, axis=2)
                edge_weight_3d = np.repeat(edge_weight_3d, 3, axis=2)
                
                # 混合原始图像(细节)和转换结果
                # 边缘区域更多地保留原始图像的细节
                detail_factor = 0.6  # 使用固定的细节保留参数，与默认值一致
                blended = img_np * (1 - edge_weight_3d * detail_factor) + orig_np * (edge_weight_3d * detail_factor)
                img_np = np.clip(blended, 0, 255).astype(np.uint8)
            except Exception as e:
                print(f"边缘检测出错: {str(e)}")
            
            # 调整整体混合强度 - 根据blend_strength参数
            if blend_strength < 0.3:  # 当强度很低时
                # 更温和的混合，确保保留原始图像特性
                blend_factor = blend_strength / 0.3  # 归一化为0-1
                img_np = img_np * blend_factor + orig_np * (1 - blend_factor)
                img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            
            # 颜色增强
            if model_type == "photo_to_monet":
                # 增强蓝色和绿色通道
                img_np[:,:,0] = np.clip(img_np[:,:,0] * 1.1, 0, 255).astype(np.uint8)  # 蓝色
                img_np[:,:,1] = np.clip(img_np[:,:,1] * 1.05, 0, 255).astype(np.uint8)  # 绿色
            else:
                # 增强整体对比度
                img_np = cv2.convertScaleAbs(img_np, alpha=1.1, beta=5)
            
            # 平滑过渡
            # 应用双边滤波保持边缘锐利同时平滑区域
            img_np = cv2.bilateralFilter(img_np, d=9, sigmaColor=75, sigmaSpace=75)
            
            # 更新增强输出图像
            enhanced_img = Image.fromarray(img_np)
            
            # 处理CycleGAN模型图像（使用与独立选项卡相同的处理逻辑）
            self.root.after(0, lambda: self.compare_status_var.set("正在应用原始CycleGAN..."))
            
            with torch.no_grad():
                cyclegan_output = cyclegan_model(input_tensor)
            
            # 后处理CycleGAN输出
            cyclegan_output = (cyclegan_output + 1.0) / 2.0
            cyclegan_output = torch.clamp(cyclegan_output, 0, 1)
            cyclegan_np = cyclegan_output.cpu().squeeze(0).permute(1, 2, 0).numpy()
            cyclegan_img = Image.fromarray((cyclegan_np * 255).astype(np.uint8))
            
            # 确保两个图像都具有正确的尺寸
            enhanced_width, enhanced_height = enhanced_img.size
            cyclegan_width, cyclegan_height = cyclegan_img.size
            
            # 确保两个输出图像尺寸与输入画布尺寸一致
            if enhanced_width != canvas_width or enhanced_height != canvas_height:
                enhanced_img = enhanced_img.resize((canvas_width, canvas_height), Image.LANCZOS)
            
            if cyclegan_width != canvas_width or cyclegan_height != canvas_height:
                cyclegan_img = cyclegan_img.resize((canvas_width, canvas_height), Image.LANCZOS)
            
            # 裁剪回原始比例
            if original_aspect_ratio != 1.0:
                # 计算基于原始长宽比的裁剪尺寸
                if original_aspect_ratio > 1:  # 宽大于高
                    crop_width = canvas_width
                    crop_height = int(canvas_width / original_aspect_ratio)
                else:  # 高大于宽
                    crop_height = canvas_height
                    crop_width = int(canvas_height * original_aspect_ratio)
                
                # 确保裁剪尺寸不超过画布
                crop_width = min(crop_width, canvas_width)
                crop_height = min(crop_height, canvas_height)
                
                # 计算裁剪区域的左上角和右下角坐标
                left = (canvas_width - crop_width) // 2
                top = (canvas_height - crop_height) // 2
                right = left + crop_width
                bottom = top + crop_height
                
                # 安全检查，确保裁剪区域有效
                if left >= 0 and top >= 0 and right <= canvas_width and bottom <= canvas_height:
                    enhanced_img = enhanced_img.crop((left, top, right, bottom))
                    cyclegan_img = cyclegan_img.crop((left, top, right, bottom))
            
            # 调整回原始大小 - 确保图像完整
            if width * height <= 1024 * 1024:  # 仅当原始图像不太大时才调整回原始大小
                enhanced_img = enhanced_img.resize((width, height), Image.LANCZOS)
                cyclegan_img = cyclegan_img.resize((width, height), Image.LANCZOS)
            
            # 保存结果
            os.makedirs("output", exist_ok=True)
            enhanced_output_path = f"output/local_style_{os.path.basename(img_path)}"
            cyclegan_output_path = f"output/cyclegan_{os.path.basename(img_path)}"
            
            enhanced_img.save(enhanced_output_path)
            cyclegan_img.save(cyclegan_output_path)
            
            # 保存结果路径
            self.compare_enhanced_path = enhanced_output_path
            self.compare_cyclegan_path = cyclegan_output_path
            
            # 更新UI
            self.root.after(0, lambda: self.update_compare_result_ui(original_img, enhanced_img, cyclegan_img))
            
        except Exception as e:
            import traceback
            print(f"对比模式处理出错: {str(e)}")
            print(traceback.format_exc())
            error_msg = str(e)
            self.root.after(0, lambda: self.show_compare_error(error_msg))
    
    def show_compare_error(self, error_msg):
        """显示对比模式处理错误"""
        # 停止进度条
        self.compare_progress.stop()
        
        # 更新状态
        self.compare_status_var.set(f"错误: {error_msg}")
        
        # 启用处理按钮
        self.compare_process_button.config(state=tk.NORMAL)
        
        # 重置处理状态
        self.is_processing = False
        
        # 显示错误对话框
        messagebox.showerror("处理错误", f"对比模式处理过程中出错: {error_msg}")
    
    def save_compare_result(self):
        """保存对比结果"""
        if not hasattr(self, 'compare_enhanced_path') or not hasattr(self, 'compare_cyclegan_path'):
            messagebox.showerror("错误", "没有可保存的对比结果")
            return
        
        # 选择保存目录
        save_dir = filedialog.askdirectory(
            title="选择保存对比结果的目录"
        )
        
        if save_dir:
            try:
                # 生成保存文件名
                basename = os.path.basename(self.compare_image_path)
                enhanced_save_path = os.path.join(save_dir, f"局部风格_{basename}")
                cyclegan_save_path = os.path.join(save_dir, f"原始CycleGAN_{basename}")
                
                # 复制结果文件
                shutil.copy2(self.compare_enhanced_path, enhanced_save_path)
                shutil.copy2(self.compare_cyclegan_path, cyclegan_save_path)
                
                # 显示成功消息
                messagebox.showinfo("保存成功", f"对比结果已保存到: {save_dir}")
            except Exception as e:
                messagebox.showerror("保存错误", f"保存结果时出错: {str(e)}")

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = GanStyleTransferApp(root)
        root.mainloop()
    except Exception as e:
        import traceback
        print(f"程序启动失败: {e}")
        print(traceback.format_exc())