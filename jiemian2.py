import os
import sys
import glob
import tqdm
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont  # 新增ImageFont导入
import torch
import torchvision
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib

matplotlib.use('Agg')  # 使用Agg后端，避免显示问题
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 预训练模型
import timm
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReIDApp:
    def __init__(self, root):
        self.root = root
        self.root.title("行人重识别系统")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)

        self.style = ttk.Style()
        self.style.configure("TButton", font=("SimHei", 12))
        self.style.configure("TLabel", font=("SimHei", 12))
        self.style.configure("Header.TLabel", font=("SimHei", 16, "bold"))

        self.input_image_path = None
        self.output_image_path = None
        self.input_dir = r"C:\Users\sun\Desktop\xrcsb\transreid\TransReID-market\in_put"
        self.output_dir = r"C:\Users\sun\Desktop\xrcsb\transreid\TransReID-market\out_put"

        self.create_menu()
        self.create_main_frame()
        self.model = self.load_model()

    def create_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="选择输入图片", command=self.select_input_image, accelerator="Ctrl+I")
        file_menu.add_command(label="处理图片", command=self.process_image, accelerator="Ctrl+P")
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit, accelerator="Ctrl+Q")
        menubar.add_cascade(label="文件", menu=file_menu)

        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_command(label="设置输入目录", command=self.set_input_directory)
        settings_menu.add_command(label="设置输出目录", command=self.set_output_directory)
        menubar.add_cascade(label="设置", menu=settings_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="关于", command=self.show_about)
        menubar.add_cascade(label="帮助", menu=help_menu)

        self.root.config(menu=menubar)
        self.root.bind("<Control-i>", lambda event: self.select_input_image())
        self.root.bind("<Control-p>", lambda event: self.process_image())
        self.root.bind("<Control-q>", lambda event: self.root.quit())

    def create_main_frame(self):
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill=tk.X)

        self.select_btn = ttk.Button(top_frame, text="选择输入图片", command=self.select_input_image)
        self.select_btn.pack(side=tk.LEFT, padx=5)

        self.process_btn = ttk.Button(top_frame, text="处理图片", command=self.process_image, state=tk.DISABLED)
        self.process_btn.pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.image_frame = ttk.LabelFrame(self.root, text="图片显示", padding=10)
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.input_frame = ttk.LabelFrame(self.image_frame, text="输入图片", padding=5)
        self.input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.input_label = ttk.Label(self.input_frame, text="请选择一张图片")
        self.input_label.pack(expand=True)

        self.output_frame = ttk.LabelFrame(self.image_frame, text="输出结果", padding=5)
        self.output_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.output_label = ttk.Label(self.output_frame, text="处理后的结果将显示在这里")
        self.output_label.pack(expand=True)

    def load_model(self):
        try:
            from config import cfg
            from model import make_model
            from utils.logger import setup_logger

            config_file = r"configs/Market/vit_transreid_stride.yml"
            cfg.merge_from_file(config_file)
            cfg.freeze()

            output_dir = cfg.OUTPUT_DIR
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            logger = setup_logger("transreid", output_dir, if_train=False)
            logger.info("加载模型中...")

            os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

            num_classes = 751  # Market1501数据集的类别数
            camera_num = 6  # Market1501数据集的摄像头数
            view_num = 1  # 视角数

            model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
            model.load_param(cfg.TEST.WEIGHT)
            model.to(device)
            model.eval()

            self.status_var.set("模型加载成功")
            return model
        except Exception as e:
            self.status_var.set(f"模型加载失败: {str(e)}")
            messagebox.showerror("错误", f"加载模型时出错: {str(e)}")
            return None

    def select_input_image(self, event=None):
        if not os.path.exists(self.input_dir):
            os.makedirs(self.input_dir)

        file_path = filedialog.askopenfilename(
            initialdir=self.input_dir,
            title="选择图片",
            filetypes=[("图片文件", "*.png;*.jpg;*.jpeg;*.bmp")]
        )

        if file_path:
            try:
                img = Image.open(file_path)
                img.close()
                self.input_image_path = file_path
                self.display_input_image()
                self.process_btn.config(state=tk.NORMAL)
                self.status_var.set(f"已选择图片: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("错误", "选择的文件不是有效的图片")

    def display_input_image(self):
        for widget in self.input_frame.winfo_children():
            widget.destroy()

        try:
            img = Image.open(self.input_image_path)
            img.thumbnail((500, 500))
            photo = ImageTk.PhotoImage(img)

            label = ttk.Label(self.input_frame, image=photo)
            label.image = photo
            label.pack(expand=True)

            info_label = ttk.Label(self.input_frame, text=f"图片: {os.path.basename(self.input_image_path)}")
            info_label.pack(pady=5)
        except Exception as e:
            messagebox.showerror("错误", f"无法显示图片: {str(e)}")

    def process_image(self, event=None):
        if self.input_image_path and self.model:
            try:
                self.status_var.set("正在处理图片...")
                self.root.update()
                result = self.perform_reid()

                if result:
                    self.display_result()
                    self.status_var.set("处理完成")
                else:
                    self.status_var.set("处理失败")
                    messagebox.showerror("错误", "处理图片时出错")

            except Exception as e:
                self.status_var.set(f"错误: {str(e)}")
                messagebox.showerror("错误", f"处理图片时出错: {str(e)}")

    def perform_reid(self):
        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            input_vector = self.extract_feature(self.input_image_path)
            all_vectors_path = r"C:\Users\sun\Desktop\xrcsb\transreid\TransReID-market\save_model\model_mal.npy"

            if not os.path.exists(all_vectors_path):
                raise FileNotFoundError("特征数据库文件不存在")

            all_vectors = np.load(all_vectors_path, allow_pickle=True).item()
            similarities = {}

            for img_path, vector in all_vectors.items():
                if not os.path.exists(img_path):
                    continue
                try:
                    sim = np.dot(input_vector, vector) / (np.linalg.norm(input_vector) * np.linalg.norm(vector))
                    similarities[img_path] = sim
                except Exception as e:
                    print(f"计算相似度出错: {img_path}, {str(e)}")

            k = 7
            sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
            result_image_path = self.create_result_image(self.input_image_path, sorted_similarities)
            return result_image_path

        except Exception as e:
            print(f"重识别过程出错: {str(e)}")
            raise e

    def extract_feature(self, image_path):
        try:
            img_rgb = Image.open(image_path).convert("RGB")
            image = img_rgb.resize((128, 256), Image.LANCZOS)
            image = torchvision.transforms.ToTensor()(image)

            trainset_mean = [0.5, 0.5, 0.5]
            trainset_std = [0.5, 0.5, 0.5]
            image = torchvision.transforms.Normalize(mean=trainset_mean, std=trainset_std)(image).unsqueeze(0).to(
                device)

            with torch.no_grad():
                self.model.eval()
                camids = torch.zeros(1, dtype=torch.int64).to(device)
                target_view = torch.ones(1, dtype=torch.int64).to(device)
                features = self.model(image, cam_label=camids, view_label=target_view)
                vec = features.squeeze().cpu().numpy()

            img_rgb.close()
            return vec

        except Exception as e:
            print(f"提取特征出错: {str(e)}")
            raise e

    def create_result_image(self, query_image, similar_images):
        try:
            query_img = Image.open(query_image)
            query_img = query_img.resize((256, 512))

            result_width = (len(similar_images) + 1) * 256
            result_height = 512 + 30  # 增加文本区域高度
            result_img = Image.new('RGB', (result_width, result_height), (255, 255, 255))
            draw = ImageDraw.Draw(result_img)

            # 处理字体路径（优先使用系统字体）
            font_path = "simhei.ttf"
            if not os.path.exists(font_path):
                font_path = os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts/simhei.ttf")
            font = ImageFont.truetype(font_path, 14) if os.path.exists(font_path) else ImageFont.load_default()

            # 粘贴查询图片
            result_img.paste(query_img, (0, 0))
            draw.text((10, 512 + 5), "查询图片", font=font, fill=(0, 0, 0))

            # 粘贴相似图片
            for i, (img_path, score) in enumerate(similar_images):
                try:
                    img = Image.open(img_path)
                    img = img.resize((256, 512))
                    x_pos = (i + 1) * 256
                    result_img.paste(img, (x_pos, 0))

                    # 显示相似度和文件名
                    draw.text((x_pos + 10, 10), f"相似度: {score:.4f}", font=font, fill=(255, 0, 0))
                    img_name = os.path.basename(img_path)
                    draw.text((x_pos + 10, 512 + 5), img_name, font=font, fill=(0, 0, 0))
                except Exception as e:
                    print(f"处理相似图片 {img_path} 出错: {str(e)}")

            result_name = os.path.basename(query_image)
            result_path = os.path.join(self.output_dir, f"result_{result_name}")
            result_img.save(result_path)
            self.output_image_path = result_path
            return result_path

        except Exception as e:
            print(f"创建结果图像出错: {str(e)}")
            raise e

    def display_result(self):
        for widget in self.output_frame.winfo_children():
            widget.destroy()

        if self.output_image_path and os.path.exists(self.output_image_path):
            try:
                img = Image.open(self.output_image_path)
                img.thumbnail((500, 600))  # 调整显示比例
                photo = ImageTk.PhotoImage(img)

                label = ttk.Label(self.output_frame, image=photo)
                label.image = photo
                label.pack(expand=True)

                info_label = ttk.Label(self.output_frame, text=f"结果: {os.path.basename(self.output_image_path)}")
                info_label.pack(pady=5)
            except Exception as e:
                messagebox.showerror("错误", f"无法显示结果图片: {str(e)}")
        else:
            error_label = ttk.Label(self.output_frame, text="没有找到处理结果", foreground="red")
            error_label.pack(expand=True)

    def set_input_directory(self):
        directory = filedialog.askdirectory(initialdir=self.input_dir, title="选择输入目录")
        if directory:
            self.input_dir = directory
            self.status_var.set(f"输入目录已设置为: {directory}")

    def set_output_directory(self):
        directory = filedialog.askdirectory(initialdir=self.output_dir, title="选择输出目录")
        if directory:
            self.output_dir = directory
            self.status_var.set(f"输出目录已设置为: {directory}")

    def show_about(self):
        messagebox.showinfo(
            "关于",
            "行人重识别系统\n\n"
            "版本: 1.0\n"
            "描述: 基于深度学习的行人重识别应用程序\n"
            "功能: 输入一张行人图片，找出数据库中最相似的行人图片\n\n"
            "开发者: Your Name\n"
            "日期: 2025年"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = ReIDApp(root)
    root.mainloop()