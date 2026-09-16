# -*- coding: utf-8 -*-
"""
生成扫描版PDF合同
==================

将文本合同转换为模拟扫描版的PDF（带噪点、倾斜等效果）
用于测试OCR功能
"""

import os
import sys
from pathlib import Path

def generate_scanned_pdf():
    """生成扫描版PDF"""

    print("=" * 60)
    print("生成扫描版PDF合同（用于OCR测试）")
    print("=" * 60)

    # 检查依赖
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageFilter
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        import random
    except ImportError as e:
        print(f"\n❌ 缺少依赖库: {e}")
        print("\n请安装依赖:")
        print("  pip install pillow reportlab")
        return

    # 合同文本文件
    contracts = [
        "data/contracts/待审核合同_2026_004_扫描版.txt",
        "data/contracts/历史合同_2023_003_扫描版.txt"
    ]

    for txt_file in contracts:
        if not os.path.exists(txt_file):
            print(f"\n⚠ 文件不存在: {txt_file}")
            continue

        # 读取文本
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read()

        # 输出PDF文件名
        pdf_file = txt_file.replace('.txt', '.pdf')

        print(f"\n处理: {txt_file}")
        print(f"  → {pdf_file}")

        try:
            # 创建图像（模拟扫描效果）
            img_width, img_height = 2480, 3508  # A4 @ 300 DPI
            img = Image.new('RGB', (img_width, img_height), color='white')
            draw = ImageDraw.Draw(img)

            # 使用系统字体（简化处理）
            try:
                # Windows系统字体
                font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 40)
                font_small = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 30)
            except:
                # 如果找不到字体，使用默认字体
                font = ImageFont.load_default()
                font_small = ImageFont.load_default()
                print("  ⚠ 使用默认字体（可能无法显示中文）")

            # 绘制文本（分行）
            y_position = 100
            line_height = 60

            for line in text.split('\n'):
                if y_position > img_height - 100:
                    break  # 超出页面

                # 添加轻微倾斜和噪点效果
                x_offset = random.randint(-5, 5)
                draw.text((150 + x_offset, y_position), line, fill='black', font=font_small)
                y_position += line_height

            # 添加扫描效果
            # 1. 轻微模糊
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

            # 2. 添加噪点
            pixels = img.load()
            for i in range(0, img_width, 10):
                for j in range(0, img_height, 10):
                    if random.random() < 0.01:  # 1%的噪点
                        noise = random.randint(-20, 20)
                        r, g, b = pixels[i, j]
                        pixels[i, j] = (
                            max(0, min(255, r + noise)),
                            max(0, min(255, g + noise)),
                            max(0, min(255, b + noise))
                        )

            # 保存为临时图像
            temp_img = pdf_file.replace('.pdf', '_temp.png')
            img.save(temp_img, 'PNG', dpi=(300, 300))

            # 转换为PDF
            c = canvas.Canvas(pdf_file, pagesize=A4)
            c.drawImage(temp_img, 0, 0, width=A4[0], height=A4[1])
            c.save()

            # 删除临时图像
            os.remove(temp_img)

            print(f"  ✓ 生成成功")

        except Exception as e:
            print(f"  ❌ 生成失败: {e}")

    print("\n" + "=" * 60)
    print("✓ 扫描版PDF生成完成")
    print("\n说明:")
    print("  - 这些PDF模拟了扫描版文档的效果（噪点、模糊）")
    print("  - 用于测试OCR功能")
    print("  - 需要安装Tesseract OCR才能识别")
    print("=" * 60)


if __name__ == "__main__":
    generate_scanned_pdf()
