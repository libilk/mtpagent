# -*- coding: utf-8 -*-
"""
生成扫描版PDF合同
==================

OCR 自检三件套之一（造样本 → 配引擎 → 验识别），本文件负责第一步"造样本"，
另两个是 ocr_tools/ocr_config.py 与 ocr_tools/tests/test_ocr.py。

为什么非要"造假样本"：真扫描件不好找，而 PDF 解析对文字版和扫描版的走法是两条路 ——
文字版直接抽文字层就完事，只有抽不到时才会把页面渲染成图、降级交给 OCR。
所以要验证 OCR 那条降级路径，就必须造一个"里面一个字符都没有、只有一张图"的 PDF。
本脚本的做法：把合同文本用 PIL 画成 2480×3508 的位图（A4 @300 DPI），
加一点高斯模糊和 1% 噪点模拟扫描噪声，再用 reportlab 把这张图贴进 PDF。

⚠️ 别高估 OCR 在主链路里的位置：它只作为 PDF 的降级手段存在，
   而且 core/file_parser.py **不解析图片文件**（.png/.jpg 不在支持列表里），
   图片输入走的是 vqa_agent 那条路。本脚本不参与线上流程，纯粹是测试工装。

用法：先跑本脚本生成 PDF，再跑 ocr_tools/tests/test_ocr.py 验证。
      需要 pillow + reportlab（缺了会打印 pip 安装提示）。
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
            # 字体路径写死了 Windows 的黑体。找不到时会静默退化到 PIL 默认字体，
            # 而默认字体画不出中文 —— 那时生成的样本近乎空白，OCR 自然认不出东西。
            # 若 OCR 结果为空，先确认这一步是不是走了 except 分支（会打印下面的警告）。
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

            # 注意这里超出页面就 break 而不是分页：长合同会被**截断**。
            # 所以 OCR 出来的字数比原文少是设计如此，不是 OCR 漏字。
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
            # 以 10 像素为步长采样而不是逐像素：这张图有 870 万个像素，
            # 逐点遍历（还要 load/put 像素）会慢到不可接受，而模拟扫描噪点不需要那么密。
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
            # 关键是"贴图"而不是"写字"：reportlab 只把它当一张图片放进页面，
            # PDF 里因此不含任何文字层 —— 这正是逼出 OCR 降级路径的必要条件。
            # （先后落盘成临时 PNG 是因为 drawImage 要的是文件，不是内存里的 PIL 对象。）
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
