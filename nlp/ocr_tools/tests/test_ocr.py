# -*- coding: utf-8 -*-
"""
OCR功能测试
===========

测试Tesseract OCR是否正常工作
"""

import os
import sys
import logging

# 设置Windows控制台UTF-8编码
if sys.platform == 'win32':
    import codecs
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_ocr_installation():
    """测试OCR安装"""

    print("=" * 60)
    print("OCR功能测试")
    print("=" * 60)

    # 1. 检查pytesseract
    print("\n[步骤1] 检查pytesseract...")
    try:
        import pytesseract
        print("  ✓ pytesseract已安装")

        # 使用项目中的Tesseract配置
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from ocr_tools.ocr_config import setup_tesseract
        if setup_tesseract():
            print("  ✓ 已配置项目中的Tesseract")
    except ImportError:
        print("  ❌ pytesseract未安装")
        print("  请运行: pip install pytesseract")
        return False

    # 2. 检查Tesseract引擎
    print("\n[步骤2] 检查Tesseract OCR引擎...")
    try:
        version = pytesseract.get_tesseract_version()
        print(f"  ✓ Tesseract版本: {version}")
    except Exception as e:
        print(f"  ❌ Tesseract未在PATH中找到")
        print(f"  错误: {e}")

        # 尝试常见安装路径
        print("\n  尝试查找常见安装路径...")
        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Tesseract-OCR\tesseract.exe",
        ]

        found = False
        for path in possible_paths:
            if os.path.exists(path):
                print(f"  ✓ 找到Tesseract: {path}")
                pytesseract.pytesseract.tesseract_cmd = path
                try:
                    version = pytesseract.get_tesseract_version()
                    print(f"  ✓ Tesseract版本: {version}")
                    found = True
                    break
                except:
                    pass

        if not found:
            print("\n  ❌ 未找到Tesseract安装")
            print("\n  解决方法:")
            print("  1. 下载安装包:")
            print("     https://github.com/UB-Mannheim/tesseract/wiki")
            print("  2. 运行安装程序（建议安装到默认路径）")
            print("  3. 安装时勾选 'Add to PATH' 选项")
            print("  4. 重启终端后重新运行此脚本")
            return False

    # 3. 检查语言包
    print("\n[步骤3] 检查语言包...")
    try:
        langs = pytesseract.get_languages()
        print(f"  已安装语言包: {', '.join(langs)}")

        if 'chi_sim' in langs:
            print("  ✓ 中文简体语言包已安装")
        else:
            print("  ⚠ 中文简体语言包未安装")
            print("  请重新安装Tesseract并勾选'Chinese - Simplified'")

        if 'eng' in langs:
            print("  ✓ 英文语言包已安装")
        else:
            print("  ⚠ 英文语言包未安装")

    except Exception as e:
        print(f"  ❌ 无法获取语言包列表: {e}")
        return False

    # 4. 检查其他依赖
    print("\n[步骤4] 检查其他依赖...")

    try:
        from PIL import Image
        print("  ✓ Pillow已安装")
    except ImportError:
        print("  ❌ Pillow未安装")
        print("  请运行: pip install Pillow")
        return False

    try:
        import pdfplumber
        print("  ✓ pdfplumber已安装")
    except ImportError:
        print("  ❌ pdfplumber未安装")
        print("  请运行: pip install pdfplumber")
        return False

    print("\n" + "=" * 60)
    print("✓ OCR环境检查通过")
    print("=" * 60)

    return True


def test_ocr_recognition():
    """测试OCR识别功能"""

    print("\n\n" + "=" * 60)
    print("OCR识别功能测试")
    print("=" * 60)

    try:
        import pytesseract
        from PIL import Image, ImageDraw, ImageFont

        # 使用项目中的Tesseract配置
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from ocr_tools.ocr_config import setup_tesseract
        setup_tesseract()

        # 创建测试图像
        print("\n[步骤1] 创建测试图像...")
        img = Image.new('RGB', (800, 200), color='white')
        draw = ImageDraw.Draw(img)

        # 绘制测试文本
        test_text = "测试文本 Test OCR 1234567890"

        try:
            # 尝试使用中文字体
            font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 40)
        except:
            font = ImageFont.load_default()
            print("  ⚠ 使用默认字体（可能无法显示中文）")

        draw.text((50, 80), test_text, fill='black', font=font)

        # 保存测试图像
        test_img_path = "test_ocr_image.png"
        img.save(test_img_path)
        print(f"  ✓ 测试图像已保存: {test_img_path}")

        # OCR识别
        print("\n[步骤2] OCR识别...")
        ocr_text = pytesseract.image_to_string(img, lang='chi_sim+eng')

        print(f"\n  原始文本: {test_text}")
        print(f"  识别结果: {ocr_text.strip()}")

        # 清理
        os.remove(test_img_path)

        print("\n" + "=" * 60)
        print("✓ OCR识别测试完成")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ OCR识别测试失败: {e}")
        return False


def test_pdf_ocr():
    """测试PDF OCR功能"""

    print("\n\n" + "=" * 60)
    print("PDF OCR功能测试")
    print("=" * 60)

    # 检查是否有扫描版PDF
    pdf_files = [
        "data/contracts/待审核合同_2026_004_扫描版.pdf",
        "data/contracts/历史合同_2023_003_扫描版.pdf"
    ]

    found_pdf = False
    for pdf_file in pdf_files:
        if os.path.exists(pdf_file):
            found_pdf = True
            print(f"\n测试文件: {pdf_file}")

            try:
                # 使用init_vector_db中的函数
                sys.path.insert(0, os.path.dirname(__file__))
                from init_vector_db import extract_pdf_with_ocr

                print("  开始OCR识别...")
                text, tables = extract_pdf_with_ocr(pdf_file)

                if text:
                    print(f"  ✓ 识别成功")
                    print(f"  文本长度: {len(text)} 字符")
                    print(f"  表格数量: {len(tables.split('【')) - 1 if tables else 0}")
                    print(f"\n  文本预览（前200字符）:")
                    print(f"  {text[:200]}...")
                else:
                    print("  ❌ 识别失败")

            except Exception as e:
                print(f"  ❌ 处理失败: {e}")

    if not found_pdf:
        print("\n⚠ 未找到扫描版PDF测试文件")
        print("请先运行: python generate_scanned_pdf.py")

    print("\n" + "=" * 60)
    print("✓ PDF OCR测试完成")
    print("=" * 60)


def main():
    """主函数"""

    # 测试1：检查安装
    if not test_ocr_installation():
        print("\n❌ OCR环境检查失败，请先安装必要的依赖")
        return

    # 测试2：OCR识别
    test_ocr_recognition()

    # 测试3：PDF OCR
    test_pdf_ocr()

    print("\n\n" + "=" * 60)
    print("所有测试完成")
    print("=" * 60)
    print("\n下一步:")
    print("  1. 如果测试通过，运行: python init_vector_db.py")
    print("  2. 如果测试失败，查看 OCR_SETUP.md 获取帮助")


if __name__ == "__main__":
    main()
