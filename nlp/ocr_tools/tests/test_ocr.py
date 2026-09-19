# -*- coding: utf-8 -*-
"""
OCR功能测试
===========

OCR 自检三件套之三（造样本 → 配引擎 → 验识别），负责最后一步。
样本由 ocr_tools/scripts/generate_scanned_pdf.py 生成，引擎路径由 ocr_tools/ocr_config.py 配置。

三段测试，逐层收窄：
  1. test_ocr_installation —— 环境检查：pytesseract 包、Tesseract 程序、语言包、Pillow、pdfplumber
  2. test_ocr_recognition  —— 现场画一张图跑识别，验证"引擎真能出字"
  3. test_pdf_ocr          —— 拿扫描版 PDF 跑 init_vector_db 的提取函数，验证"PDF→OCR 这条链"

⚠️ 这里打印的 ✓ 不能当成"OCR 可用"的证据，有三处会误导：
  - 第 1 段一开始的 "✓ pytesseract已安装" 只说明 Python 包在，与 Tesseract 程序无关；
    本机正是"包在、程序坏"的状态。
  - 第 2 段只要求 image_to_string 不抛异常，**识别出空文本也算通过**（它不看结果对不对）。
  - 第 3 段的 import 路径是错的：它从 ocr_tools/tests/ 下去找 init_vector_db，
    而该文件实际在 tools/scripts/ —— ImportError 被 except 吞掉，只打印"处理失败"，
    看着像 OCR 坏了，其实是这行脚本自己的问题。所以第 3 段在当前仓库**必然报失败**。
  想知道 Tesseract 到底行不行，最可靠的还是直接跑 ocr_tools/ocr_config.py，
  看它有没有打印出版本号。

另注：本文件不是 pytest 用例（没有 test_ 前缀的收集约定问题，但靠的是 main() 手动串起来），
两个 test_ 函数的返回值只有第 1 段被 main() 用到，第 2、3 段的成败不影响整体退出状态。
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
        # 这行 sys.path 是为了能 `from ocr_tools...`：本文件在 ocr_tools/tests/ 下，
        # 往上三层才是项目根（nlp/）。
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from ocr_tools.ocr_config import setup_tesseract
        # 这个 True 只代表"exe 路径存在"，不代表引擎能跑 —— 详见 ocr_config.py 文件头
        if setup_tesseract():
            print("  ✓ 已配置项目中的Tesseract")
    except ImportError:
        print("  ❌ pytesseract未安装")
        print("  请运行: pip install pytesseract")
        return False

    # 2. 检查Tesseract引擎
    # 本段是整份测试里唯一能真正证伪"引擎缺失"的地方：
    # 它去执行 tesseract.exe，失败才说明程序不可用。
    # 注意失败后又去猜几个常见安装路径，都找不到就 return False —— main() 会就此结束，
    # 后面的识别测试根本不会跑。所以只看到"环境检查失败"时别以为识别也测过了。
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
    # 语言包（chi_sim/eng）来自 Tesseract 自己的安装数据，不是 pip 包的一部分，
    # 所以"pytesseract 装好了"和"中文语言包在"是两回事。
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
        # lang 这里直接写死 'chi_sim+eng'，没读 ocr_config.OCR_LANG ——
        # 所以改那个配置项不会影响本测试（它本来也没被别处消费）。
        print("\n[步骤2] OCR识别...")
        ocr_text = pytesseract.image_to_string(img, lang='chi_sim+eng')

        print(f"\n  原始文本: {test_text}")
        print(f"  识别结果: {ocr_text.strip()}")

        # 清理：这行只在成功路径上执行。一旦上面识别抛异常就直接跳到 except，
        # 临时图片会留在 ocr_tools/tests/ 下（仓库里现在正躺着一个 test_ocr_image.png）——
        # 看到它不用奇怪，那是一次失败运行的残留。
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
                # 注意：这里加的是本目录（ocr_tools/tests/），但 init_vector_db.py
                # 实际在 tools/scripts/ —— 所以这行 import 必然 ImportError，
                # 被外层的 except 吞掉后表现为"处理失败"。是本测试自己的路径写错，
                # 不是 OCR 或 PDF 的问题（改这里需要连逻辑一起改，故仅记录）。
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
    # 只有这一段的返回值被用来决定要不要继续；后面两段无论成败都会被跑完。
    # 另外下面提示里那句 "python init_vector_db.py" 路径是过期的 ——
    # 脚本现在在 tools/scripts/init_vector_db.py，照原样敲会找不到文件。
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
