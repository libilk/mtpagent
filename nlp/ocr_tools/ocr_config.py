# -*- coding: utf-8 -*-
"""
OCR配置文件
==========

Tesseract（OCR 引擎：本项目用的开源 OCR 程序）的路径与参数配置。
OCR（光学字符识别）= 从图片里认出文字。

OCR 自检三件套之一（造样本 → 配引擎 → 验识别），另两个是
ocr_tools/scripts/generate_scanned_pdf.py 和 ocr_tools/tests/test_ocr.py，本文件负责"配引擎"。

⚠️ 跑之前先确认本机真能跑起 Tesseract —— 因为本文件只判断"文件在不在"，
   不判断"能不能跑"：

   Windows 下优先用项目自带的 ocr_tools/tesseract.exe。只要这个文件存在
   （**哪怕它缺 DLL、根本起不来**），就不会再去试下面的系统安装路径。
   本仓库现在正是这种情况：ocr_tools/tesseract.exe 在，但一执行就报
   "libtiff-6.dll: cannot open shared object file"；ocr_tools/tessdata/ 目录也不存在。
   后果有两层，都容易被误判成"代码 bug"：
     - setup_tesseract() 仍然返回 True（它只做 os.path.exists 判断），
       "✓ Tesseract配置成功"这句话并不代表引擎可用；
     - 真去识别时引擎起不来，表现是识别出空文本或直接抛错。
   pytesseract 这个 Python 包本身是装好的（0.3.13）——"包在、程序不在"是本机的现状。

另外一个容易白费功夫的点：本文件里的 OCR_LANG 和 OCR_CONFIG 目前全项目无人读取，
真正生效的是 core/file_parser.py 里写死的 resolution=300 与 lang='chi_sim+eng'。
改这里不会改变识别行为。
"""

import os
import sys

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Tesseract可执行文件路径
if sys.platform == 'win32':
    # Windows系统 - 优先使用项目内的Tesseract
    TESSERACT_CMD = os.path.join(PROJECT_ROOT, 'ocr_tools', 'tesseract.exe')

    # 如果项目内没有，尝试系统安装路径
    # 注意判据只看"文件存在"，没验证可执行 —— 项目内那个坏掉的 exe 会挡住这条兜底路径
    if not os.path.exists(TESSERACT_CMD):
        TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    # Linux/Mac系统
    TESSERACT_CMD = 'tesseract'

# Tesseract数据目录
# 语言包（chi_sim/eng）就装在这里；目录不存在时 TESSDATA_PREFIX 不会被设置，
# 引擎即使能跑也找不到中文包 —— 这是"识别出空文本"的另一个常见原因。
TESSDATA_DIR = os.path.join(PROJECT_ROOT, 'ocr_tools', 'tessdata')

# OCR语言配置
# （本项当前无消费方，见文件头说明；生效值写死在 file_parser.py）
OCR_LANG = 'chi_sim+eng'  # 中文简体 + 英文

# OCR参数
# （同上，当前无消费方 —— 改 resolution 不会改变渲染精度）
OCR_CONFIG = {
    'resolution': 300,  # DPI分辨率
    'min_text_length': 50,  # 触发OCR的最小文本长度
}


def setup_tesseract():
    """
    配置pytesseract使用项目内的Tesseract

    Returns:
        bool: 配置是否成功

    注意返回 True 的语义很弱：它只表示"exe 路径存在且已写给 pytesseract"，
    **不代表 Tesseract 真能执行**（可执行性没被检查，见文件头）。
    想确认真能用，得看本文件 __main__ 里 get_tesseract_version() 那一步有没有报错。
    """
    try:
        import pytesseract

        # 设置Tesseract命令路径
        if os.path.exists(TESSERACT_CMD):
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

            # 设置TESSDATA环境变量
            if os.path.exists(TESSDATA_DIR):
                os.environ['TESSDATA_PREFIX'] = TESSDATA_DIR

            return True
        else:
            print(f"警告: Tesseract未找到: {TESSERACT_CMD}")
            return False

    except ImportError:
        print("错误: pytesseract未安装")
        return False


def get_tesseract_info():
    """
    获取Tesseract信息

    Returns:
        dict: Tesseract配置信息
    """
    # 这两个 exists 布尔比 setup_tesseract() 的返回值诚实：它们如实反映文件/目录在不在。
    # 快速体检就看 tessdata_exists —— False 基本可以断定中文识别必然失败。
    return {
        'tesseract_cmd': TESSERACT_CMD,
        'tessdata_dir': TESSDATA_DIR,
        'ocr_lang': OCR_LANG,
        'tesseract_exists': os.path.exists(TESSERACT_CMD),
        'tessdata_exists': os.path.exists(TESSDATA_DIR),
    }


if __name__ == '__main__':
    # 设置UTF-8编码
    if sys.platform == 'win32':
        import codecs
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

    # 测试配置
    print("OCR配置信息:")
    print("-" * 60)

    info = get_tesseract_info()
    for key, value in info.items():
        print(f"{key}: {value}")

    print("-" * 60)

    if setup_tesseract():
        print("✓ Tesseract配置成功")

        # 下面这段才是真正的"体检"：它真的去执行 tesseract.exe。
        # 引擎缺失时这里会报 CalledProcessError（不是配置失败），
        # 所以看到"配置成功"紧接着"获取Tesseract信息失败"，就知道是引擎跑不起来。
        try:
            import pytesseract
            version = pytesseract.get_tesseract_version()
            print(f"✓ Tesseract版本: {version}")

            langs = pytesseract.get_languages()
            print(f"✓ 可用语言: {', '.join(langs)}")
        except Exception as e:
            print(f"✗ 获取Tesseract信息失败: {e}")
    else:
        print("✗ Tesseract配置失败")
