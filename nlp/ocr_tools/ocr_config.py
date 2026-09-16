# -*- coding: utf-8 -*-
"""
OCR配置文件
==========

配置Tesseract OCR的路径和参数
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
    if not os.path.exists(TESSERACT_CMD):
        TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    # Linux/Mac系统
    TESSERACT_CMD = 'tesseract'

# Tesseract数据目录
TESSDATA_DIR = os.path.join(PROJECT_ROOT, 'ocr_tools', 'tessdata')

# OCR语言配置
OCR_LANG = 'chi_sim+eng'  # 中文简体 + 英文

# OCR参数
OCR_CONFIG = {
    'resolution': 300,  # DPI分辨率
    'min_text_length': 50,  # 触发OCR的最小文本长度
}


def setup_tesseract():
    """
    配置pytesseract使用项目内的Tesseract

    Returns:
        bool: 配置是否成功
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
