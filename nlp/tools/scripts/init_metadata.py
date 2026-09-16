# -*- coding: utf-8 -*-
"""
自动生成并应用元数据
==================

完全自动化：扫描 -> 推断 -> 生成 -> 应用
"""

import os
import sys
import codecs
import json
from pathlib import Path

# 设置Windows控制台UTF-8编码
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


def infer_domain_from_filename(filename: str) -> str:
    """
    从文件名推断领域

    Args:
        filename: 文件名

    Returns:
        推断的领域
    """
    filename_lower = filename.lower()

    # 技术相关关键词
    tech_keywords = ['tech', 'api', 'code', 'algorithm', 'database', 'vector',
                     'embedding', 'llm', 'rag', 'python', 'java']

    # 销售相关关键词
    sales_keywords = ['sales', 'product', 'price', 'customer', 'deal', 'revenue']

    # 客服相关关键词
    cs_keywords = ['service', 'support', 'ticket', 'faq', 'help']

    for keyword in tech_keywords:
        if keyword in filename_lower:
            return "tech"

    for keyword in sales_keywords:
        if keyword in filename_lower:
            return "sales"

    for keyword in cs_keywords:
        if keyword in filename_lower:
            return "customer_service"

    return "general"


def infer_audiences_from_domain(domain: str) -> list:
    """
    从领域推断受众

    Args:
        domain: 领域

    Returns:
        受众列表
    """
    domain_to_audiences = {
        "tech": ["developer"],
        "sales": ["sales_rep"],
        "product": ["product_manager"],
        "customer_service": ["customer_service_rep"],
        "general": ["all"]
    }

    return domain_to_audiences.get(domain, ["all"])


def extract_title_from_file(file_path: Path) -> str:
    """
    从文件中提取标题

    Args:
        file_path: 文件路径

    Returns:
        标题
    """
    try:
        ext = file_path.suffix.lower()

        if ext in ['.md', '.txt']:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                # 去掉Markdown标题符号
                title = first_line.lstrip('#').strip()
                return title if title else file_path.stem

        # 其他格式直接使用文件名
        return file_path.stem.replace('_', ' ').title()

    except Exception:
        return file_path.stem.replace('_', ' ').title()


def auto_generate_metadata(docs_dir: str = "documents"):
    """
    自动生成元数据

    Args:
        docs_dir: 文档目录

    Returns:
        元数据字典
    """
    print("=" * 60)
    print("扫描文档并生成元数据")
    print("=" * 60)

    docs_path = Path(docs_dir)
    if not docs_path.exists():
        print(f"错误：文档目录不存在: {docs_dir}")
        return {}

    # 支持的文件扩展名
    supported_extensions = ['.md', '.txt', '.pdf', '.docx', '.xlsx', '.csv', '.json']

    metadata_dict = {}

    # 遍历文档目录
    for file_path in docs_path.rglob('*'):
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()
        if ext not in supported_extensions:
            continue

        # 跳过samples目录
        if 'samples' in file_path.parts:
            continue

        # 生成doc_id（使用文件名，不含扩展名）
        doc_id = file_path.stem

        # 推断领域
        domain = infer_domain_from_filename(file_path.name)

        # 推断受众
        audiences = infer_audiences_from_domain(domain)

        # 提取标题
        title = extract_title_from_file(file_path)

        # 生成元数据
        metadata_dict[doc_id] = {
            "doc_id": doc_id,
            "title": title,
            "domain": domain,
            "audiences": audiences,
            "tags": [],
            "priority": 5
        }

        print(f"✓ {file_path.name} -> {domain} ({', '.join(audiences)})")

    return metadata_dict


def auto_apply_metadata():
    """
    自动生成并应用元数据
    """

    print("自动生成元数据")


    # 1. 生成元数据
    metadata = auto_generate_metadata()

    if not metadata:
        print("❌ 没有找到文档")
        return

    # 2. 保存为正式文件
    output_file = "data/document_metadata.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"✅ 元数据已自动生成: {output_file}")
    print("=" * 60)
    print(f"\n共生成 {len(metadata)} 个文档的元数据")
    print("\n自动判断规则：")
    print("  • 文件名包含 vector/database/llm/rag/api → tech (developer)")
    print("  • 文件名包含 sales/product/price → sales (sales_rep)")
    print("  • 文件名包含 faq/service/support → customer_service (customer_service_rep)")
    print("  • 其他 → general (all)")
    print("\n下一步：")
    print("  python init_vector_db.py  # 初始化向量数据库")
    print("  python main.py            # 启动系统")


if __name__ == "__main__":
    auto_apply_metadata()
