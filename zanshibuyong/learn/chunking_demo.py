# -*- coding: utf-8 -*-
"""
结构感知切分示例：一个文档如何被切成块
运行: python learn/chunking_demo.py
"""
import sys
import re
import codecs
from dataclasses import dataclass
from enum import Enum

# Windows 控制台 UTF-8
if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)


class ElementType(str, Enum):
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    CODE = "code"


@dataclass
class Element:
    type: ElementType
    text: str
    level: int = 0
    heading_path: str = ""


# ============ 第1步：解析成元素树（简化版，只处理Markdown） ============
def parse_markdown(text):
    elements = []
    heading_path = []
    lines = text.split("\n")
    i = 0
    in_code = False
    code_buf, table_buf, para_buf = [], [], []

    def flush_para():
        p = " ".join(para_buf).strip()
        if p:
            elements.append(Element(ElementType.PARAGRAPH, p, heading_path=" > ".join(heading_path)))
        para_buf.clear()

    def flush_table():
        rows = []
        for line in table_buf:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-+:?", c) for c in cells):  # 跳过表头分隔行
                continue
            rows.append(cells)
        headers, data = rows[0], rows[1:]
        md = " | ".join(headers) + "\n" + "\n".join(" | ".join(r) for r in data)
        elements.append(Element(ElementType.TABLE, md, heading_path=" > ".join(heading_path)))
        table_buf.clear()

    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            if in_code:
                elements.append(Element(ElementType.CODE, "\n".join(code_buf), heading_path=" > ".join(heading_path)))
                code_buf.clear(); in_code = False
            else:
                flush_para(); in_code = True
            i += 1; continue
        if in_code:
            code_buf.append(line); i += 1; continue

        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            flush_para()
            level = len(m.group(1))
            heading_path = heading_path[:level - 1] + [m.group(2).strip()]
            elements.append(Element(ElementType.HEADING, m.group(2).strip(), level=level, heading_path=" > ".join(heading_path)))
            i += 1; continue

        if line.strip().startswith("|"):
            if not table_buf and para_buf:
                flush_para()
            table_buf.append(line); i += 1; continue
        if table_buf:
            if line.strip():
                table_buf.append(line); i += 1; continue
            flush_table(); i += 1; continue

        if line.strip():
            para_buf.append(line.strip())
        else:
            flush_para()
        i += 1

    flush_para()
    if table_buf:
        flush_table()
    if in_code:
        elements.append(Element(ElementType.CODE, "\n".join(code_buf), heading_path=" > ".join(heading_path)))
    return elements


# ============ 第2步：结构感知切块 ============
def structure_aware_chunk(elements, max_chars=300):
    chunks = []
    current, current_path = "", ""

    for el in elements:
        # 标题 = 强制开新块，并更新当前标题路径
        if el.type in (ElementType.TITLE, ElementType.HEADING):
            if current.strip():
                chunks.append({"content": current.strip(), "heading": current_path})
                current = ""
            current_path = el.heading_path
            current = el.text + "\n"
            continue

        # 表格 / 代码 = 原子单元，单独成块，禁止从中间切开
        if el.type in (ElementType.TABLE, ElementType.CODE):
            if current.strip():
                chunks.append({"content": current.strip(), "heading": current_path})
                current = ""
            chunks.append({"content": f"[{el.type}] {el.text}", "heading": current_path})
            continue

        # 段落 = 累积到 max_chars 才开新块
        if len(current) + len(el.text) > max_chars:
            chunks.append({"content": current.strip(), "heading": current_path})
            current = ""
        current += el.text + "\n"

    if current.strip():
        chunks.append({"content": current.strip(), "heading": current_path})
    return chunks


if __name__ == "__main__":
    doc = """# 采购合同

## 一、合同双方
本合同由甲方与乙方签订。
甲方：上海XX科技有限公司。

## 二、价格与付款方式
双方约定按以下节点付款：

| 阶段 | 金额（元） | 付款时间 |
| --- | --- | --- |
| 预付款 | 30000 | 签订后3日 |
| 中期款 | 50000 | 交付验收后 |

## 三、质保条款
乙方提供12个月质保期。
质保期内免费修复故障。

## 四、验收代码
```python
def get_status(contract):
    return contract.status
```"""

    elements = parse_markdown(doc)

    print("=" * 60)
    print("第1步：解析出的元素树")
    print("=" * 60)
    for el in elements:
        print(f"[{el.type.value:9s} | {el.heading_path}] {el.text[:36]}...")

    print("\n" + "=" * 60)
    print("第2步：切分结果（每块带标题路径 = 上下文）")
    print("=" * 60)
    for i, c in enumerate(structure_aware_chunk(elements), 1):
        print(f"\n--- Chunk {i}（上下文: {c['heading']}） ---")
        print(c["content"])
