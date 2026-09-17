# -*- coding: utf-8 -*-
"""
知识图谱 · 建库与种子
====================

建出 `database/knowledge_graph.db`，把「商品 → 类别 → 政策适用性」这条链显式存下来。

**为什么需要它：** 判断「这个商品能不能退」现在是靠模型读政策文档、自己把商品和政策
例外条款语义接上。答得对，但那是"理解到位"不是"算出来的"，而且**讲不出规则依据**。
有了图谱，判定变成沿关系推导，每一步都能对用户讲清楚。

**本脚本只铺骨架**，不负责抽取：
- 建表
- 种下**类别层级**（无线耳机 → 3C数码产品 → 全部商品）
- 种下**政策节点**
- 种下**条件词表**（封闭枚举，LLM 之后只能从里面选）
- 商品名 → 类别的**锚点映射**

「类别 → 政策」的适用/排除关系由 `extract_graph_relations.py` 从政策文档抽取，
再经人工核对后才生效（`status='verified'`）。

## 三个关键设计（改代码前先读）

### 1. 条件必须外置成封闭枚举

「已激活或已拆封」不是一段自由文本，而是一个 `code`（`ACTIVATED`）加一组关键词
（`["已激活","已拆封","拆开","开过机"]`）。这样条件词汇是**可校验的枚举**，
LLM 只能从里面选、不能自己发明新条件。

### 2. `condition_code IS NULL` 和 `unknown` 是两件完全不同的事

    NULL      = 无条件生效（「生鲜类不支持换货」，天生如此）
    unknown   = 有条件，但当前不确定满足没满足

**把 NULL 误当成 unknown，会把"本该被排除的商品"判成"可退"** —— 错误方向最危险。

### 3. 需要一个根类别「全部商品」

否则挂在通用类别上的政策（如「质量问题退货 15 日」）就查不到。
专属政策挂具体类别（3C数码 → 排除七天无理由），通用政策挂根类别。

## 用法

    .venv/Scripts/python.exe tools/scripts/init_knowledge_graph.py
"""

import os
import sys
import json
import sqlite3

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DB_PATH = os.path.join(PROJECT_ROOT, 'database', 'knowledge_graph.db')
ECOMMERCE_DB = os.path.join(PROJECT_ROOT, 'database', 'ecommerce.db')

if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


# ============================================================
# 建表
# ============================================================

SCHEMA_SQL = """
-- 节点：类别 或 政策
CREATE TABLE entities (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    entity_type TEXT NOT NULL CHECK(entity_type IN ('category', 'policy')),
    description TEXT,                       -- 给 LLM 看的说明（仅类别用得上）
    source_doc_id TEXT,                     -- 政策节点记它出自哪篇文档
    -- 政策族：同一族里"更具体的类别覆盖更泛的"。
    -- 例如「质量问题退货（15日）」和「家电质量问题退货（7日）」同属「质量问题退货」，
    -- 家电查询时只保留 7日 那条，否则会同时出现两条、看起来像"都适用"。
    -- 留空表示这条政策不参与覆盖（如「七天无理由退货」本身就是唯一的）。
    family      TEXT
);

-- 条件词表（封闭枚举）。关系只引用 code，不存自由文本。
CREATE TABLE condition_atoms (
    code       TEXT PRIMARY KEY,
    label      TEXT NOT NULL,               -- 人话标签，如「已激活或已拆封」
    patterns   TEXT NOT NULL,               -- JSON 数组，关键词
    match_mode TEXT NOT NULL DEFAULT 'keyword'
);

-- 边
CREATE TABLE relations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id    INTEGER NOT NULL REFERENCES entities(id),
    predicate     TEXT NOT NULL CHECK(predicate IN ('IS_A', 'APPLIES_TO', 'EXCLUDES')),
    object_id     INTEGER NOT NULL REFERENCES entities(id),
    condition_code TEXT REFERENCES condition_atoms(code),   -- NULL = 无条件生效，见文件头说明
    -- 状态机：proposed → verified / rejected / superseded
    -- rejected 的记录**不删除** —— 下次重抽必然再吐出同一条幻觉，
    -- 留着才能自动打回，否则人工要重复审同一句话
    status        TEXT NOT NULL DEFAULT 'proposed'
                  CHECK(status IN ('proposed', 'verified', 'rejected', 'superseded')),
    source_doc_id TEXT,                     -- 血缘：出自哪篇文档
    source_quote  TEXT,                     -- 血缘：原文精确摘录（必须是原文子串）
    source_offset INTEGER,                  -- 血缘：在文档中的位置，用于校验
    batch_id      TEXT,
    reviewed_by   TEXT,
    reviewed_at   TEXT,
    note          TEXT                      -- 校验/审核备注
);

-- 商品名 → 类别 的锚点。
-- 注意：product_name 是营销名（「云听 Pro 主动降噪无线耳机」），
-- 跟文档里的类别词（「无线耳机」）对不上，所以必须有一层显式映射。
CREATE TABLE product_categories (
    pattern     TEXT PRIMARY KEY,
    category_id INTEGER NOT NULL REFERENCES entities(id)
);

-- 只有"已核实"的关系才参与查询。
-- 把它做成视图而不是让应用层自己写 WHERE —— 应用层不可能写漏。
CREATE VIEW v_rel AS
    SELECT * FROM relations WHERE status = 'verified';
"""


# ============================================================
# 种子数据
# ============================================================

# 条件词表（封闭枚举）。patterns 要覆盖用户真实会说的说法。
CONDITION_ATOMS = [
    ("ACTIVATED", "已激活或已拆封",
     ["已激活", "已拆封", "拆开", "拆过", "开过机", "开封", "用过了", "使用过"]),
    ("QUALITY_ISSUE", "存在质量问题",
     ["质量问题", "坏了", "破损", "故障", "不能用", "没声音", "漏发", "少发", "发错"]),
    ("DAMAGED_BY_USER", "人为损坏",
     ["人为损坏", "摔了", "进水", "自己拆", "私自改装", "非官方维修"]),
    ("CUSTOM_MADE", "定制或定作",
     ["定制", "定作", "刻字", "订做"]),
    ("PERISHABLE", "鲜活易腐",
     ["生鲜", "水果", "冷冻", "鲜花", "绿植"]),
    ("OVERSIZE", "大件或超规格",
     ["大件", "超规格", "超重"]),
    ("BULK_PROMO", "大促期间下单",
     ["大促", "618", "双11", "双12", "年货节"]),
    ("THIRD_PARTY", "第三方商家商品",
     ["第三方", "非自营", "旗舰店"]),
]

# 类别层级：(类别名, 父类别名, 说明)。根类别是「全部商品」。
# 分层是有意的 —— 专属政策挂具体类别，通用政策挂根类别，递归向上找能同时拿到两者。
CATEGORIES = [
    ("全部商品", None, "根类别。通用政策挂在这里，所有商品都适用"),
    ("一般商品", "全部商品", "没有特殊类别的普通商品"),
    ("3C数码产品", "全部商品", "手机、平板电脑、无线耳机等数码产品"),
    ("无线耳机", "3C数码产品", "蓝牙/降噪耳机等"),
    ("大件商品", "全部商品", "体积大或需特殊物流的商品"),
    ("家电", "大件商品", "空调、净化器、冰箱等"),
    ("家具", "大件商品", "沙发、床、定制尺寸家具等"),
    ("生鲜类商品", "全部商品", "果蔬、冷冻食品、鲜花绿植"),
    ("贴身用品", "全部商品", "内衣、泳衣、袜子等"),
    ("内衣", "贴身用品", ""),
    ("泳衣", "贴身用品", ""),
    # 美妆个护归在贴身用品下 —— 政策原文的例外清单就是这么列的：
    # 「已拆封的贴身用品，如内衣、泳衣、袜子、美妆个护类商品」
    ("美妆个护", "贴身用品", "护肤品、化妆品、洗护"),
    ("定制商品", "全部商品", "刻字饰品、定制印花服饰、订做尺寸"),
    ("数字化商品", "全部商品", "在线下载的软件、音像制品"),
    ("食品保健品药品", "全部商品", "入口类商品"),
    ("预售商品", "全部商品", "按详情页标注时间发货"),
    ("进口保税商品", "全部商品", "跨境/保税仓发货"),
    ("虚拟商品", "全部商品", "充值类、话费流量类"),
]

# 政策节点。这些是**规则**不是文档 —— 一篇文档里可能有多条规则。
# 格式：(政策名, 出处文档, 政策族)。族留空 = 不参与"更具体覆盖更泛"。
POLICIES = [
    ("七天无理由退货", "seven_day_return_policy", None),
    ("换货申请", "return_exchange_process", None),

    # 「质量问题退货/换货」这组是有时限差异的：
    #   一般商品（含 3C数码等大多数类别）走 15日/30日
    #   家电走 7日/15日
    # 时限编码在名字里，靠 family 做覆盖。最初词表里没有家电那两个节点，
    # 导致抽取时 LLM 只能从「15日/30日」里挑 —— 错不在模型，在节点设计。
    ("质量问题退货（15日）", "quality_issue_policy", "质量问题退货"),
    ("质量问题换货（30日）", "quality_issue_policy", "质量问题换货"),
    ("家电质量问题退货（7日）", "quality_issue_policy", "质量问题退货"),
    ("家电质量问题换货（15日）", "quality_issue_policy", "质量问题换货"),

    ("质量问题退货运费由平台承担", "shipping_fee_policy", None),
    ("无理由退货运费由消费者承担", "shipping_fee_policy", None),
    ("大件商品运费报销上限100元", "shipping_fee_policy", None),
    ("会员免费退货次数", "member_benefits_policy", None),
    ("会员退货运费补贴", "member_benefits_policy", None),
    ("48小时发货", "late_shipment_policy", None),
    ("超时未发货赔付", "late_shipment_policy", None),
    ("退款到账时效", "refund_timeline", None),
    ("生鲜质量问题凭照片直接退款", "quality_issue_policy", None),
]

# 商品名锚点：演示库里 order_items 的真实商品名 → 类别
PRODUCT_CATEGORIES = [
    ("云听 Pro 主动降噪无线耳机", "无线耳机"),
    ("云影 15 轻薄笔记本电脑", "3C数码产品"),
    ("云净空气净化器 3 代", "家电"),
    ("纯棉圆领短袖 T 恤（男）", "一般商品"),
    ("透气运动短裤", "一般商品"),
    ("特级龙井茶叶礼盒 250g", "食品保健品药品"),
    ("氨基酸温和洁面乳", "美妆个护"),
    ("不锈钢保温杯 500ml", "一般商品"),
    ("羊毛混纺针织开衫", "一般商品"),
]


# ============================================================
# 执行
# ============================================================

def get_or_create_entity(conn, cache: dict, name: str, etype: str,
                         description: str = None, doc_id: str = None,
                         family: str = None) -> int:
    """按名字取实体 id，没有就建。cache 避免重复查库。"""
    if name in cache:
        return cache[name]

    row = conn.execute("SELECT id FROM entities WHERE name = ?", (name,)).fetchone()
    if row:
        cache[name] = row[0]
        return row[0]

    cur = conn.execute(
        "INSERT INTO entities (name, entity_type, description, source_doc_id, family) "
        "VALUES (?,?,?,?,?)",
        (name, etype, description, doc_id, family),
    )
    cache[name] = cur.lastrowid
    return cur.lastrowid


def build(conn: sqlite3.Connection) -> dict:
    """建表 + 灌种子。返回统计。"""
    # 先 DROP 再建，保证脚本可反复跑（幂等）
    conn.executescript("""
        DROP VIEW  IF EXISTS v_rel;
        DROP TABLE IF EXISTS product_categories;
        DROP TABLE IF EXISTS relations;
        DROP TABLE IF EXISTS condition_atoms;
        DROP TABLE IF EXISTS entities;
    """)
    conn.executescript(SCHEMA_SQL)

    # ---- 条件原子 ----
    for code, label, patterns in CONDITION_ATOMS:
        conn.execute(
            "INSERT INTO condition_atoms (code, label, patterns) VALUES (?,?,?)",
            (code, label, json.dumps(patterns, ensure_ascii=False)),
        )

    # ---- 实体 ----
    cache: dict = {}
    # 类别（先建父级：列表本身就是从根往下排的）
    for name, parent, desc in CATEGORIES:
        get_or_create_entity(conn, cache, name, "category", desc)
    # 政策
    for name, doc_id, family in POLICIES:
        get_or_create_entity(conn, cache, name, "policy", None, doc_id, family)

    # ---- IS_A 层级（唯一的真相源，不额外存 parent_id，否则两份真相会漂移）----
    is_a_count = 0
    for name, parent, _ in CATEGORIES:
        if parent is None:
            continue
        conn.execute(
            "INSERT INTO relations (subject_id, predicate, object_id, condition_code, status, note) "
            "VALUES (?, 'IS_A', ?, NULL, 'verified', '类别层级，建库时手工声明')",
            (cache[name], cache[parent]),
        )
        is_a_count += 1

    # ---- 商品名锚点 ----
    for pattern, cat_name in PRODUCT_CATEGORIES:
        conn.execute(
            "INSERT INTO product_categories (pattern, category_id) VALUES (?,?)",
            (pattern, cache[cat_name]),
        )

    conn.commit()
    return {
        "entities": len(CATEGORIES) + len(POLICIES),
        "categories": len(CATEGORIES),
        "policies": len(POLICIES),
        "condition_atoms": len(CONDITION_ATOMS),
        "is_a": is_a_count,
        "product_anchors": len(PRODUCT_CATEGORIES),
    }


def check_product_coverage(conn: sqlite3.Connection) -> list:
    """
    检查演示库里的每个商品是否都有类别锚点。

    **为什么要有这一步：** 商品名到类别的锚点是这套设计里最脆弱的一环 ——
    新增商品忘了加映射，查询就会落到"没有类别"，从而查不到任何政策。
    这里主动把缺口打出来，而不是等它静默出错。
    """
    if not os.path.exists(ECOMMERCE_DB):
        return []

    patterns = [r[0] for r in conn.execute("SELECT pattern FROM product_categories")]
    econn = sqlite3.connect(ECOMMERCE_DB)
    try:
        products = sorted({r[0] for r in econn.execute("SELECT DISTINCT product_name FROM order_items")})
    finally:
        econn.close()

    missing = [p for p in products if not any(pat in p or p in pat for pat in patterns)]
    return missing


def main() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    try:
        stats = build(conn)
        print(f"\n知识图谱已生成: {DB_PATH}\n")
        print(f"  类别节点      {stats['categories']}")
        print(f"  政策节点      {stats['policies']}")
        print(f"  条件原子      {stats['condition_atoms']}")
        print(f"  IS_A 层级边   {stats['is_a']}")
        print(f"  商品锚点      {stats['product_anchors']}")

        missing = check_product_coverage(conn)
        if missing:
            print(f"\n  ⚠ 演示库里有 {len(missing)} 个商品没有类别锚点（查询会落空）:")
            for m in missing:
                print(f"      - {m}")
        else:
            print("\n  ✓ 演示库里所有商品都有类别锚点")

        pending = conn.execute(
            "SELECT COUNT(*) FROM relations WHERE status = 'proposed'"
        ).fetchone()[0]
        print(f"\n  待核对的抽取关系: {pending}")
        if pending == 0:
            print("    （下一步跑 extract_graph_relations.py 抽取「类别 → 政策」关系）")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
