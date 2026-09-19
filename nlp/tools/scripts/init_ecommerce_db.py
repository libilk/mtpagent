# -*- coding: utf-8 -*-
"""
电商售后演示数据库初始化脚本
==========================

用途
----
建出 `database/ecommerce.db`，为「电商售后助手」提供一套**可查询的订单/售后数据**。

它替代了原项目的 `database/chinook.db`（那是张音乐商店的示例库，卖歌不卖货，
跟售后场景完全对不上）。

表结构（6 张）
--------------
    customers     会员信息（含会员等级 —— 售后权益按等级不同）
    orders        订单主表（含收货信息、订单状态）
    order_items   订单明细（一个订单可能买多件商品）
    logistics     物流信息（承运商、运单号、当前状态、预计到达）
    refunds       退款/退货单（含申请原因、金额、处理状态）
    tickets       售后工单（客服 Agent 读写的地方）

设计说明
--------
- **破坏性 + 幂等（重跑结果一致）**：每次跑都先 `DROP TABLE` 清空这 6 张表再重建，
  **库里原有数据会全部丢失**；也正因为先删后建，跑第二遍的结果和第一遍完全相同，不会数据翻倍。
- **日期固定**：不用 `datetime.now()`，否则每次跑数据都变，没法复现和写测试。
  基准日定在 2026-09-16。
- **优先级枚举**：tickets.priority 用的是 low/normal/high/urgent，
  必须和 `agents/customer_service_agent/agent.py` 里 `create_ticket` 工具的枚举一致，
  否则 Agent 建工单时会参数校验失败。

用法
----
    python tools/scripts/init_ecommerce_db.py

或者从项目根目录：
    .venv/Scripts/python.exe tools/scripts/init_ecommerce_db.py
"""

import os
import sqlite3
import sys

# 定位项目根目录（本脚本在 tools/scripts/ 下，根目录在上两级）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DB_PATH = os.path.join(PROJECT_ROOT, 'database', 'ecommerce.db')

# Windows 控制台默认不是 UTF-8，中文会变成乱码，这里强制改掉
if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


# ============================================================
# 一、建表语句
# ============================================================
# DDL（建表语句）= 定义表结构的 SQL。
# 表用 primary key（主键，唯一标识一行）定位记录，用 foreign key（外键，指向另一张表的主键）
# 保证"引用的东西真实存在"；order_items.item_id 用 AUTOINCREMENT 自增整数，
# 因为这类明细行没有天然的唯一业务编号。
#
# ⚠ 一个真实踩过的坑：SQLite 会把建表语句的**原文（含 `--` 注释）**存进 sqlite_master。
# 读取端 core/sqlite_mcp_service.py 若按逗号裸切列名，切出的第一个词就是 `--`，
# list_tables 输出的列名会是错的。所以那支解析器必须先剥注释再解析。

SCHEMA_SQL = """
-- 会员表：售后权益按会员等级区分（银卡以上有免费退货运费补贴）
CREATE TABLE customers (
    user_id         TEXT PRIMARY KEY,      -- 会员ID，如 U10001
    name            TEXT NOT NULL,         -- 姓名
    phone           TEXT,                  -- 手机号
    email           TEXT,
    member_level    TEXT NOT NULL,         -- 普通会员 / 银卡 / 金卡 / 钻石
    register_date   TEXT,                  -- 注册日期 YYYY-MM-DD
    total_orders    INTEGER DEFAULT 0,     -- 历史订单数
    total_spent     REAL DEFAULT 0         -- 历史消费总额
);

-- 订单主表
CREATE TABLE orders (
    order_id        TEXT PRIMARY KEY,      -- 订单号，如 SO20260909001
    user_id         TEXT NOT NULL,
    order_date      TEXT NOT NULL,         -- 下单时间
    status          TEXT NOT NULL,         -- 待发货 / 已发货 / 运输中 / 已签收 / 已完成 / 已取消
    total_amount    REAL NOT NULL,         -- 订单实付金额
    payment_method  TEXT,                  -- 余额 / 支付宝 / 微信支付 / 银行卡 / 花呗
    receiver_name   TEXT,
    receiver_phone  TEXT,
    receiver_address TEXT,
    FOREIGN KEY (user_id) REFERENCES customers(user_id)
);

-- 订单明细：一个订单可以买多件商品
CREATE TABLE order_items (
    item_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id        TEXT NOT NULL,
    product_id      TEXT,                  -- 商品编码
    product_name    TEXT NOT NULL,         -- 商品名称（中文，方便 Agent 理解）
    category        TEXT,                  -- 数码 / 服饰 / 家居 / 食品 / 美妆
    quantity        INTEGER NOT NULL DEFAULT 1,
    unit_price      REAL NOT NULL,
    subtotal        REAL NOT NULL,         -- 小计 = quantity * unit_price
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

-- 物流表：售后助手回答「我的货到哪了」时查这里
CREATE TABLE logistics (
    logistics_id      TEXT PRIMARY KEY,
    order_id          TEXT NOT NULL,
    carrier           TEXT,                -- 顺丰速运 / 中通快递 / 云集自营
    tracking_no       TEXT,                -- 运单号
    status            TEXT,                -- 已揽收 / 运输中 / 派送中 / 已签收
    shipped_at        TEXT,                -- 发货时间
    estimated_arrival TEXT,                -- 预计到达
    delivered_at      TEXT,                -- 实际签收时间（未签收则为 NULL）
    current_location  TEXT,                -- 当前所在城市/网点
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

-- 退款/退货单
CREATE TABLE refunds (
    refund_id     TEXT PRIMARY KEY,        -- 如 RF20260912001
    order_id      TEXT NOT NULL,
    user_id       TEXT NOT NULL,
    refund_type   TEXT,                    -- 退货退款 / 仅退款 / 换货
    reason        TEXT,                    -- 申请原因
    amount        REAL,                    -- 退款金额
    status        TEXT,                    -- 待审核 / 审核通过 / 退款中 / 已到账 / 已拒绝
    applied_at    TEXT,                    -- 申请时间
    processed_at  TEXT,                    -- 处理完成时间
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (user_id) REFERENCES customers(user_id)
);

-- 售后工单：客服 Agent 读写的核心表
-- 注意 priority 的取值必须与 create_ticket 工具的枚举保持一致
CREATE TABLE tickets (
    ticket_id     TEXT PRIMARY KEY,        -- 如 TK20260916001
    -- user_id 允许为 NULL：用户没提供身份时代表"匿名咨询"。
    -- 但**只要给了值，就必须是 customers 表里真实存在的用户**（靠外键兜住），
    -- 这样模型臆造出来的 user_id 会被数据库拒绝，而不是悄悄存进去。
    user_id       TEXT,
    order_id      TEXT,                    -- 可能不关联具体订单（如纯咨询）
    category      TEXT,                    -- 退货 / 换货 / 物流 / 发票 / 投诉 / 咨询
    priority      TEXT NOT NULL DEFAULT 'normal',   -- low / normal / high / urgent
    sentiment     TEXT,                    -- positive / neutral / negative
    content       TEXT,                    -- 问题描述
    status        TEXT NOT NULL DEFAULT '待处理',    -- 待处理 / 处理中 / 已解决 / 已关闭
    created_at    TEXT,
    updated_at    TEXT,
    FOREIGN KEY (user_id) REFERENCES customers(user_id),
    -- order_id 也要约束：工单关联到一个不存在的订单没有意义，
    -- 而且模型臆造订单号是真实发生过的（实测填过 SO99999999）
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);
"""


# ============================================================
# 二、种子数据
# ============================================================
# 基准日：2026-09-16。日期都写成固定的字面量，保证每次跑结果一样。

CUSTOMERS = [
    # (user_id, name, phone, email, member_level, register_date, total_orders, total_spent)
    ("U10001", "张伟", "13800138001", "zhangwei@example.com", "金卡", "2023-04-12", 47, 18620.50),
    ("U10002", "李娜", "13800138002", "lina@example.com", "钻石", "2021-11-03", 132, 68940.00),
    ("U10003", "王强", "13800138003", "wangqiang@example.com", "普通会员", "2026-08-20", 2, 358.00),
    ("U10004", "刘敏", "13800138004", "liumin@example.com", "银卡", "2024-06-18", 18, 5420.30),
    ("U10005", "陈杰", "13800138005", "chenjie@example.com", "银卡", "2025-02-09", 11, 3180.80),
    ("U10006", "赵磊", "13800138006", "zhaolei@example.com", "普通会员", "2026-05-30", 5, 890.00),
]

ORDERS = [
    # (order_id, user_id, order_date, status, total_amount, payment_method, receiver_name, receiver_phone, receiver_address)
    # ↓ 主场景订单：张伟买的无线耳机，已签收 5 天 —— 用户会说「上周买的耳机坏了」
    ("SO20260909001", "U10001", "2026-09-09 14:23", "已签收", 599.00, "支付宝",
     "张伟", "13800138001", "浙江省杭州市西湖区文三路 100 号 3 幢 502 室"),

    ("SO20260912002", "U10002", "2026-09-12 09:41", "运输中", 2380.00, "花呗",
     "李娜", "13800138002", "广东省深圳市南山区科技园南路 8 号 A 座 1201"),

    ("SO20260901003", "U10003", "2026-09-01 20:15", "已完成", 358.00, "微信支付",
     "王强", "13800138003", "江苏省南京市鼓楼区中山北路 200 号 6 栋 302"),

    ("SO20260905004", "U10004", "2026-09-05 11:08", "已发货", 1299.00, "银行卡",
     "刘敏", "13800138004", "上海市浦东新区张江路 500 弄 12 号 801 室"),

    # ↓ 超时未发货场景
    ("SO20260910005", "U10005", "2026-09-10 16:52", "待发货", 268.00, "余额",
     "陈杰", "13800138005", "北京市朝阳区建国路 88 号 SOHO 现代城 1508"),

    ("SO20260820006", "U10006", "2026-08-20 10:30", "已完成", 890.00, "支付宝",
     "赵磊", "13800138006", "四川省成都市武侯区天府大道 666 号 2201"),

    ("SO20260914007", "U10001", "2026-09-14 08:12", "已签收", 79.00, "微信支付",
     "张伟", "13800138001", "浙江省杭州市西湖区文三路 100 号 3 幢 502 室"),

    ("SO20260908008", "U10002", "2026-09-08 19:37", "已完成", 1560.00, "支付宝",
     "李娜", "13800138002", "广东省深圳市南山区科技园南路 8 号 A 座 1201"),
]

ORDER_ITEMS = [
    # (order_id, product_id, product_name, category, quantity, unit_price, subtotal)
    # 主场景：无线耳机 —— 注意「已激活的 3C 数码」不适用七天无理由，
    # 只能走质量问题退货（15 日），这是知识库里特意写明的例外
    ("SO20260909001", "P-88231", "云听 Pro 主动降噪无线耳机", "数码", 1, 599.00, 599.00),

    ("SO20260912002", "P-90112", "云影 15 轻薄笔记本电脑", "数码", 1, 2380.00, 2380.00),

    ("SO20260901003", "P-55201", "纯棉圆领短袖 T 恤（男）", "服饰", 2, 129.00, 258.00),
    ("SO20260901003", "P-55210", "透气运动短裤", "服饰", 1, 100.00, 100.00),

    ("SO20260905004", "P-77340", "云净空气净化器 3 代", "家居", 1, 1299.00, 1299.00),

    ("SO20260910005", "P-33416", "特级龙井茶叶礼盒 250g", "食品", 1, 268.00, 268.00),

    ("SO20260820006", "P-66120", "氨基酸温和洁面乳", "美妆", 2, 445.00, 890.00),

    ("SO20260914007", "P-21008", "不锈钢保温杯 500ml", "家居", 1, 79.00, 79.00),

    ("SO20260908008", "P-45077", "羊毛混纺针织开衫", "服饰", 1, 1560.00, 1560.00),
]

LOGISTICS = [
    # (logistics_id, order_id, carrier, tracking_no, status, shipped_at, estimated_arrival, delivered_at, current_location)
    # 主场景订单：9/9 下单，9/10 发货，9/11 签收 —— 用户 9/16 来说「坏了」，已签收 5 天
    ("LG20260909001", "SO20260909001", "顺丰速运", "SF1234567890", "已签收",
     "2026-09-10 09:30", "2026-09-12", "2026-09-11 15:42", "杭州市西湖区文三路营业点"),

    ("LG20260912002", "SO20260912002", "中通快递", "ZT9876543210", "运输中",
     "2026-09-13 10:15", "2026-09-17", None, "武汉转运中心"),

    ("LG20260901003", "SO20260901003", "云集自营", "YJ2026090103", "已签收",
     "2026-09-02 08:20", "2026-09-04", "2026-09-03 11:05", "南京市鼓楼区营业点"),

    ("LG20260905004", "SO20260905004", "顺丰速运", "SF2233445566", "派送中",
     "2026-09-06 14:00", "2026-09-16", None, "上海市浦东新区张江营业点"),

    # 待发货订单没有物流记录 —— 这正是「超时未发货」场景要处理的：查不到物流
    # 所以这里故意不建 SO20260910005 的物流行。

    ("LG20260820006", "SO20260820006", "中通快递", "ZT5566778899", "已签收",
     "2026-08-21 09:00", "2026-08-24", "2026-08-23 16:20", "成都市武侯区营业点"),

    ("LG20260914007", "SO20260914007", "云集自营", "YJ2026091407", "已签收",
     "2026-09-14 15:40", "2026-09-16", "2026-09-15 10:18", "杭州市西湖区文三路营业点"),

    ("LG20260908008", "SO20260908008", "顺丰速运", "SF7788990011", "已签收",
     "2026-09-09 11:20", "2026-09-11", "2026-09-10 14:33", "深圳市南山区科技园营业点"),
]

REFUNDS = [
    # (refund_id, order_id, user_id, refund_type, reason, amount, status, applied_at, processed_at)
    ("RF20260912001", "SO20260820006", "U10006", "退货退款", "商品与描述不符，包装破损",
     445.00, "已到账", "2026-08-26 10:12", "2026-08-29 16:40"),

    ("RF20260915002", "SO20260908008", "U10002", "换货", "尺码偏小，申请换大一号",
     0.00, "审核通过", "2026-09-15 13:25", None),

    ("RF20260916003", "SO20260901003", "U10003", "仅退款", "少发一件运动短裤",
     100.00, "待审核", "2026-09-16 08:50", None),
]

TICKETS = [
    # (ticket_id, user_id, order_id, category, priority, sentiment, content, status, created_at, updated_at)
    ("TK20260914001", "U10004", "SO20260905004", "物流", "normal", "neutral",
     "订单显示已发货三天了，但物流信息一直停在发货地没有更新，麻烦帮忙查一下。",
     "处理中", "2026-09-14 09:20", "2026-09-15 10:00"),

    ("TK20260915002", "U10002", "SO20260912002", "咨询", "low", "positive",
     "想问一下笔记本电脑大概什么时候能到深圳？急着出差用。",
     "已解决", "2026-09-15 11:05", "2026-09-15 11:30"),

    ("TK20260916003", "U10006", None, "投诉", "high", "negative",
     "上次退的洁面乳退款只退了一件，另一件的钱一直没到账，已经催过两次了！",
     "待处理", "2026-09-16 07:45", "2026-09-16 07:45"),
]


# ============================================================
# 三、执行
# ============================================================

def create_tables(conn: sqlite3.Connection) -> None:
    """建表。先 DROP 再 CREATE，保证脚本可以反复跑。"""
    # 注意 DROP 顺序：有外键引用的表要先删
    # 本连接没开 PRAGMA foreign_keys（SQLite 配置指令，默认关），所以构建期外键并不校验；
    # 约束是运行时由 core/ecommerce_crm.py 的连接显式 `PRAGMA foreign_keys = ON` 后才生效。
    drop_order = ["tickets", "refunds", "logistics", "order_items", "orders", "customers"]
    for table in drop_order:
        conn.execute(f"DROP TABLE IF EXISTS {table}")

    conn.executescript(SCHEMA_SQL)
    conn.commit()


def insert_seed_data(conn: sqlite3.Connection) -> None:
    """灌入演示数据。用 executemany 批量插入，比逐条 execute 快也更简洁。"""

    def insert(table: str, columns: list, rows: list) -> None:
        """通用插入：根据列名拼 SQL，省去为每张表重复写一遍。"""
        placeholders = ", ".join("?" for _ in columns)
        sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
        conn.executemany(sql, rows)

    insert("customers",
           ["user_id", "name", "phone", "email", "member_level",
            "register_date", "total_orders", "total_spent"],
           CUSTOMERS)

    insert("orders",
           ["order_id", "user_id", "order_date", "status", "total_amount",
            "payment_method", "receiver_name", "receiver_phone", "receiver_address"],
           ORDERS)

    insert("order_items",
           ["order_id", "product_id", "product_name", "category",
            "quantity", "unit_price", "subtotal"],
           ORDER_ITEMS)

    insert("logistics",
           ["logistics_id", "order_id", "carrier", "tracking_no", "status",
            "shipped_at", "estimated_arrival", "delivered_at", "current_location"],
           LOGISTICS)

    insert("refunds",
           ["refund_id", "order_id", "user_id", "refund_type", "reason",
            "amount", "status", "applied_at", "processed_at"],
           REFUNDS)

    insert("tickets",
           ["ticket_id", "user_id", "order_id", "category", "priority",
            "sentiment", "content", "status", "created_at", "updated_at"],
           TICKETS)

    conn.commit()


def print_summary(conn: sqlite3.Connection) -> None:
    """打印每张表的行数，让人一眼确认建库成功了。"""
    print(f"\n数据库已生成: {DB_PATH}\n")
    print(f"{'表名':<14}{'行数':>6}")
    print("-" * 22)
    for table in ["customers", "orders", "order_items", "logistics", "refunds", "tickets"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table:<14}{count:>6}")

    # 顺手把主场景订单打出来，方便肉眼核对
    # JOIN（联表查询）：按外键把订单/会员/商品三表拼成一行 —— orders 里只有 user_id，看不出"谁买了什么"
    row = conn.execute(
        "SELECT o.order_id, c.name, c.member_level, o.status, i.product_name "
        "FROM orders o "
        "JOIN customers c ON o.user_id = c.user_id "
        "JOIN order_items i ON o.order_id = i.order_id "
        "WHERE o.order_id = 'SO20260909001'"
    ).fetchone()
    if row:
        print(f"\n主场景订单: {row[0]} | {row[1]}({row[2]}) | {row[3]} | {row[4]}")


def main() -> None:
    # 确保目录存在（database/ 在 .gitignore 里，clone 下来不会有）
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    try:
        create_tables(conn)
        insert_seed_data(conn)
        print_summary(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
