# -*- coding: utf-8 -*-
"""
Mock CRM系统
============

模拟CRM接口，使用JSON文件存储数据
"""

import json
import os
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MockCRM:
    """Mock CRM系统"""

    def __init__(self, data_dir: str = "data/crm"):
        """
        初始化

        Args:
            data_dir: 数据目录
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.users_file = os.path.join(data_dir, "users.json")
        self.tickets_file = os.path.join(data_dir, "tickets.json")

        # 初始化数据
        self._init_data()

        logger.info("MockCRM初始化完成")

    def _init_data(self):
        """初始化数据文件"""
        # 初始化用户数据
        if not os.path.exists(self.users_file):
            default_users = {
                "user_001": {
                    "user_id": "user_001",
                    "name": "张三",
                    "email": "zhangsan@example.com",
                    "phone": "13800138000",
                    "vip_level": "gold",
                    "register_date": "2023-01-15",
                    "total_orders": 25,
                    "total_amount": 15800.50
                },
                "user_002": {
                    "user_id": "user_002",
                    "name": "李四",
                    "email": "lisi@example.com",
                    "phone": "13900139000",
                    "vip_level": "silver",
                    "register_date": "2023-06-20",
                    "total_orders": 10,
                    "total_amount": 5200.00
                }
            }
            self._save_json(self.users_file, default_users)

        # 初始化工单数据
        if not os.path.exists(self.tickets_file):
            default_tickets = {
                "T00001": {
                    "ticket_id": "T00001",
                    "user_id": "user_001",
                    "issue": "订单未发货",
                    "priority": "high",
                    "status": "processing",
                    "create_time": "2024-03-01 10:30:00",
                    "update_time": "2024-03-01 14:20:00",
                    "description": "订单号12345已支付3天，仍未发货"
                }
            }
            self._save_json(self.tickets_file, default_tickets)

    def _load_json(self, file_path: str) -> Dict:
        """加载JSON文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载文件失败: {file_path}, {e}")
            return {}

    def _save_json(self, file_path: str, data: Dict):
        """保存JSON文件"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存文件失败: {file_path}, {e}")

    def get_user_info(self, user_id: str) -> Optional[Dict]:
        """
        查询用户信息

        Args:
            user_id: 用户ID

        Returns:
            用户信息
        """
        users = self._load_json(self.users_file)
        return users.get(user_id)

    def create_ticket(
        self,
        user_id: Optional[str],
        issue: str,
        priority: str = "normal",
        description: str = ""
    ) -> Dict:
        """
        创建工单

        Args:
            user_id: 用户ID
            issue: 问题描述
            priority: 优先级（low/normal/high/urgent）
            description: 详细描述

        Returns:
            工单信息
        """
        tickets = self._load_json(self.tickets_file)

        # 生成工单ID
        ticket_count = len(tickets) + 1
        ticket_id = f"T{ticket_count:05d}"

        # 创建工单
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ticket = {
            "ticket_id": ticket_id,
            "user_id": user_id or "anonymous",
            "issue": issue,
            "priority": priority,
            "status": "open",
            "create_time": now,
            "update_time": now,
            "description": description
        }

        tickets[ticket_id] = ticket
        self._save_json(self.tickets_file, tickets)

        logger.info(f"创建工单: {ticket_id}")
        return ticket

    def get_ticket(self, ticket_id: str) -> Optional[Dict]:
        """
        查询工单

        Args:
            ticket_id: 工单ID

        Returns:
            工单信息
        """
        tickets = self._load_json(self.tickets_file)
        return tickets.get(ticket_id)

    def update_ticket_status(self, ticket_id: str, status: str) -> bool:
        """
        更新工单状态

        Args:
            ticket_id: 工单ID
            status: 状态（open/processing/resolved/closed）

        Returns:
            是否成功
        """
        tickets = self._load_json(self.tickets_file)

        if ticket_id not in tickets:
            return False

        tickets[ticket_id]["status"] = status
        tickets[ticket_id]["update_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self._save_json(self.tickets_file, tickets)
        logger.info(f"更新工单状态: {ticket_id} -> {status}")
        return True

    def get_user_tickets(self, user_id: str) -> List[Dict]:
        """
        查询用户的所有工单

        Args:
            user_id: 用户ID

        Returns:
            工单列表
        """
        tickets = self._load_json(self.tickets_file)
        user_tickets = [
            ticket for ticket in tickets.values()
            if ticket["user_id"] == user_id
        ]
        return user_tickets
