"""
FastMCP 服务端 —— 暴露工具 + 资源

运行方式：
  python learn/fast_mcp_server.py
"""

from mcp.server.fastmcp import FastMCP

# 创建 MCP 服务端
mcp = FastMCP("Demo")


# ---- 工具 ----
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def get_weather(city: str) -> str:
    """查询城市天气"""
    # 模拟数据
    weather_db = {
        "北京": "晴，25°C",
        "上海": "多云，28°C",
        "深圳": "阵雨，30°C",
    }
    return weather_db.get(city, f"未找到 {city} 的天气数据")


# ---- 资源 ----
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"


@mcp.resource("config://app")
def get_app_config() -> str:
    """返回应用配置"""
    return '{"version": "1.0", "debug": false, "max_tokens": 4096}'


if __name__ == "__main__":
    # stdio 模式启动
    mcp.run(transport="stdio")
