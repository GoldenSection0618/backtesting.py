# ============================================================
# backtesting/test/__init__.py — 测试数据和辅助工具模块
# ============================================================
# 上下文层：本模块为测试和示例提供：
#           1. 真实金融历史数据（BTCUSD、GOOG、EURUSD）
#           2. 技术指标辅助函数（SMA）
#           3. 数据加载工具（_read_file）
#           这些数据同时被 run_demo.py 和测试套件使用。
# 设计层：CSV 数据文件与源码一起打包（include_package_data=True），
#          通过 __file__ 相对路径定位，保证任何环境下都能加载。
# ============================================================

# 功能层：启用"延迟注解求值"（PEP 563），类型注解以字符串形式存储
from __future__ import annotations

import pandas as pd


# --- 内部数据加载函数 ---
# 功能层：读取与本模块同目录下的 CSV 文件，返回 pandas DataFrame
# 设计层：使用 os.path.dirname(__file__) 获取本文件所在目录的绝对路径，
#          确保无论从哪个工作目录运行都能找到数据文件。
#          index_col=0 将第一列设为行索引（日期），
#          parse_dates=True 自动将日期字符串解析为 DatetimeIndex。
def _read_file(filename):
    from os.path import dirname, join

    return pd.read_csv(join(dirname(__file__), filename),
                       index_col=0, parse_dates=True)


# --- 示例行情数据 ---
# 上下文层：这些数据是 Backtesting.py 官方提供的示例数据，
#           用于演示回测功能、运行测试用例和教学示例。
#           数据涵盖三种典型金融品种：股票、加密货币、外汇。

# 功能层：BTC/USD 月度价格数据（2012-2024，12年）
# 设计层：模块级常量，大写命名遵循 PEP 8 常量命名规范
BTCUSD = _read_file('BTCUSD.csv')
"""DataFrame of monthly BTC/USD histrical index data from 2012 through 2024 (12 years)."""

# 功能层：GOOG (Google/Alphabet) 日线价格数据（2004-2013）
# 上下文层：GOOG 是 run_demo.py 和其他示例主要使用的数据集
GOOG = _read_file('GOOG.csv')
"""DataFrame of daily NASDAQ:GOOG (Google/Alphabet) stock price data from 2004 to 2013."""

# 功能层：EUR/USD 小时线外汇数据（2017.04-2018.02）
EURUSD = _read_file('EURUSD.csv')
"""DataFrame of hourly EUR/USD forex data from April 2017 to February 2018."""


# --- 技术指标辅助函数 ---
# 功能层：计算序列 arr 的 n 周期简单移动平均线（SMA）
# 设计层：SMA 是最基础的趋势指标，用于 run_demo.py 中的 SmaCross 策略。
#          使用 pandas 的 rolling().mean() 方法，利用向量化计算而非 Python 循环。
# 上下文层：这个函数被 run_demo.py 中的 SmaCross 策略通过 Strategy.I() 注册使用。
def SMA(arr: pd.Series, n: int) -> pd.Series:
    """
    Returns `n`-period simple moving average of array `arr`.
    """
    return pd.Series(arr).rolling(n).mean()
