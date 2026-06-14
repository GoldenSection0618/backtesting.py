# ============================================================
# backtesting/test/__init__.py — 测试数据和工具
# ============================================================
# 上下文层：提供三份示例行情数据和 SMA 函数，被 run_demo.py 和测试套件共用。
# 功能层：_read_file 用 __file__ 相对路径加载 CSV，保证任何目录运行都能找到。
# 设计层：数据以模块级常量暴露（BTCUSD/GOOG/EURUSD），大写遵循 PEP 8。

from __future__ import annotations

import pandas as pd


def _read_file(filename):
    """读取同目录下的 CSV，返回日期索引的 DataFrame。"""
    from os.path import dirname, join
    return pd.read_csv(join(dirname(__file__), filename),
                       index_col=0, parse_dates=True)


# 三份示例数据，覆盖股票/加密货币/外汇
BTCUSD = _read_file('BTCUSD.csv')
"""BTC/USD 月线（2012–2024）"""

GOOG = _read_file('GOOG.csv')
"""NASDAQ:GOOG 日线（2004–2013）—— run_demo.py 用的就是这份"""

EURUSD = _read_file('EURUSD.csv')
"""EUR/USD 小时线（2017.04–2018.02）"""


def SMA(arr: pd.Series, n: int) -> pd.Series:
    """n 周期简单移动平均，用 pandas rolling().mean() 向量化实现。"""
    return pd.Series(arr).rolling(n).mean()
