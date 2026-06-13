# ============================================================
# backtesting/_stats.py — 回测统计指标计算模块
# ============================================================
# 上下文层：本模块是回测结果的"计算引擎"。
#           在 Backtest.run() 执行完毕后，compute_stats() 负责
#           将所有原始交易数据转换为约 30 项有意义的金融指标。
#           这些指标涵盖了收益、风险、交易表现三大维度。
# 设计层：使用 pandas Series 和 numpy 向量化计算，
#          所有指标通过 FinancialMath/empyrical 兼容的计算方法得出。
# ============================================================

from __future__ import annotations

from typing import TYPE_CHECKING, List, Union, cast

import numpy as np
import pandas as pd

from ._util import _data_period, _indicator_warmup_nbars

# 功能层：TYPE_CHECKING 仅在静态类型检查时为 True，
#          避免循环导入（backtesting.py 也 import 了 _stats）。
if TYPE_CHECKING:
    from .backtesting import Strategy, Trade


# --- compute_drawdown_duration_peaks：回撤持续期和峰值分析 ---
# 功能层：给定回撤序列 dd（0 = 创新高, >0 = 回撤中），
#          计算每次回撤的持续时间和最大回撤幅度。
# 设计层：使用 numpy 向量化操作定位回撤区间边界，
#          然后用 pandas apply 计算区间内的统计值。
def compute_drawdown_duration_peaks(dd: pd.Series):
    # 功能层：找到回撤为 0 的位置（创新高点），加上最后一个索引
    #         np.r_ 是 numpy 的数组拼接工具，等价于 hstack
    iloc = np.unique(np.r_[(dd == 0).values.nonzero()[0], len(dd) - 1])
    # 功能层：构建 DataFrame，每行表示一个回撤区间的 [起点, 终点]
    iloc = pd.Series(iloc, index=dd.index[iloc])
    df = iloc.to_frame('iloc').assign(prev=iloc.shift())
    # 功能层：过滤掉相邻的零回撤点（区间长度为 0 或 1）
    df = df[df['iloc'] > df['prev'] + 1].astype(np.int64)

    # 功能层：如果没有回撤（无交易），返回 NaN 序列
    if not len(df):
        return (dd.replace(0, np.nan),) * 2

    # 功能层：计算每个回撤区间的持续时间和最大回撤幅度
    df['duration'] = df['iloc'].map(dd.index.__getitem__) - df['prev'].map(dd.index.__getitem__)
    df['peak_dd'] = df.apply(lambda row: dd.iloc[row['prev']:row['iloc'] + 1].max(), axis=1)
    # 功能层：reindex 扩展回原索引长度，缺失位置为 NaN
    df = df.reindex(dd.index)
    return df['duration'], df['peak_dd']


# --- geometric_mean：几何平均数 ---
# 功能层：计算周期收益率的几何平均值。
# 设计层：几何平均值适用于描述复合收益（各期收益连乘后开方），
#          比算术平均更能反映真实收益水平。
#          使用对数变换避免浮点乘积累积误差。
def geometric_mean(returns: pd.Series) -> float:
    returns = returns.fillna(0) + 1  # 将净收益率转为总收益率（1 + r）
    if np.any(returns <= 0):
        return 0  # 任意期收益 ≤ -100% 则整体归零
    return np.exp(np.log(returns).sum() / (len(returns) or np.nan)) - 1


# --- compute_stats：核心统计计算函数 ---
# 上下文层：这是回测运行的最后一步、也是关键一步。
#           Backtest.run() 调用本函数生成最终统计结果。
#           输入的 trades 可以是 Trade 对象列表或 DataFrame。
#           输出的 Series 被 _Stats 子类包装以优化打印显示。
def compute_stats(
        trades: Union[List['Trade'], pd.DataFrame],
        equity: np.ndarray,
        ohlc_data: pd.DataFrame,
        strategy_instance: Strategy | None,
        risk_free_rate: float = 0,
) -> pd.Series:
    # 功能层：无风险利率必须在 (-1, 1) 范围内
    assert -1 < risk_free_rate < 1

    index = ohlc_data.index

    # --- 回撤计算 ---
    # 功能层：dd = 1 - equity / running_max(equity)
    #          np.maximum.accumulate 计算到当前为止的历史最高权益
    dd = 1 - equity / np.maximum.accumulate(equity)
    dd_dur, dd_peaks = compute_drawdown_duration_peaks(pd.Series(dd, index=index))

    # 功能层：构建权益曲线 DataFrame
    equity_df = pd.DataFrame({
        'Equity': equity,
        'DrawdownPct': dd,
        'DrawdownDuration': dd_dur},
        index=index)

    # --- 交易数据处理 ---
    if isinstance(trades, pd.DataFrame):
        # 上下文层：trades 已经是 DataFrame，直接使用（来自 _optimize_sambo）
        trades_df: pd.DataFrame = trades
        commissions = None  # 不显示佣金总额
    else:
        # 上下文层：trades 是 Trade 对象列表（正常 Backtest.run() 的输出）
        # 功能层：将 Trade 对象列表转换为规范化的 DataFrame
        trades_df = pd.DataFrame({
            'Size': [t.size for t in trades],
            'EntryBar': [t.entry_bar for t in trades],
            'ExitBar': [t.exit_bar for t in trades],
            'EntryPrice': [t.entry_price for t in trades],
            'ExitPrice': [t.exit_price for t in trades],
            'SL': [t.sl for t in trades],
            'TP': [t.tp for t in trades],
            'PnL': [t.pl for t in trades],
            'Commission': [t._commissions for t in trades],
            'ReturnPct': [t.pl_pct for t in trades],
            'EntryTime': [t.entry_time for t in trades],
            'ExitTime': [t.exit_time for t in trades],
        })
        trades_df['Duration'] = trades_df['ExitTime'] - trades_df['EntryTime']
        trades_df['Tag'] = [t.tag for t in trades]

        # 功能层：记录每笔交易进出场时的指标值（用于事后分析）
        if len(trades_df) and strategy_instance:
            for ind in strategy_instance._indicators:
                ind = np.atleast_2d(ind)
                for i, values in enumerate(ind):  # 处理多维指标
                    suffix = f'_{i}' if len(ind) > 1 else ''
                    trades_df[f'Entry_{ind.name}{suffix}'] = values[trades_df['EntryBar'].values]
                    trades_df[f'Exit_{ind.name}{suffix}'] = values[trades_df['ExitBar'].values]

        commissions = sum(t._commissions for t in trades)
    del trades  # 功能层：显式删除，释放内存

    # --- 提取关键列 ---
    pl = trades_df['PnL']
    returns = trades_df['ReturnPct']
    durations = trades_df['Duration']

    # --- 时间差舍入辅助函数 ---
    # 功能层：将 Timedelta 向上舍入到数据周期的精度
    def _round_timedelta(value, _period=_data_period(index)):
        if not isinstance(value, pd.Timedelta):
            return value
        resolution = getattr(_period, 'resolution_string', None) or _period.resolution
        return value.ceil(resolution)

    # --- 构建统计结果 Series ---
    # 设计层：使用 pd.Series(dtype=object) 然后逐个赋值，
    #          以支持混合类型（数字、字符串、Timedelta、DataFrame）。
    s = pd.Series(dtype=object)

    # ====== 基础信息 ======
    s.loc['Start'] = index[0]
    s.loc['End'] = index[-1]
    s.loc['Duration'] = s.End - s.Start

    # ====== 持仓暴露时间 ======
    # 功能层：通过遍历所有交易的进出场区间，标记每个 K 线是否有持仓
    have_position = np.repeat(0, len(index))
    for t in trades_df[['EntryBar', 'ExitBar']].itertuples(index=False):
        have_position[t.EntryBar:t.ExitBar + 1] = 1
    s.loc['Exposure Time [%]'] = have_position.mean() * 100  # 以 K 线时间为单位

    # ====== 权益和收益率 ======
    s.loc['Equity Final [$]'] = equity[-1]
    s.loc['Equity Peak [$]'] = equity.max()
    if commissions:
        s.loc['Commissions [$]'] = commissions
    s.loc['Return [%]'] = (equity[-1] - equity[0]) / equity[0] * 100

    # 功能层：Buy & Hold 收益率——作为策略表现的基准线
    first_trading_bar = _indicator_warmup_nbars(strategy_instance)
    c = ohlc_data.Close.values
    s.loc['Buy & Hold Return [%]'] = (c[-1] - c[first_trading_bar]) / c[first_trading_bar] * 100

    # ====== 年化指标计算 ======
    # 上下文层：年化指标需要识别数据的交易频率和交易天数。
    #           不同频率（日/周/月/年）采用不同的年化因子。
    gmean_day_return: float = 0
    day_returns = np.array(np.nan)
    annual_trading_days = np.nan
    is_datetime_index = isinstance(index, pd.DatetimeIndex)
    if is_datetime_index:
        freq_days = cast(pd.Timedelta, _data_period(index)).days
        # 功能层：通过周末占比判断是否包含周末交易日
        have_weekends = index.dayofweek.to_series().between(5, 6).mean() > 2 / 7 * .6
        # 功能层：根据数据频率确定年化交易天数
        annual_trading_days = (
            52 if freq_days == 7 else       # 周线：52 周
            12 if freq_days == 31 else       # 月线：12 月
            1 if freq_days == 365 else       # 年线：1 年
            (365 if have_weekends else 252))  # 日线：加密货币 365, 股票 252
        # 功能层：按周期重采样权益曲线，计算周期收益率
        freq = {7: 'W', 31: 'ME', 365: 'YE'}.get(freq_days, 'D')
        day_returns = equity_df['Equity'].resample(freq).last().dropna().pct_change()
        gmean_day_return = geometric_mean(day_returns)

    # 设计层：年化收益率和波动率使用"几何布朗运动"假设（Brownian motion compounding）。
    #          参考：https://dx.doi.org/10.2139/ssrn.3054517
    annualized_return = (1 + gmean_day_return)**annual_trading_days - 1
    s.loc['Return (Ann.) [%]'] = annualized_return * 100

    # 功能层：年化波动率——使用复合收益方差公式
    s.loc['Volatility (Ann.) [%]'] = np.sqrt((day_returns.var(ddof=int(bool(day_returns.shape))) + (1 + gmean_day_return)**2)**annual_trading_days - (1 + gmean_day_return)**(2 * annual_trading_days)) * 100  # noqa: E501

    # ====== CAGR（复合年增长率） ======
    if is_datetime_index:
        time_in_years = (s.loc['Duration'].days + s.loc['Duration'].seconds / 86400) / annual_trading_days
        s.loc['CAGR [%]'] = ((s.loc['Equity Final [$]'] / equity[0])**(1 / time_in_years) - 1) * 100 if time_in_years else np.nan  # noqa: E501

    # ====== 风险调整收益 ======
    # 功能层：Sharpe Ratio = 年化超额收益 / 年化波动率
    s.loc['Sharpe Ratio'] = (s.loc['Return (Ann.) [%]'] - risk_free_rate * 100) / (s.loc['Volatility (Ann.) [%]'] or np.nan)  # noqa: E501

    # 功能层：Sortino Ratio（仅考虑下行波动率，比 Sharpe 更合理）
    with np.errstate(divide='ignore'):  # 抑制除零警告
        s.loc['Sortino Ratio'] = (annualized_return - risk_free_rate) / (np.sqrt(np.mean(day_returns.clip(-np.inf, 0)**2)) * np.sqrt(annual_trading_days))  # noqa: E501

    # 功能层：Calmar Ratio = 年化收益率 / 最大回撤
    max_dd = -np.nan_to_num(dd.max())
    s.loc['Calmar Ratio'] = annualized_return / (-max_dd or np.nan)

    # ====== Alpha 和 Beta（CAPM 模型） ======
    # 功能层：Beta：策略收益对市场收益的敏感度
    equity_log_returns = np.log(equity[1:] / equity[:-1])
    market_log_returns = np.log(c[1:] / c[:-1])
    beta = np.nan
    if len(equity_log_returns) > 1 and len(market_log_returns) > 1:
        cov_matrix = np.cov(equity_log_returns, market_log_returns)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    # 功能层：Jensen's Alpha：策略超出 CAPM 预期收益的超额部分
    s.loc['Alpha [%]'] = s.loc['Return [%]'] - risk_free_rate * 100 - beta * (s.loc['Buy & Hold Return [%]'] - risk_free_rate * 100)  # noqa: E501
    s.loc['Beta'] = beta

    # ====== 回撤指标 ======
    s.loc['Max. Drawdown [%]'] = max_dd * 100
    s.loc['Avg. Drawdown [%]'] = -dd_peaks.mean() * 100
    s.loc['Max. Drawdown Duration'] = _round_timedelta(dd_dur.max())
    s.loc['Avg. Drawdown Duration'] = _round_timedelta(dd_dur.mean())

    # ====== 交易表现统计 ======
    s.loc['# Trades'] = n_trades = len(trades_df)
    # 功能层：胜率 = 盈利交易占比
    win_rate = np.nan if not n_trades else (pl > 0).mean()
    s.loc['Win Rate [%]'] = win_rate * 100
    s.loc['Best Trade [%]'] = returns.max() * 100
    s.loc['Worst Trade [%]'] = returns.min() * 100

    # 功能层：平均每笔交易的几何平均收益率
    mean_return = geometric_mean(returns)
    s.loc['Avg. Trade [%]'] = mean_return * 100
    s.loc['Max. Trade Duration'] = _round_timedelta(durations.max())
    s.loc['Avg. Trade Duration'] = _round_timedelta(durations.mean())

    # 功能层：Profit Factor = 总盈利 / 总亏损（>1 表示策略有盈利能力）
    s.loc['Profit Factor'] = returns[returns > 0].sum() / (abs(returns[returns < 0].sum()) or np.nan)  # noqa: E501

    # 功能层：数学期望（每笔交易的平均收益百分比）
    s.loc['Expectancy [%]'] = returns.mean() * 100

    # 功能层：SQN (System Quality Number)——综合衡量策略质量的指标
    s.loc['SQN'] = np.sqrt(n_trades) * pl.mean() / (pl.std() or np.nan)

    # 功能层：Kelly Criterion——最优仓位比例（基于胜率和胜负比）
    s.loc['Kelly Criterion'] = win_rate - (1 - win_rate) / (pl[pl > 0].mean() / -pl[pl < 0].mean())

    # --- 附加数据 ---
    # 功能层：_strategy/_equity_curve/_trades 以 _ 前缀标记为内部字段
    s.loc['_strategy'] = strategy_instance
    s.loc['_equity_curve'] = equity_df
    s.loc['_trades'] = trades_df

    # 功能层：包装为 _Stats 子类以优化打印格式
    s = _Stats(s)
    return s


# --- _Stats：统计结果 Series（打印格式化子类） ---
# 设计层：仅重写 __repr__ 方法以自定义打印格式。
#          Python 高级特性：通过子类化 pandas Series 来定制显示行为。
class _Stats(pd.Series):
    def __repr__(self):
        # 功能层：使用 pandas 的 option_context 临时调整显示选项
        with pd.option_context(
            'display.max_colwidth', 20,    # 限制 _equity_curve 和 _trades 列宽
            'display.max_rows', len(self),  # 显示全部行（默认会截断）
            'display.precision', 5,         # 保留 5 位小数
        ):
            return super().__repr__()


# --- dummy_stats：伪统计（用于预先获取统计字段名） ---
# 上下文层：在 Backtest.optimize() 之前调用，
#          用于获取 stats 对象的字段名列表（用于构建优化结果的 DataFrame 列）。
# 功能层：创建一个虚拟 Trade 并运行 compute_stats，返回字段名结构。
def dummy_stats():
    from .backtesting import Trade, _Broker
    index = pd.DatetimeIndex(['2025'])
    data = pd.DataFrame({col: [np.nan] for col in ('Close',)}, index=index)
    trade = Trade(_Broker(data=data, cash=10000, spread=.01, commission=.01, margin=.1,
                          trade_on_close=True, hedging=True, exclusive_orders=False, index=index),
                  1, 1, 0, None)
    trade._replace(exit_price=1, exit_bar=0)
    trade._commissions = np.nan
    return compute_stats([trade], np.r_[[np.nan]], data, None, 0)
