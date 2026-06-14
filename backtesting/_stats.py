# ============================================================
# backtesting/_stats.py -- 回测统计指标计算
# ============================================================
# 上下文层: Backtest.run() 的最后一步, 把交易列表+权益数组
#           变成约 30 项金融指标(收益, 风险, 回撤, 交易表现).
# 功能层: compute_stats() 是核心----计算收益率/年化/波动率/Sharpe/
#           Sortino/Calmar/Alpha/Beta/回撤/胜率/盈亏比/SQN 等.
# 设计层: 用 numpy 向量化计算, 结果存在 pd.Series 中, _Stats 子类定制打印.

from __future__ import annotations

from typing import TYPE_CHECKING, List, Union, cast

import numpy as np
import pandas as pd

from ._util import _data_period, _indicator_warmup_nbars

# TYPE_CHECKING 只对静态检查为 True----避免循环导入(backtesting.py 也 import 了这里)
if TYPE_CHECKING:
    from .backtesting import Strategy, Trade


def compute_drawdown_duration_peaks(dd: pd.Series):
    """给定回撤序列(0=新高), 返回每次回撤的持续时间和最大幅度."""
    iloc = np.unique(np.r_[(dd == 0).values.nonzero()[0], len(dd) - 1])
    iloc = pd.Series(iloc, index=dd.index[iloc])
    df = iloc.to_frame('iloc').assign(prev=iloc.shift())
    df = df[df['iloc'] > df['prev'] + 1].astype(np.int64)

    if not len(df):
        return (dd.replace(0, np.nan),) * 2

    df['duration'] = (df['iloc'].map(dd.index.__getitem__)
                      - df['prev'].map(dd.index.__getitem__))
    df['peak_dd'] = df.apply(
        lambda row: dd.iloc[row['prev']:row['iloc'] + 1].max(), axis=1)
    df = df.reindex(dd.index)
    return df['duration'], df['peak_dd']


def geometric_mean(returns: pd.Series) -> float:
    """几何平均收益率----用对数变换避免浮点累积误差."""
    returns = returns.fillna(0) + 1
    if np.any(returns <= 0):
        return 0
    return np.exp(np.log(returns).sum() / (len(returns) or np.nan)) - 1


def compute_stats(
        trades: Union[List['Trade'], pd.DataFrame],
        equity: np.ndarray,
        ohlc_data: pd.DataFrame,
        strategy_instance: Strategy | None,
        risk_free_rate: float = 0,
) -> pd.Series:
    """核心统计函数----输入交易+权益, 输出 30+ 项指标."""
    assert -1 < risk_free_rate < 1

    index = ohlc_data.index

    # ---- 回撤 ----
    dd = 1 - equity / np.maximum.accumulate(equity)
    dd_dur, dd_peaks = compute_drawdown_duration_peaks(pd.Series(dd, index=index))

    equity_df = pd.DataFrame({
        'Equity': equity,
        'DrawdownPct': dd,
        'DrawdownDuration': dd_dur},
        index=index)

    # ---- 交易数据 ----
    if isinstance(trades, pd.DataFrame):
        trades_df: pd.DataFrame = trades
        commissions = None
    else:
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

        # 每笔交易进出场时的指标值
        if len(trades_df) and strategy_instance:
            for ind in strategy_instance._indicators:
                ind = np.atleast_2d(ind)
                for i, values in enumerate(ind):
                    suffix = f'_{i}' if len(ind) > 1 else ''
                    trades_df[f'Entry_{ind.name}{suffix}'] = \
                        values[trades_df['EntryBar'].values]
                    trades_df[f'Exit_{ind.name}{suffix}'] = \
                        values[trades_df['ExitBar'].values]

        commissions = sum(t._commissions for t in trades)
    del trades

    pl = trades_df['PnL']
    returns = trades_df['ReturnPct']
    durations = trades_df['Duration']

    def _round_timedelta(value, _period=_data_period(index)):
        if not isinstance(value, pd.Timedelta):
            return value
        resolution = getattr(_period, 'resolution_string', None) or _period.resolution
        return value.ceil(resolution)

    s = pd.Series(dtype=object)

    # ====== 基础 ======
    s.loc['Start'] = index[0]
    s.loc['End'] = index[-1]
    s.loc['Duration'] = s.End - s.Start

    # ====== 持仓暴露 ======
    have_position = np.repeat(0, len(index))
    for t in trades_df[['EntryBar', 'ExitBar']].itertuples(index=False):
        have_position[t.EntryBar:t.ExitBar + 1] = 1
    s.loc['Exposure Time [%]'] = have_position.mean() * 100

    # ====== 收益 ======
    s.loc['Equity Final [$]'] = equity[-1]
    s.loc['Equity Peak [$]'] = equity.max()
    if commissions:
        s.loc['Commissions [$]'] = commissions
    s.loc['Return [%]'] = (equity[-1] - equity[0]) / equity[0] * 100
    first_trading_bar = _indicator_warmup_nbars(strategy_instance)
    c = ohlc_data.Close.values
    s.loc['Buy & Hold Return [%]'] = \
        (c[-1] - c[first_trading_bar]) / c[first_trading_bar] * 100

    # ====== 年化 ======
    gmean_day_return: float = 0
    day_returns = np.array(np.nan)
    annual_trading_days = np.nan
    is_datetime_index = isinstance(index, pd.DatetimeIndex)
    if is_datetime_index:
        freq_days = cast(pd.Timedelta, _data_period(index)).days
        have_weekends = (
            index.dayofweek.to_series().between(5, 6).mean() > 2 / 7 * .6)
        annual_trading_days = (
            52 if freq_days == 7 else
            12 if freq_days == 31 else
            1 if freq_days == 365 else
            (365 if have_weekends else 252))       # 加密货币 365, 股票 252
        freq = {7: 'W', 31: 'ME', 365: 'YE'}.get(freq_days, 'D')
        day_returns = equity_df['Equity'].resample(
            freq).last().dropna().pct_change()
        gmean_day_return = geometric_mean(day_returns)

    # 年化收益率: 几何平均日收益按复利换算到年
    annualized_return = (1 + gmean_day_return)**annual_trading_days - 1
    s.loc['Return (Ann.) [%]'] = annualized_return * 100
    # 年化波动率: 日收益的方差按年化系数放大(复利假设下的波动率公式)
    s.loc['Volatility (Ann.) [%]'] = np.sqrt(
        (day_returns.var(ddof=int(bool(day_returns.shape)))
         + (1 + gmean_day_return)**2)**annual_trading_days
        - (1 + gmean_day_return)**(2 * annual_trading_days)) * 100

    if is_datetime_index:
        time_in_years = (s.loc['Duration'].days
                         + s.loc['Duration'].seconds / 86400) / annual_trading_days
        s.loc['CAGR [%]'] = (
            (s.loc['Equity Final [$]'] / equity[0])**(1 / time_in_years) - 1
        ) * 100 if time_in_years else np.nan

    # ====== 风险调整收益 ======
    # Sharpe: 年化超额收益 / 年化波动率(衡量每单位总风险的超额回报)
    s.loc['Sharpe Ratio'] = (s.loc['Return (Ann.) [%]'] - risk_free_rate * 100) / (
        s.loc['Volatility (Ann.) [%]'] or np.nan)
    # Sortino: 类似 Sharpe 但分母只用下行波动率(只惩罚亏损侧的波动)
    with np.errstate(divide='ignore'):
        s.loc['Sortino Ratio'] = (annualized_return - risk_free_rate) / (
            np.sqrt(np.mean(day_returns.clip(-np.inf, 0)**2))
            * np.sqrt(annual_trading_days))
    max_dd = -np.nan_to_num(dd.max())
    s.loc['Calmar Ratio'] = annualized_return / (-max_dd or np.nan)

    # ====== Alpha / Beta(CAPM) ======
    equity_log_returns = np.log(equity[1:] / equity[:-1])
    market_log_returns = np.log(c[1:] / c[:-1])
    beta = np.nan
    if len(equity_log_returns) > 1 and len(market_log_returns) > 1:
        cov_matrix = np.cov(equity_log_returns, market_log_returns)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    s.loc['Alpha [%]'] = (
        s.loc['Return [%]'] - risk_free_rate * 100
        - beta * (s.loc['Buy & Hold Return [%]'] - risk_free_rate * 100))
    s.loc['Beta'] = beta

    # ====== 回撤 ======
    s.loc['Max. Drawdown [%]'] = max_dd * 100
    s.loc['Avg. Drawdown [%]'] = -dd_peaks.mean() * 100
    s.loc['Max. Drawdown Duration'] = _round_timedelta(dd_dur.max())
    s.loc['Avg. Drawdown Duration'] = _round_timedelta(dd_dur.mean())

    # ====== 交易表现 ======
    s.loc['# Trades'] = n_trades = len(trades_df)
    win_rate = np.nan if not n_trades else (pl > 0).mean()
    s.loc['Win Rate [%]'] = win_rate * 100
    s.loc['Best Trade [%]'] = returns.max() * 100
    s.loc['Worst Trade [%]'] = returns.min() * 100
    mean_return = geometric_mean(returns)
    s.loc['Avg. Trade [%]'] = mean_return * 100
    s.loc['Max. Trade Duration'] = _round_timedelta(durations.max())
    s.loc['Avg. Trade Duration'] = _round_timedelta(durations.mean())
    s.loc['Profit Factor'] = returns[returns > 0].sum() / (
        abs(returns[returns < 0].sum()) or np.nan)
    s.loc['Expectancy [%]'] = returns.mean() * 100
    s.loc['SQN'] = np.sqrt(n_trades) * pl.mean() / (pl.std() or np.nan)
    s.loc['Kelly Criterion'] = win_rate - (1 - win_rate) / (
        pl[pl > 0].mean() / -pl[pl < 0].mean())

    # 内部字段(_ 前缀, print 时隐藏)
    s.loc['_strategy'] = strategy_instance
    s.loc['_equity_curve'] = equity_df
    s.loc['_trades'] = trades_df

    s = _Stats(s)
    return s


class _Stats(pd.Series):
    """pd.Series 子类----只定制打印格式(字段缩略, 小数位)."""
    def __repr__(self):
        with pd.option_context(
            'display.max_colwidth', 20,
            'display.max_rows', len(self),
            'display.precision', 5,
        ):
            return super().__repr__()


def dummy_stats():
    """跑一个伪回测获取 stats 字段名列表----optimize 之前用来知道有哪些列."""
    from .backtesting import Trade, _Broker
    index = pd.DatetimeIndex(['2025'])
    data = pd.DataFrame({col: [np.nan] for col in ('Close',)}, index=index)
    trade = Trade(_Broker(
        data=data, cash=10000, spread=.01, commission=.01, margin=.1,
        trade_on_close=True, hedging=True, exclusive_orders=False,
        index=index), 1, 1, 0, None)
    trade._replace(exit_price=1, exit_bar=0)
    trade._commissions = np.nan
    return compute_stats([trade], np.r_[[np.nan]], data, None, 0)
