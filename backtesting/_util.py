# ============================================================
# backtesting/_util.py — 内部工具函数和数据结构
# ============================================================
# 上下文层：基础设施模块，被 backtesting.py、lib.py、_stats.py、
#           _plotting.py 等几乎所有模块依赖。用户一般不直接碰。
# 功能层：提供 try_、patch 上下文管理器、_as_str 格式化、_batch 分批、
#          核心数据结构 _Array（ndarray 子类）、_Data（OHLCV 访问器）、
#          SharedMemoryManager（多进程共享内存）。
# 设计层：_Array 继承 np.ndarray 并重写 __new__/__array_finalize__——
#          Python 高级特性：NumPy 子类化。_Data 用 __getattr__ 做属性代理。

from __future__ import annotations

import os
import sys
import warnings
from contextlib import contextmanager
from functools import partial
from itertools import chain
from multiprocessing import resource_tracker as _mprt
from multiprocessing import shared_memory as _mpshm
from numbers import Number
from threading import Lock
from typing import Dict, List, Optional, Sequence, Union, cast

import numpy as np
import pandas as pd

# tqdm 是可选依赖：装了就走进度条，没装就原样透传
try:
    from tqdm.auto import tqdm as _tqdm
    _tqdm = partial(_tqdm, leave=False)   # 进度条完成后自动消失
except ImportError:
    def _tqdm(seq, **_):
        return seq


def try_(lazy_func, default=None, exception=Exception):
    """执行 lazy 函数，抛指定异常时返回 default。lambda 是为了延迟求值。"""
    try:
        return lazy_func()
    except exception:
        return default


@contextmanager
def patch(obj, attr, newvalue):
    """临时设置 obj.attr = newvalue，退出 with 块自动恢复。
    用在 SharedMemory 里临时禁用资源追踪注册。"""
    had_attr = hasattr(obj, attr)
    orig_value = getattr(obj, attr, None)
    setattr(obj, attr, newvalue)
    try:
        yield
    finally:
        if had_attr:
            setattr(obj, attr, orig_value)
        else:
            delattr(obj, attr)


def _as_str(value) -> str:
    """各种类型 → 短字符串，给图表图例和调试信息用。"""
    if isinstance(value, (Number, str)):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return 'df'
    name = str(getattr(value, 'name', '') or '')
    if name in ('Open', 'High', 'Low', 'Close', 'Volume'):
        return name[:1]   # OHLCV → 首字母
    if callable(value):
        name = getattr(value, '__name__', value.__class__.__name__).replace('<lambda>', 'λ')
    if len(name) > 10:
        name = name[:9] + '…'
    return name


def _as_list(value) -> List:
    """值 → 列表（已是非字符串序列则直接转 list）。"""
    if isinstance(value, Sequence) and not isinstance(value, str):
        return list(value)
    return [value]


def _batch(seq):
    """按 CPU 核心数分块，用于多进程负载均衡。"""
    n = np.clip(int(len(seq) // (os.cpu_count() or 1)), 1, 300)
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _data_period(index) -> Union[pd.Timedelta, Number]:
    """用最后 100 个时间戳的间隔中位数推断数据频率。"""
    values = pd.Series(index[-100:])
    return values.diff().dropna().median()


def _strategy_indicators(strategy):
    """遍历策略 __dict__，筛出 _Indicator 类型的属性。"""
    return {attr: indicator
            for attr, indicator in strategy.__dict__.items()
            if isinstance(indicator, _Indicator)}.items()


def _indicator_warmup_nbars(strategy):
    """所有指标中最长的 NaN 前缀长度——回测要从这之后才开始。"""
    if strategy is None:
        return 0
    nbars = max((np.isnan(indicator.astype(float)).argmin(axis=-1).max()
                 for _, indicator in _strategy_indicators(strategy)
                 if not indicator._opts['scatter']), default=0)
    return nbars


# ═══════════════════════════════════════════════════════════
# _Array — 扩展的 NumPy ndarray
# ═══════════════════════════════════════════════════════════
# 上下文层：框架里流转的核心数据结构。带 .name 和 ._opts 元数据，
#           既保持 ndarray 高性能，又能被框架追踪命名和绘图参数。
# 设计层：继承 np.ndarray，重写 __new__（ndarray 是不可变对象）、
#          __array_finalize__（切片/运算后属性传播）、
#          __reduce__/__setstate__（pickle 序列化支持多进程）。
class _Array(np.ndarray):
    """ndarray extended to supply .name and other arbitrary properties."""

    def __new__(cls, array, *, name=None, **kwargs):
        obj = np.asarray(array).view(cls)   # .view() 把已有数组转成 _Array
        obj.name = name or array.name
        obj._opts = kwargs                  # 绘图元数据（scatter/color 等）
        return obj

    def __array_finalize__(self, obj):
        # ndarray 切片/运算后 NumPy 自动调这里，把 name 和 _opts "传播"到新对象
        if obj is not None:
            self.name = getattr(obj, 'name', '')
            self._opts = getattr(obj, '_opts', {})

    def __reduce__(self):
        # pickle 序列化——多进程优化时需要把指标传给子进程
        value = super().__reduce__()
        return value[:2] + (value[2] + (self.__dict__,),)

    def __setstate__(self, state):
        self.__dict__.update(state[-1])
        super().__setstate__(state[:-1])

    def __bool__(self):
        """取最后一个元素的布尔值，让 `if self.ma1:` 能直接用。"""
        try:
            return bool(self[-1])
        except IndexError:
            return super().__bool__()

    def __float__(self):
        try:
            return float(self[-1])
        except IndexError:
            return super().__float__()

    def to_series(self):
        warnings.warn("`.to_series()` is deprecated. Use `.s` accessor instead.")
        return self.s

    @property
    def s(self) -> pd.Series:
        """转成 pandas Series（带时间索引）。"""
        values = np.atleast_2d(self)
        index = self._opts['index'][:values.shape[1]]
        return pd.Series(values[0], index=index, name=self.name)

    @property
    def df(self) -> pd.DataFrame:
        """转成 pandas DataFrame。"""
        values = np.atleast_2d(np.asarray(self))
        index = self._opts['index'][:values.shape[1]]
        df = pd.DataFrame(values.T, index=index, columns=[self.name] * len(values))
        return df


# _Indicator 是 _Array 的标记子类——只用于 isinstance 区分"指标"和"普通数组"
class _Indicator(_Array):
    pass


# ═══════════════════════════════════════════════════════════
# _Data — OHLCV 数据访问器
# ═══════════════════════════════════════════════════════════
# 上下文层：包裹 pd.DataFrame，但返回的是 _Array（ndarray）而非 Series，
#           避免回测主循环中过 pandas 的性能开销。
# 设计层：代理模式——.Close/.Open 等属性通过 __getattr__ 动态转发到列访问。
class _Data:
    """Provides access to OHLCV columns as ndarray for performance."""

    def __init__(self, df: pd.DataFrame):
        self.__df = df
        self.__len = len(df)                     # 当前可见长度（回测中逐步增加）
        self.__pip: Optional[float] = None       # pip 值，惰性计算
        self.__cache: Dict[str, _Array] = {}     # 当前长度的切片缓存
        self.__arrays: Dict[str, _Array] = {}    # 完整数组映射
        self._update()

    def __getitem__(self, item):
        return self.__get_array(item)            # data['Close']

    def __getattr__(self, item):
        # data.Close —— 属性回退到列访问
        try:
            return self.__get_array(item)
        except KeyError:
            raise AttributeError(f"Column '{item}' not in data") from None

    def _set_length(self, length):
        """回测每推进一根 K 线后调用。"""
        self.__len = length
        self.__cache.clear()

    def _update(self):
        index = self.__df.index.copy()
        self.__arrays = {col: _Array(arr, index=index)
                         for col, arr in self.__df.items()}
        self.__arrays['__index'] = index

    def __repr__(self):
        i = min(self.__len, len(self.__df)) - 1
        index = self.__arrays['__index'][i]
        items = ', '.join(f'{k}={v}' for k, v in self.__df.iloc[i].items())
        return f'<Data i={i} ({index}) {items}>'

    def __len__(self):
        return self.__len

    @property
    def df(self) -> pd.DataFrame:
        return (self.__df.iloc[:self.__len]
                if self.__len < len(self.__df)
                else self.__df)

    @property
    def pip(self) -> float:
        """动态推断最小价格变动单位（分析 Close 小数位数）。"""
        if self.__pip is None:
            self.__pip = float(
                10**-np.median([len(s.partition('.')[-1])
                               for s in self.__arrays['Close'].astype(str)]))
        return self.__pip

    def __get_array(self, key) -> _Array:
        arr = self.__cache.get(key)
        if arr is None:
            arr = self.__cache[key] = cast(_Array, self.__arrays[key][:self.__len])
        return arr

    @property
    def Open(self) -> _Array:
        return self.__get_array('Open')

    @property
    def High(self) -> _Array:
        return self.__get_array('High')

    @property
    def Low(self) -> _Array:
        return self.__get_array('Low')

    @property
    def Close(self) -> _Array:
        return self.__get_array('Close')

    @property
    def Volume(self) -> _Array:
        return self.__get_array('Volume')

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.__get_array('__index')

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state


# SharedMemory 版本兼容：Python 3.13 API 有变化
if sys.version_info >= (3, 13):
    SharedMemory = _mpshm.SharedMemory
else:
    class SharedMemory(_mpshm.SharedMemory):
        __lock = Lock()

        def __init__(self, *args, track: bool = True, **kwargs):
            self._track = track
            if track:
                return super().__init__(*args, **kwargs)
            with self.__lock:
                with patch(_mprt, 'register', lambda *a, **kw: None):
                    super().__init__(*args, **kwargs)

        def unlink(self):
            if _mpshm._USE_POSIX and self._name:
                _mpshm._posixshmem.shm_unlink(self._name)
                if self._track:
                    _mprt.unregister(self._name, "shared_memory")


# ═══════════════════════════════════════════════════════════
# SharedMemoryManager — 共享内存管理器
# ═══════════════════════════════════════════════════════════
# 上下文层：多进程优化时，把 DataFrame 写入共享内存传给子进程，
#           比 pickle 传大数据高效得多。
# 设计层：实现 __enter__/__exit__，用上下文管理器协议保证资源释放。
class SharedMemoryManager:
    def __init__(self, create=False) -> None:
        self._shms: list[SharedMemory] = []
        self.__create = create

    def SharedMemory(self, *, name=None, create=False, size=0, track=True):
        shm = SharedMemory(name=name, create=create, size=size, track=track)
        shm._create = create
        self._shms.append(shm)   # 保持引用防 Windows GC
        return shm

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        for shm in self._shms:
            try:
                shm.close()
                if shm._create:
                    shm.unlink()
            except Exception:
                warnings.warn(f'Failed to unlink shared memory {shm.name!r}',
                              category=ResourceWarning, stacklevel=2)
                raise

    def arr2shm(self, vals):
        """一维数组 → 共享内存，返回 (name, shape, dtype)。"""
        assert vals.ndim == 1, (vals.ndim, vals.shape, vals)
        shm = self.SharedMemory(size=vals.nbytes, create=True)
        buf = np.ndarray(vals.shape, dtype=vals.dtype.base, buffer=shm.buf)
        has_tz = getattr(vals.dtype, 'tz', None)
        buf[:] = vals.tz_localize(None) if has_tz else vals
        return shm.name, vals.shape, vals.dtype

    def df2shm(self, df):
        """整个 DataFrame → 共享内存。"""
        return tuple((
            (column, *self.arr2shm(values))
            for column, values in chain([(self._DF_INDEX_COL, df.index)], df.items())
        ))

    @staticmethod
    def shm2s(shm, shape, dtype) -> pd.Series:
        arr = np.ndarray(shape, dtype=dtype.base, buffer=shm.buf)
        arr.setflags(write=False)   # 只读，防跨进程误改
        return pd.Series(arr, dtype=dtype)

    _DF_INDEX_COL = '__bt_index'

    @staticmethod
    def shm2df(data_shm):
        """共享内存 → DataFrame。"""
        shm = [SharedMemory(name=name, create=False, track=False)
               for _, name, _, _ in data_shm]
        df = pd.DataFrame({
            col: SharedMemoryManager.shm2s(shm, shape, dtype)
            for shm, (col, _, shape, dtype) in zip(shm, data_shm)})
        df.set_index(SharedMemoryManager._DF_INDEX_COL, drop=True, inplace=True)
        df.index.name = None
        return df, shm
