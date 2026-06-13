# ============================================================
# backtesting/_util.py — 内部工具函数与数据结构
# ============================================================
# 上下文层：本模块是回测框架的"基础设施层"。
#           它提供的工具函数和数据结构被 backtesting.py、lib.py、
#           _stats.py、_plotting.py 等几乎所有模块使用。
#           用户通常不直接接触本模块中的对象。
# ============================================================

# 功能层：启用"延迟注解求值"（PEP 563）
# 设计层：使得前向引用（如 Union、List 等）无需实际导入即可使用
from __future__ import annotations

# --- 标准库导入 ---
import os                     # CPU 核心数检测（用于批处理分块）
import sys                    # Python 版本检测（3.13 SharedMemory API 差异）
import warnings               # 发出警告（弃用、资源泄漏等）
from contextlib import contextmanager  # 上下文管理器装饰器（patch 函数）
from functools import partial           # 偏函数（tqdm 默认参数绑定）
from itertools import chain             # 可迭代对象扁平化（df2shm 数据序列化）
from multiprocessing import resource_tracker as _mprt  # 共享内存资源追踪
from multiprocessing import shared_memory as _mpshm    # Python 3.8+ 共享内存支持
from numbers import Number               # 数字类型检测（_as_str 格式化）
from threading import Lock               # 线程锁（SharedMemory 线程安全）
from typing import Dict, List, Optional, Sequence, Union, cast  # 类型注解

import numpy as np
import pandas as pd

# --- 进度条（可选依赖） ---
# 上下文层：tqdm 是可选依赖，安装后回测运行时自动显示进度条。
#           未安装时降级为无操作（透传迭代器）。
try:
    from tqdm.auto import tqdm as _tqdm
    # 功能层：partial 预设 leave=False，进度条完成后自动消失
    _tqdm = partial(_tqdm, leave=False)
except ImportError:
    # 功能层：无 tqdm 时返回原始迭代器，不影响回测逻辑
    def _tqdm(seq, **_):
        return seq


# --- try_：忽略特定异常的辅助函数 ---
# 功能层：执行 lazy_func 并返回结果；若抛出 exception 则返回 default。
# 设计层：使用惰性求值——通过 lambda 包装可能失败的代码，
#          避免在 try_ 调用前就触发异常。
#          这是一个 Python 中的常见小工具模式。
def try_(lazy_func, default=None, exception=Exception):
    try:
        return lazy_func()
    except exception:
        return default


# --- patch：临时修改对象属性的上下文管理器 ---
# 设计层：使用 @contextmanager 装饰器，将生成器函数转换为上下文管理器。
#          yield 之前的代码在 __enter__ 时执行，
#          finally 块中的代码在 __exit__ 时执行（无论是否异常）。
#          Python 高级特性：上下文管理器协议 + 生成器。
# 上下文层：在 SharedMemory 构造函数中用于临时禁用资源追踪器注册。
# 功能层：进入 with 块时设置 obj.attr = newvalue，
#          退出时恢复原值（或删除临时添加的属性）。
@contextmanager
def patch(obj, attr, newvalue):
    # 功能层：记录属性是否原本存在及其原始值
    had_attr = hasattr(obj, attr)
    orig_value = getattr(obj, attr, None)
    # 功能层：设置新值
    setattr(obj, attr, newvalue)
    try:
        yield  # 上下文管理器暂停点，用户代码在此时执行
    finally:
        # 功能层：退出时恢复原状态
        if had_attr:
            setattr(obj, attr, orig_value)
        else:
            delattr(obj, attr)


# --- _as_str：将对象转换为短字符串表示 ---
# 功能层：用于图表图例、调试信息和策略表示的文本格式化。
# 设计层：对不同类型采用不同策略——
#           数字/字符串直接输出，
#           DataFrame 压缩为 'df'，
#           OHLC 列名缩写为首字母（O/H/L/C/V），
#           可调用对象取其函数名等。
#          这是一种"人类友好"的摘要函数。
def _as_str(value) -> str:
    # 功能层：数字和字符串直接转换
    if isinstance(value, (Number, str)):
        return str(value)
    # 功能层：DataFrame 统一显示为 'df'（避免打印巨大表格）
    if isinstance(value, pd.DataFrame):
        return 'df'
    # 功能层：获取对象的 name 属性
    name = str(getattr(value, 'name', '') or '')
    # 功能层：常见的 OHLCV 列名缩写为首字母
    if name in ('Open', 'High', 'Low', 'Close', 'Volume'):
        return name[:1]
    # 功能层：可调用对象取其函数名，lambda 用 λ 表示
    if callable(value):
        name = getattr(value, '__name__', value.__class__.__name__).replace('<lambda>', 'λ')
    # 功能层：截断过长的名称（保留 9 字符 + 省略号）
    if len(name) > 10:
        name = name[:9] + '…'
    return name


# --- _as_list：将值包装为列表 ---
# 功能层：如果已是序列（非字符串）则转为列表；否则包装为单元素列表。
# 设计层：字符串虽然是 Sequence 但通常应作为原子值处理，所以显式排除。
def _as_list(value) -> List:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return list(value)
    return [value]


# --- _batch：将序列分块用于并行处理 ---
# 设计层：批大小根据 CPU 核心数动态计算，最小 1，最大 300。
#          这是一种简单的工作负载均衡策略：
#          每个 CPU 核心分配一个大块而非每元素一个任务。
# 功能层：将 seq 按批大小分成多个子序列。
def _batch(seq):
    # XXX:  标记了需要替换为 Python 3.12+ 的 itertools.batched
    # 功能层：批大小 = min(max(总长度/CPU核心数, 1), 300)
    n = np.clip(int(len(seq) // (os.cpu_count() or 1)), 1, 300)
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# --- _data_period：推断数据时间周期 ---
# 功能层：取 index 最后 100 个时间戳，计算相邻时间差的中位数。
# 上下文层：用于统计模块判断日线/周线/月线/年线，以正确年化指标。
def _data_period(index) -> Union[pd.Timedelta, Number]:
    """Return data index period as pd.Timedelta"""
    values = pd.Series(index[-100:])
    return values.diff().dropna().median()


# --- _strategy_indicators：提取策略中的所有指标 ---
# 功能层：遍历策略实例的 __dict__，筛选出 _Indicator 类型的属性。
# 设计层：通过 isinstance 检查实现类型过滤，
#          返回 (属性名, 指标数组) 的迭代器。
def _strategy_indicators(strategy):
    return {attr: indicator
            for attr, indicator in strategy.__dict__.items()
            if isinstance(indicator, _Indicator)}.items()


# --- _indicator_warmup_nbars：计算指标预热所需的最少 K 线数 ---
# 上下文层：在回测中，指标需要一定数量的历史数据才能计算出有效值
#          （例如 SMA(20) 需要 20 根 K 线才有第一个有效值）。
# 功能层：返回所有非散点指标中最长的 NaN 前缀长度。
def _indicator_warmup_nbars(strategy):
    if strategy is None:
        return 0
    nbars = max((np.isnan(indicator.astype(float)).argmin(axis=-1).max()
                 for _, indicator in _strategy_indicators(strategy)
                 if not indicator._opts['scatter']), default=0)
    return nbars


# ============================================================
# _Array —— 扩展的 NumPy 数组
# ============================================================
# 上下文层：_Array 是框架中最核心的内部数据结构之一。
#           它将 NumPy ndarray 扩展为携带 .name 和 ._opts 元数据的数组。
#           这使得指标值可以像 pandas Series 一样有名称和属性，
#           但保持 ndarray 的高性能（避免 pandas 在热路径上的开销）。
# 设计层：通过继承 np.ndarray 并重写 __new__（不可变对象的构造方法）、
#          __array_finalize__（数组运算后的属性传播）、
#          __reduce__/__setstate__（pickle 序列化）来实现。
#          Python 高级特性：NumPy 子类化。
class _Array(np.ndarray):
    """
    ndarray extended to supply .name and other arbitrary properties
    in ._opts dict.
    """
    # 设计层：__new__ 而非 __init__，因为 np.ndarray 是不可变对象，
    #          其构造逻辑必须在 __new__（对象分配阶段）完成。
    # 功能层：将任意 array-like 对象转换为 _Array 实例。
    def __new__(cls, array, *, name=None, **kwargs):
        # np.asarray 确保输入被转换为 ndarray（如果输入已经是 ndarray 则无拷贝）
        obj = np.asarray(array).view(cls)  # .view() 将已有的 ndarray 转换为 _Array 类型
        obj.name = name or array.name       # 继承原始 array 的 name 属性
        obj._opts = kwargs                  # 存储额外选项（如 scatter、color 等绘图参数）
        return obj

    # 设计层：__array_finalize__ 是 NumPy 子类化协议的关键方法。
    #          当 NumPy 从已有数组创建新数组时（如切片、reshape、ufunc 操作），
    #          这个方法被自动调用来"传播"自定义属性。
    # 功能层：从源对象 obj 复制 name 和 _opts 属性到当前对象。
    def __array_finalize__(self, obj):
        if obj is not None:
            self.name = getattr(obj, 'name', '')
            self._opts = getattr(obj, '_opts', {})

    # --- Pickle 序列化支持 ---
    # 上下文层：回测优化（Backtest.optimize）使用多进程，
    #          需要通过 pickle 将指标数据传递给子进程。
    # 功能层：重写 __reduce__ 在序列化元组末尾附加 __dict__（包含 name 和 _opts）。
    def __reduce__(self):
        value = super().__reduce__()
        return value[:2] + (value[2] + (self.__dict__,),)

    # 功能层：重写 __setstate__ 在反序列化时恢复 __dict__ 中的属性。
    def __setstate__(self, state):
        self.__dict__.update(state[-1])
        super().__setstate__(state[:-1])

    # --- 布尔/浮点数转换 ---
    # 功能层：返回最后一个元素的布尔值/浮点值。
    # 上下文层：这些方法允许策略中直接使用 `if self.ma1:` 这样的简洁语法。
    #          对于空数组，回退到父类默认行为。
    def __bool__(self):
        try:
            return bool(self[-1])
        except IndexError:
            return super().__bool__()

    def __float__(self):
        try:
            return float(self[-1])
        except IndexError:
            return super().__float__()

    # --- pandas 互操作 ---
    # 功能层：已弃用的 to_series() 方法，引导用户使用 .s 属性。
    def to_series(self):
        warnings.warn("`.to_series()` is deprecated. For pd.Series conversion, use accessor `.s`")
        return self.s

    # 设计层：@property 将方法伪装为属性访问，提供更自然的 API。
    # 功能层：将 _Array 转换为 pandas Series（取第一维数据，携带时间索引）。
    @property
    def s(self) -> pd.Series:
        values = np.atleast_2d(self)
        index = self._opts['index'][:values.shape[1]]
        return pd.Series(values[0], index=index, name=self.name)

    # 功能层：将 _Array 转换为 pandas DataFrame（多维指标时的表格视图）。
    @property
    def df(self) -> pd.DataFrame:
        values = np.atleast_2d(np.asarray(self))
        index = self._opts['index'][:values.shape[1]]
        df = pd.DataFrame(values.T, index=index, columns=[self.name] * len(values))
        return df


# --- _Indicator：指标数组的类型标记 ---
# 上下文层：_Indicator 是 _Array 的子类，纯粹用于类型区分。
#           策略中通过 Strategy.I() 注册的指标会被包装为 _Indicator 实例。
#           这允许 _strategy_indicators() 函数从策略属性中区分指标和普通变量。
# 设计层：空子类仅用于 isinstance 检查，不添加额外行为。
#          Python 高级特性：使用继承实现类型标记（标记子类模式）。
class _Indicator(_Array):
    pass


# ============================================================
# _Data —— OHLCV 数据访问器
# ============================================================
# 上下文层：_Data 包裹 pandas DataFrame，提供对 OHLCV 列的
#           高性能 ndarray 访问。它在回测主循环中被频繁访问，
#           因此缓存了数组并支持动态长度控制。
# 设计层：这是一个"代理/适配器"模式——
#           对外表现为 DataFrame 式的 .Open/.Close 属性，
#           内部返回 _Array（ndarray 子类）以保证性能。
#           Python 高级特性：__getattr__ 和 __getitem__ 的重载。
class _Data:
    """
    A data array accessor. Provides access to OHLCV "columns"
    as a standard `pd.DataFrame` would, except it's not a DataFrame
    and the returned "series" are _not_ `pd.Series` but `np.ndarray`
    for performance reasons.
    """
    def __init__(self, df: pd.DataFrame):
        self.__df = df                # 持有的原始 DataFrame
        self.__len = len(df)          # 当前"可见"长度（回测中逐步增加）
        self.__pip: Optional[float] = None   # pip 值（价格最小变动单位），惰性计算
        self.__cache: Dict[str, _Array] = {}  # 截断长度后的数组缓存
        self.__arrays: Dict[str, _Array] = {}  # 完整长度的数组缓存（列名→_Array）
        self._update()  # 初始化时构建完整数组缓存

    # 功能层：使用 data['Close'] 语法访问列
    def __getitem__(self, item):
        return self.__get_array(item)

    # 功能层：使用 data.Close 语法访问列
    # 设计层：__getattr__ 是 Python 的属性回退机制——
    #          当常规属性查找失败时被调用。
    #          KeyError 被转换为更语义化的 AttributeError。
    def __getattr__(self, item):
        try:
            return self.__get_array(item)
        except KeyError:
            raise AttributeError(f"Column '{item}' not in data") from None

    # 功能层：设置当前回测的可视长度（每次推进一根 K 线后调用）
    #         同时清空缓存，确保下次访问拿到新长度的数组切片。
    def _set_length(self, length):
        self.__len = length
        self.__cache.clear()

    # 功能层：重建完整数组缓存（在 __init__ 和 DataFrame 更新后调用）
    #         将 DataFrame 的每一列转换为携带 index 元数据的 _Array。
    def _update(self):
        index = self.__df.index.copy()
        self.__arrays = {col: _Array(arr, index=index)
                         for col, arr in self.__df.items()}
        # __index 列保留为 pd.DatetimeIndex（非 _Array），
        # 因为 pd.Timestamp 的 API 更友好（如 .dayofweek、.days 等）
        self.__arrays['__index'] = index

    # 功能层：友好的调试表示
    def __repr__(self):
        i = min(self.__len, len(self.__df)) - 1
        index = self.__arrays['__index'][i]
        items = ', '.join(f'{k}={v}' for k, v in self.__df.iloc[i].items())
        return f'<Data i={i} ({index}) {items}>'

    def __len__(self):
        return self.__len

    # 功能层：返回当前可见范围内的 DataFrame 切片
    @property
    def df(self) -> pd.DataFrame:
        return (self.__df.iloc[:self.__len]
                if self.__len < len(self.__df)
                else self.__df)

    # 功能层：动态推断价格的最小变动单位（pip size）
    #         通过分析 Close 价格字符串的小数位数来自动判断。
    #         例如：价格为 "1.2345" 则 pip = 0.0001
    @property
    def pip(self) -> float:
        if self.__pip is None:
            self.__pip = float(10**-np.median([len(s.partition('.')[-1])
                                               for s in self.__arrays['Close'].astype(str)]))
        return self.__pip

    # --- 内部数组获取方法 ---
    # 功能层：从缓存获取指定列的 _Array 切片（长度为 self.__len）
    # 设计层：使用字典缓存避免每次属性访问都创建新的切片对象。
    #          切片操作 [:self.__len] 产生视图（view）而非拷贝，内存效率高。
    def __get_array(self, key) -> _Array:
        arr = self.__cache.get(key)
        if arr is None:
            arr = self.__cache[key] = cast(_Array, self.__arrays[key][:self.__len])
        return arr

    # --- OHLCV 属性访问器 ---
    # 上下文层：这些 @property 定义了 .Open/.High/.Low/.Close/.Volume 五个标准列，
    #          是回测中最频繁访问的属性。它们返回的是 _Array（ndarray 子类），
    #          保证了回测热路径上不经过 pandas 的性能开销。

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

    # --- Pickle 序列化支持 ---
    # 上下文层：由于定义了 catch-all 的 __getattr__，
    #          必须显式提供 __getstate__/__setstate__ 来支持 pickle。
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state


# --- SharedMemory 兼容性处理 ---
# 上下文层：Python 3.13 修改了 SharedMemory 的 API，
#          使得 track 参数可以直接传递给构造函数。
#          之前的版本中需要通过变通方法实现。
# 设计层：通过 sys.version_info 进行版本分支，
#          确保代码在 Python 3.9-3.12 和 3.13+ 上都能正常运行。
if sys.version_info >= (3, 13):
    # 功能层：Python 3.13+ 原生支持 track 参数，直接使用
    SharedMemory = _mpshm.SharedMemory
else:
    # Python 3.12 及以下：需要手动处理资源追踪
    class SharedMemory(_mpshm.SharedMemory):
        # 功能层：类级别的线程锁，确保多线程创建 SharedMemory 时互斥
        # 来源：https://github.com/python/cpython/issues/82300#issuecomment-2169035092
        __lock = Lock()

        def __init__(self, *args, track: bool = True, **kwargs):
            self._track = track
            if track:
                # 功能层：需要追踪时直接调用父类构造
                return super().__init__(*args, **kwargs)
            # 功能层：不需要追踪时，通过 patch 临时禁用资源追踪注册
            # 设计层：使用 with 语句和上一节定义的 patch 上下文管理器
            with self.__lock:
                with patch(_mprt, 'register', lambda *a, **kw: None):
                    super().__init__(*args, **kwargs)

        def unlink(self):
            # 功能层：POSIX 系统上直接调用底层的共享内存释放
            if _mpshm._USE_POSIX and self._name:
                _mpshm._posixshmem.shm_unlink(self._name)
                if self._track:
                    _mprt.unregister(self._name, "shared_memory")


# ============================================================
# SharedMemoryManager —— 共享内存管理器
# ============================================================
# 上下文层：在 Backtest.optimize() 多进程优化中，
#           需要将 DataFrame 和数组通过共享内存传递给子进程。
#           这种方式比 pickle 传递大数据集更高效。
# 设计层：上下文管理器协议（__enter__/__exit__）确保资源释放。
#          Python 高级特性：上下文管理器 + 共享内存。
class SharedMemoryManager:
    """
    A simple shared memory contextmanager based on
    https://docs.python.org/3/library/multiprocessing.shared_memory.html#multiprocessing.shared_memory.SharedMemory
    """
    def __init__(self, create=False) -> None:
        # 功能层：_shms 列表持有所有创建的共享内存引用
        #         保持引用是 Windows 上的关键要求
        self._shms: list[SharedMemory] = []
        self.__create = create

    # 功能层：创建一个新的共享内存块
    def SharedMemory(self, *, name=None, create=False, size=0, track=True):
        shm = SharedMemory(name=name, create=create, size=size, track=track)
        shm._create = create
        # 上下文层：必须保持对共享内存的引用，否则 Windows 上会被垃圾回收
        # 参考：https://stackoverflow.com/questions/74193377/
        self._shms.append(shm)
        return shm

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        # 功能层：退出时关闭并（如果是创建者）释放所有共享内存块
        for shm in self._shms:
            try:
                shm.close()       # 关闭当前进程的连接
                if shm._create:
                    shm.unlink()  # 创建者负责释放底层共享内存
            except Exception:
                warnings.warn(f'Failed to unlink shared memory {shm.name!r}',
                              category=ResourceWarning, stacklevel=2)
                raise

    # --- 序列化辅助方法 ---
    # 功能层：将一维 NumPy 数组写入共享内存
    # 返回 (共享内存名称, 形状, 数据类型) 用于接收端还原
    def arr2shm(self, vals):
        """Array to shared memory. Returns (shm_name, shape, dtype) used for restore."""
        assert vals.ndim == 1, (vals.ndim, vals.shape, vals)
        shm = self.SharedMemory(size=vals.nbytes, create=True)
        # 上下文层：处理带时区的 pandas datetime——numpy 不直接支持 tz-aware 类型
        buf = np.ndarray(vals.shape, dtype=vals.dtype.base, buffer=shm.buf)
        has_tz = getattr(vals.dtype, 'tz', None)
        buf[:] = vals.tz_localize(None) if has_tz else vals  # 复制到共享内存
        return shm.name, vals.shape, vals.dtype

    # 功能层：将整个 DataFrame 写入共享内存
    #         对每列调用 arr2shm，索引也作为特殊列处理
    def df2shm(self, df):
        return tuple((
            (column, *self.arr2shm(values))
            for column, values in chain([(self._DF_INDEX_COL, df.index)], df.items())
        ))

    # 功能层：从共享内存恢复为 pandas Series
    @staticmethod
    def shm2s(shm, shape, dtype) -> pd.Series:
        arr = np.ndarray(shape, dtype=dtype.base, buffer=shm.buf)
        arr.setflags(write=False)  # 设为只读，防止意外的跨进程修改
        return pd.Series(arr, dtype=dtype)

    # 功能层：索引在共享内存中的列名标记
    _DF_INDEX_COL = '__bt_index'

    # 功能层：从共享内存恢复为完整的 DataFrame
    # 设计层：静态方法，不依赖实例状态，可被独立调用。
    @staticmethod
    def shm2df(data_shm):
        shm = [SharedMemory(name=name, create=False, track=False) for _, name, _, _ in data_shm]
        df = pd.DataFrame({
            col: SharedMemoryManager.shm2s(shm, shape, dtype)
            for shm, (col, _, shape, dtype) in zip(shm, data_shm)})
        df.set_index(SharedMemoryManager._DF_INDEX_COL, drop=True, inplace=True)
        df.index.name = None
        return df, shm
