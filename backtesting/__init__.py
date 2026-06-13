# ============================================================
# backtesting/__init__.py — 包入口模块
# ============================================================
# 上下文层：这是 backtesting 包的根入口。
#           当用户执行 `import backtesting` 或 `from backtesting import ...`
#           时，Python 解释器首先执行本文件。
#           本文件负责：
#           1. 提供包的顶层文档字符串（docstring）
#           2. 暴露对外公开的 API 对象（Backtest、Strategy 等）
#           3. 提供跨平台兼容的进程池配置
# ============================================================

# --- 包的顶层文档字符串 ---
# 功能层：本字符串是 backtesting 包的 __doc__，
#          同时也是 pdoc3 自动生成 API 文档的来源。
#          包含使用手册链接、教程、FAQ 和许可证说明。
# 设计层：使用 reStructuredText 风格的标记和链接引用语法，
#          pdoc3 将其渲染为 HTML 文档。
"""
![xkcd.com/1570](https://imgs.xkcd.com/comics/engineer_syllogism.png){: height=263}

## Manuals

* [**Quick Start User Guide**](../examples/Quick Start User Guide.html)

## Tutorials

The tutorials encompass most framework features, so it's important
and advisable to go through all of them. They are short.

* [Library of Utilities and Composable Base Strategies](../examples/Strategies Library.html)
* [Multiple Time Frames](../examples/Multiple Time Frames.html)
* [**Parameter Heatmap & Optimization**](../examples/Parameter Heatmap &amp; Optimization.html)
* [Trading with Machine Learning](../examples/Trading with Machine Learning.html)

These tutorials are also available as live Jupyter notebooks:
[![Binder](https://mybinder.org/badge_logo.svg)][binder]
[![Google Colab](https://colab.research.google.com/assets/colab-badge.png)][colab]
<br>In Colab, you might have to `!pip install backtesting`.

[binder]: \
    https://mybinder.org/v2/gh/kernc/backtesting.py/master?\
urlpath=lab%2Ftree%2Fdoc%2Fexamples%2FQuick%20Start%20User%20Guide.ipynb
[colab]: https://colab.research.google.com/github/kernc/backtesting.py/

## Video Tutorials

* Some [**coverage on YouTube**](https://github.com/kernc/backtesting.py/discussions/677).
* [YouTube search](https://www.youtube.com/results?q=%22backtesting.py%22)

## Example Strategies

* (contributions welcome)


.. tip::
    For an overview of recent changes, see
    [What's New, i.e. the **Change Log**](https://github.com/kernc/backtesting.py/blob/master/CHANGELOG.md).


## FAQ

Some answers to frequent and popular questions can be found on the
[issue tracker](https://github.com/kernc/backtesting.py/issues?q=label%3Aquestion+-label%3Ainvalid)
or on the [discussion forum](https://github.com/kernc/backtesting.py/discussions) on GitHub.
Please use the search!

## License

This software is licensed under the terms of [AGPL 3.0]{: rel=license},
meaning you can use it for any reasonable purpose and remain in
complete ownership of all the excellent trading strategies you produce,
but you are also encouraged to make sure any upgrades to _Backtesting.py_
itself find their way back to the community.

[AGPL 3.0]: https://www.gnu.org/licenses/agpl-3.0.html

# API Reference Documentation
"""

# --- 版本号获取 ---
# 上下文层：_version.py 由 setuptools_scm 在安装时自动生成。
#           开发模式下（未安装）该文件可能不存在，因此 try/except 兜底。
try:
    from ._version import version as __version__
except ImportError:
    # 功能层：未安装时显示占位符版本号
    __version__ = '?.?.?'  # Package not installed

# --- 公开 API 导入 ---
# 上下文层：以下 import 语句决定了 `from backtesting import ...` 可以使用哪些名称。
#           这是 Python 包的"门面"——用户不需要关心内部模块结构。
# 设计层：每个 import 后面都有 `# noqa: F401` 注释，
#          告诉 flake8/ruff "这个导入虽然看起来未使用，但它是为了重新导出"。

# 功能层：导入 lib 模块（策略辅助函数库），使其作为 backtesting.lib 可用
from . import lib  # noqa: F401

# 功能层：导入 set_bokeh_output 函数（配置 Bokeh 输出模式）
from ._plotting import set_bokeh_output  # noqa: F401

# 功能层：导入 try_ 工具函数（忽略特定异常的辅助函数）
from ._util import try_

# 功能层：导入核心类 Backtest（回测引擎）和 Strategy（策略基类）
from .backtesting import Backtest, Strategy  # noqa: F401


# --- 可替换的进程池 Pool ---
# 上下文层：backtesting.Pool 用于 Backtest.optimize() 的并行参数优化。
#           用户可以替换这个 Pool 为自己的实现（如 multiprocessing.Pool）。
# 设计层：这是一个"可覆盖的默认值"模式。
#           不为 Pool 设置类型注解，允许用户替换为任意兼容对象。
# 功能层：返回适用于当前平台的进程池工厂函数。
def Pool(processes=None, initializer=None, initargs=()):
    # multiprocessing：Python 标准库的跨平台多进程支持
    import multiprocessing as mp
    import sys

    # 上下文层：Linux 上默认使用 fork 方式创建子进程（速度快、内存共享），
    #          但 Python 3.14 改变了默认值。这里主动恢复为 fork 以保证性能。
    # 设计层：在 Linux 上，fork 比 spawn 更快，因为子进程直接继承父进程内存空间。
    if sys.platform.startswith('linux') and mp.get_start_method(allow_none=True) != 'fork':
        # try_：如果 set_start_method 已经被其他代码调用过，忽略错误
        try_(lambda: mp.set_start_method('fork'))

    # 上下文层：Windows 和 macOS 默认使用 spawn 方式。
    #          spawn 下多进程优化有额外限制，这里降级为线程池。
    if mp.get_start_method() == 'spawn':
        import warnings
        # 功能层：警告用户 spawn 模式下多进程优化的限制
        warnings.warn(
            "If you want to use multi-process optimization with "
            "`multiprocessing.get_start_method() == 'spawn'` (e.g. on Windows),"
            "set `backtesting.Pool = multiprocessing.Pool` (or of the desired context) "
            "and hide `bt.optimize()` call behind a `if __name__ == '__main__'` guard. "
            "Currently using thread-based paralellism, "
            "which might be slightly slower for non-numpy / non-GIL-releasing code. "
            "See https://github.com/kernc/backtesting.py/issues/1256",
            category=RuntimeWarning, stacklevel=3)

        # 功能层：使用线程池（multiprocessing.dummy.Pool）作为降级方案
        # 设计层：线程池 API 与进程池完全一致，可以无缝替换
        from multiprocessing.dummy import Pool
        return Pool(processes, initializer, initargs)
    else:
        # 功能层：fork 模式下直接使用标准多进程池
        return mp.Pool(processes, initializer, initargs)
