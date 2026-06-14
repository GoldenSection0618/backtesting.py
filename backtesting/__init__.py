# ============================================================
# backtesting/__init__.py -- 包入口
# ============================================================
# 上下文层: `import backtesting` 时最先执行.负责暴露公开 API, 
#           读取版本号, 提供跨平台进程池.
# 功能层: 从子模块导入 Backtest/Strategy 等核心类并重新导出, 
#          用户只需 `from backtesting import Backtest`.
# 设计层: noqa: F401 告诉 lint 这些导入是为了重新导出而非本文件使用.

"""
包入口 -- 当用户 `import backtesting` 时这里首先执行.
职责: 暴露公开 API, 获取版本号, 提供跨平台的进程池.
下面的内容同时也是 pdoc3 生成文档的原材料.

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

# 从 setuptools_scm 生成的 _version.py 读版本号
try:
    from ._version import version as __version__
except ImportError:
    __version__ = '?.?.?'  # Package not installed

# 重新导出----用户只需 from backtesting import Backtest
from . import lib  # noqa: F401
from ._plotting import set_bokeh_output  # noqa: F401
from ._util import try_
from .backtesting import Backtest, Strategy  # noqa: F401


# 可替换的进程池, 用于 Backtest.optimize() 并行执行
# 用户可以 backtesting.Pool = multiprocessing.Pool 来自定义
def Pool(processes=None, initializer=None, initargs=()):
    import multiprocessing as mp
    import sys

    # Linux 上 fork 比 spawn 快(子进程继承父进程内存), 
    # Python 3.14 改了默认值, 这里主动切回 fork
    if sys.platform.startswith('linux') and mp.get_start_method(allow_none=True) != 'fork':
        try_(lambda: mp.set_start_method('fork'))

    # Windows/macOS 默认 spawn, 降级为线程池
    if mp.get_start_method() == 'spawn':
        import warnings
        warnings.warn(
            "If you want to use multi-process optimization with "
            "`multiprocessing.get_start_method() == 'spawn'` (e.g. on Windows),"
            "set `backtesting.Pool = multiprocessing.Pool` (or of the desired context) "
            "and hide `bt.optimize()` call behind a `if __name__ == '__main__'` guard. "
            "Currently using thread-based paralellism, "
            "which might be slightly slower for non-numpy / non-GIL-releasing code. "
            "See https://github.com/kernc/backtesting.py/issues/1256",
            category=RuntimeWarning, stacklevel=3)
        from multiprocessing.dummy import Pool
        return Pool(processes, initializer, initargs)
    else:
        return mp.Pool(processes, initializer, initargs)
