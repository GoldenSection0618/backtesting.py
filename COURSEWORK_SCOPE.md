# 课程大作业范围说明

我选择 Backtesting.py 作为《Python 高级程序设计》大作业的复现和注释对象。
这是一个 GitHub 上真实的 Python 回测框架, 代码规模适中 (~6300 行), 用到了元类、上下文管理器、生成器、ndarray 子类化等 Python 高级特性。

## 我保留了哪些文件

这是我保留的核心源码 (也是逐行注释的对象):

- `backtesting/backtesting.py` -- 回测引擎, 约 1280 行, 包含 Strategy/Order/Trade/_Broker/Backtest
- `backtesting/lib.py` -- 策略工具库 (crossover, resample_apply, SignalStrategy 等)
- `backtesting/_util.py` -- 内部工具: _Array (ndarray 子类), _Data, 共享内存
- `backtesting/_stats.py` -- 统计指标: Sharpe, Sortino, 回撤, 胜率等约 30 项
- `backtesting/_plotting.py` -- Bokeh 可视化, 生成 K 线图+权益曲线+回撤+交易标记
- `backtesting/autoscale_cb.js` -- Y 轴自动缩放的 JS 回调
- `backtesting/__init__.py` -- 包入口, 暴露 Backtest/Strategy
- `backtesting/test/__init__.py` -- 加载 GOOG/BTCUSD/EURUSD 示例数据
- `backtesting/test/__main__.py` -- `python -m backtesting.test` 入口
- `backtesting/test/_test.py` -- 76 个测试用例

工程配置文件 (也做了注释):

- `setup.py` -- 安装/打包/依赖声明
- `setup.cfg` -- flake8, mypy, coverage 配置
- `pyproject.toml` -- ruff linter 配置
- `MANIFEST.in` -- sdist 打包清单
- `requirements.txt` -- pip 安装入口
- `.gitignore` -- Git 忽略规则

我额外加的辅助文件:

- `run_demo.py` -- 最小的 SMA 交叉策略, 用来验证工程能跑
- `tools/count_lines.py` -- 统计代码行数
- `line_count_report.txt` -- 行数统计结果 (19 个文件, ~6300 行 / ~5300 非空行)
- `constraints.txt` -- 锁定 pandas<3, 避免 FractionalBacktest 测试报错
- `ANNOTATION_GUIDE.md` -- 我的注释规范
- 本文件

## 我没动这些文件

- `README.md` -- 原作者的项目说明
- `LICENSE.md` -- AGPL-3.0 协议
- `backtesting/test/GOOG.csv`, `BTCUSD.csv`, `EURUSD.csv` -- 行情数据, 改了就跑不起来了

## 运行结果

```bash
python run_demo.py
```

能正常输出约 30 项回测统计 (收益率、夏普比、最大回撤等), 并生成 `sma_cross_result.html` 交互式图表。

完整测试 `python -m backtesting.test`: 76 tests, 1 error, 1 skipped。
- 1 error: FractionalBacktest -- pandas 3.x 下上游兼容性问题 (数组只读), 通过 `constraints.txt` 锁定 pandas<3 可避开
- 1 skipped: test_examples -- 我把 doc/ 目录删了, 测试找不到示例脚本, 这是预期行为
