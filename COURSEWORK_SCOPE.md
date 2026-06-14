# 课程大作业范围说明

我选择 [Backtesting.py](https://github.com/kernc/backtesting.py) 作为《Python 高级程序设计》大作业的复现和注释对象。
这是一个 GitHub 上真实的 Python 回测框架, 代码规模适中 (~6300 行), 用到了元类、上下文管理器、生成器、ndarray 子类化等 Python 高级特性。

## 文件分类

### 核心源码 (逐行或关键块注释)

- [`backtesting/backtesting.py`](backtesting/backtesting.py) -- 回测引擎, 约 1280 行, 包含 Strategy/Order/Trade/_Broker/Backtest
- [`backtesting/lib.py`](backtesting/lib.py) -- 策略工具库 (crossover, resample_apply, SignalStrategy 等)
- [`backtesting/_util.py`](backtesting/_util.py) -- 内部工具: _Array (ndarray 子类), _Data, 共享内存
- [`backtesting/_stats.py`](backtesting/_stats.py) -- 统计指标: Sharpe, Sortino, 回撤, 胜率等约 30 项
- [`backtesting/_plotting.py`](backtesting/_plotting.py) -- [Bokeh](https://bokeh.org) 可视化, 生成 K 线图+权益曲线+回撤+交易标记
- [`backtesting/autoscale_cb.js`](backtesting/autoscale_cb.js) -- Y 轴自动缩放的 JS 回调
- [`backtesting/__init__.py`](backtesting/__init__.py) -- 包入口, 暴露 Backtest/Strategy
- [`backtesting/test/__init__.py`](backtesting/test/__init__.py) -- 加载 GOOG/BTCUSD/EURUSD 示例数据
- [`backtesting/test/__main__.py`](backtesting/test/__main__.py) -- `python -m backtesting.test` 入口
- [`backtesting/test/_test.py`](backtesting/test/_test.py) -- 76 个测试用例

### 工程配置 (已注释)

- [`setup.py`](setup.py) -- 安装/打包/依赖声明
- [`setup.cfg`](setup.cfg) -- flake8, mypy, coverage 配置
- [`pyproject.toml`](pyproject.toml) -- ruff linter 配置
- [`MANIFEST.in`](MANIFEST.in) -- sdist 打包清单
- [`requirements.txt`](requirements.txt) -- pip 安装入口
- [`.gitignore`](.gitignore) -- Git 忽略规则

### 辅助交付件

- [`run_demo.py`](run_demo.py) -- SMA 交叉策略运行示例
- [`tools/count_lines.py`](tools/count_lines.py) -- 代码行数统计
- [`line_count_report.txt`](line_count_report.txt) -- 行数统计结果 (19 个文件, ~6300 / ~5300 非空行)
- [`constraints.txt`](constraints.txt) -- 环境版本约束 (锁定 pandas<3)
- [`ANNOTATION_GUIDE.md`](ANNOTATION_GUIDE.md) -- 注释规范
- 本文件

### 保留原貌

- [`README.md`](README.md) -- 原项目说明
- [`LICENSE.md`](LICENSE.md) -- [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html) 协议
- `backtesting/test/GOOG.csv`, `BTCUSD.csv`, `EURUSD.csv` -- 行情数据, 修改会破坏读取

## 运行结果

```bash
python run_demo.py
```

正常输出约 30 项回测统计 (收益率、夏普比、最大回撤等), 生成 `sma_cross_result.html` 交互式图表。

`python -m backtesting.test`: 76 tests, 1 error, 1 skipped。

- 1 error: FractionalBacktest -- pandas 3.x 下[上游兼容性问题](https://github.com/kernc/backtesting.py/issues/134) (数组只读), `constraints.txt` 锁定 pandas<3 可避开
- 1 skipped: test_examples -- doc/ 目录已删除, 预期行为
