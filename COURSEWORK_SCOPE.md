# 课程大作业范围说明

本项目基于 [Backtesting.py](https://github.com/kernc/backtesting.py) 开源回测框架进行复现和中文注释, 当前分支为 `coursework-annotated`。纳入注释统计的文件共 19 个, 总计 5018 行、4248 非空行, 超过课程要求的约 2000 行有效代码标准。工程使用了元类、上下文管理器、生成器、ndarray 子类化等 Python 高级特性, 适合作为源码阅读和工程设计分析的对象。

## 文件分类

### 核心源码 (逐行或关键块注释)

- [`backtesting/backtesting.py`](backtesting/backtesting.py) -- 回测引擎核心, 包含 Strategy, Order, Trade, _Broker, Backtest 五个核心类
- [`backtesting/lib.py`](backtesting/lib.py) -- 策略辅助函数库, 提供 crossover, resample_apply, SignalStrategy, TrailingStrategy 等工具
- [`backtesting/_util.py`](backtesting/_util.py) -- 内部工具模块: _Array (ndarray 子类), _Data (OHLCV 访问器), 共享内存管理器
- [`backtesting/_stats.py`](backtesting/_stats.py) -- 统计指标计算, 输出约 30 项金融指标 (Sharpe, Sortino, 回撤, 胜率等)
- [`backtesting/_plotting.py`](backtesting/_plotting.py) -- 基于 [Bokeh](https://bokeh.org) 的交互式可视化, 生成 K 线图, 权益曲线, 回撤和交易标记
- [`backtesting/autoscale_cb.js`](backtesting/autoscale_cb.js) -- 图表 Y 轴自动缩放的 JavaScript 回调
- [`backtesting/__init__.py`](backtesting/__init__.py) -- 包入口, 重新导出 Backtest, Strategy 等公开 API
- [`backtesting/test/__init__.py`](backtesting/test/__init__.py) -- 示例数据加载 (GOOG, BTCUSD, EURUSD) 及 SMA 函数定义
- [`backtesting/test/__main__.py`](backtesting/test/__main__.py) -- 测试运行入口 (`python -m backtesting.test`)
- [`backtesting/test/_test.py`](backtesting/test/_test.py) -- 76 个测试用例, 覆盖引擎, 策略, 优化, 绘图和回归防护

### 工程配置 (已注释)

- [`setup.py`](setup.py) -- 安装与打包配置, 包含 install_requires 和 extras_require
- [`setup.cfg`](setup.cfg) -- flake8 代码风格, mypy 类型检查, coverage 覆盖率配置
- [`pyproject.toml`](pyproject.toml) -- ruff linter 规则配置
- [`MANIFEST.in`](MANIFEST.in) -- sdist 源码分发包文件清单
- [`requirements.txt`](requirements.txt) -- pip 安装入口 (`.[test]`)
- [`.gitignore`](.gitignore) -- Git 忽略规则

### 辅助交付件

- [`run_demo.py`](run_demo.py) -- SMA 交叉策略最小运行示例
- [`tools/count_lines.py`](tools/count_lines.py) -- 代码行数统计脚本
- [`line_count_report.txt`](line_count_report.txt) -- 行数统计结果 (19 个文件, 5018 行, 4248 非空行)
- [`constraints.txt`](constraints.txt) -- 环境版本约束, 锁定 `pandas<3` 以避开上游兼容性问题
- [`ANNOTATION_GUIDE.md`](ANNOTATION_GUIDE.md) -- 中文注释规范说明
- 本文件

### 保留原貌, 不原地注释

- [`README.md`](README.md) -- 原项目说明文档, 包含使用手册, 教程链接和 FAQ。对其进行注释会破坏文档的独立可读性。
- [`LICENSE.md`](LICENSE.md) -- [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html) 开源协议文本。法律文本不应被修改。
- `backtesting/test/GOOG.csv`, `BTCUSD.csv`, `EURUSD.csv` -- 示例行情数据文件。插入非 CSV 格式的文本会导致 pandas `read_csv` 解析失败。

## 运行验证

使用约束环境安装依赖:

```bash
python -m pip install -r requirements.txt -c constraints.txt
```

`constraints.txt` 将 pandas 约束在 3.0 以下, 绕过了 pandas 3.x 下 `FractionalBacktest` 对只读 NumPy 数组执行原地操作导致的兼容性问题。

`python run_demo.py` 成功输出约 30 项回测统计指标并生成 `sma_cross_result.html` 交互式图表。

`python -m backtesting.test` 在约束环境下结果为 76 tests OK, 1 skipped。跳过项 `test_examples` 依赖于已删除的 `doc/` 示例文档目录, 属于预期行为, 不影响核心回测功能的验证。
