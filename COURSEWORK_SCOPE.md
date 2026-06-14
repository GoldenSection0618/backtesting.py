# 课程大作业范围说明

本项目基于 Backtesting.py 开源项目进行复现和注释。当前分支保留核心源码、测试代码、示例数据和工程配置文件，用于满足真实 Python 工程复现、运行验证、逐行注释和剖析报告要求。

## 保留内容

核心源码:

- `backtesting/backtesting.py` -- 回测引擎、Strategy、Order、Trade、_Broker
- `backtesting/lib.py` -- 策略辅助函数库
- `backtesting/_util.py` -- 内部工具和数据结构
- `backtesting/_stats.py` -- 统计指标计算
- `backtesting/_plotting.py` -- Bokeh 可视化
- `backtesting/autoscale_cb.js` -- Y 轴自动缩放回调
- `backtesting/__init__.py` -- 包入口
- `backtesting/test/__init__.py` -- 示例数据加载
- `backtesting/test/__main__.py` -- 测试运行入口
- `backtesting/test/_test.py` -- 76 个测试用例

工程配置:

- `setup.py`, `setup.cfg`, `pyproject.toml`, `MANIFEST.in`, `requirements.txt`

课程辅助交付件:

- `run_demo.py` -- SMA 交叉策略最小运行示例
- `tools/count_lines.py` -- 代码规模统计脚本
- `line_count_report.txt` -- 行数统计报告
- `constraints.txt` -- 环境版本约束 (锁定 pandas<3)
- `ANNOTATION_GUIDE.md` -- 注释规范说明
- `COURSEWORK_SCOPE.md` -- 本文件

## 逐行注释对象

纳入行数统计和逐行/关键块中文注释的文件共 19 个, 总计约 6300 行 / 5300 非空行:

- Python 源码: `backtesting/backtesting.py`, `lib.py`, `_util.py`, `_stats.py`, `_plotting.py`, `__init__.py`, `test/__init__.py`, `test/__main__.py`, `test/_test.py`, `setup.py`, `run_demo.py`, `tools/count_lines.py`
- JavaScript: `autoscale_cb.js`
- 配置文件: `.gitignore`, `MANIFEST.in`, `requirements.txt`, `constraints.txt`, `pyproject.toml`, `setup.cfg`

## 保留原貌的文件

以下文件不原地注释:

- `README.md` -- 原项目说明文档, 保留原始语义
- `LICENSE.md` -- 开源协议, 不应修改
- `backtesting/test/GOOG.csv` -- 示例行情数据, 修改会破坏 pandas 读取
- `backtesting/test/BTCUSD.csv` -- 同上
- `backtesting/test/EURUSD.csv` -- 同上

## 运行验证

使用 `run_demo.py` 复现 SMA crossover 策略, 输出回测统计结果并生成 `sma_cross_result.html`。

完整测试套件 (`python -m backtesting.test`): 76 tests, 1 error (FractionalBacktest 在 pandas 3.x 下的上游兼容性问题), 1 skipped (test_examples 因 doc/ 目录已删除)。课程复现环境通过 `constraints.txt` 约束 `pandas<3` 避开已知问题。
