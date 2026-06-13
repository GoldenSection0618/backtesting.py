# 课程大作业范围说明

本项目基于 Backtesting.py 开源项目进行复现和注释。当前分支保留核心源码、测试代码、示例数据和工程配置文件，用于满足真实 Python 工程复现、运行验证、逐行注释和剖析报告要求。

## 保留内容

1. `backtesting/`：核心 Python 源码、绘图模块、统计模块、测试代码和示例数据。
2. `setup.py`、`setup.cfg`、`pyproject.toml`、`MANIFEST.in`、`requirements.txt`：安装、依赖、打包、代码规范和测试配置。
3. `README.md`：原项目说明和运行示例来源。
4. `LICENSE.md`：开源协议文件。

## 逐行注释对象

逐行注释或逐行解释以下文件：

- Python 源码文件
- JavaScript 文件
- 工程配置文件

## 保留原貌的文件

以下文件不原地注释：

- `README.md`
- `LICENSE.md`
- `backtesting/test/GOOG.csv`
- `backtesting/test/BTCUSD.csv`
- `backtesting/test/EURUSD.csv`

原因是 README 是说明文档，LICENSE 是开源协议，CSV 是可运行示例依赖的数据文件。修改这些文件会破坏原始语义或数据读取。

## 运行验证

使用 `run_demo.py` 复现 SMA crossover 策略，输出回测统计结果并生成 `sma_cross_result.html`。
