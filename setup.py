# ============================================================
# setup.py — Backtesting.py 安装与打包配置脚本
# ============================================================
# 上下文层：setup.py 是 Python 项目的"入口脚本"。
#           pip install / build / twine 等工具通过它获取项目的
#           元数据、依赖、版本号、包结构等信息。
#           本文件是 Python 包管理的核心，连接开发者、PyPI 和最终用户。
# 设计层：使用 setuptools（Python 官方推荐的打包库），
#          配置采用声明式 setup() 函数调用，
#          版本号自动从 git tag 派生（setuptools_scm）。
# 功能层：定义包名、依赖、Python 版本要求、分类标签等。
# ============================================================

# --- 标准库导入 ---
# os：路径拼接（读取 README.md）
# sys：检查 Python 版本
import os
import sys

# 功能层：要求 Python 3.9 或更高版本。
# 设计层：使用 sys.version_info 而非 sys.version 字符串比较，
#          前者返回命名元组，便于精确比较主版本号/次版本号。
# 上下文层：Python 3.9 是引入 PEP 585（内置泛型）的最低版本。
#          在 3.8 及以下运行会直接退出并提示错误。
if sys.version_info < (3, 9):
    sys.exit('ERROR: Backtesting.py requires Python 3.9+')


# 功能层：仅在直接执行本脚本时（非 import）才调用 setup()。
# 设计层：`if __name__ == '__main__':` 是 Python 的标准惯用法。
#         这允许本文件既可执行（python setup.py），也可被导入
#         （某些工具会 import setup 读取元数据）。
if __name__ == '__main__':
    # setuptools：Python 打包工具，提供 setup() 函数和 find_packages()
    from setuptools import setup, find_packages

    # ----------------------------------------------------------
    # setup() —— 项目的完整声明
    # 上下文层：这是整个安装配置的核心。pip install 读取这里的一切。
    # ----------------------------------------------------------
    setup(
        # --- 基本信息 ---
        # 功能层：PyPI 上的包名，用户通过 `pip install backtesting` 安装
        name='backtesting',

        # 功能层：简短描述，出现在 PyPI 搜索结果中
        description="Backtest trading strategies in Python",

        # 功能层：开源许可证标识（AGPL-3.0 或更高版本）
        # 设计层：AGPL 是 GPL 的强化版，要求网络服务使用者也必须开源
        license='AGPL-3.0',

        # 功能层：项目主页/文档站 URL
        url='https://kernc.github.io/backtesting.py/',

        # 功能层：PyPI 侧边栏的项目链接
        # 设计层：使用字典结构，key 是链接类别（可被 PyPI 识别），value 是 URL
        project_urls={
            'Documentation': 'https://kernc.github.io/backtesting.py/doc/backtesting/',
            'Source': 'https://github.com/kernc/backtesting.py/',
            'Tracker': 'https://github.com/kernc/backtesting.py/issues',
        },

        # --- README ---
        # 功能层：从 README.md 读取长描述，显示在 PyPI 项目页面上
        # 设计层：使用 os.path.join 拼接路径，跨平台安全
        #         encoding 参数确保跨平台正确读取 UTF-8
        long_description=open(os.path.join(os.path.dirname(__file__), 'README.md'),
                              encoding='utf-8').read(),
        # 功能层：声明长描述的格式为 Markdown
        long_description_content_type='text/markdown',

        # --- 包发现 ---
        # 功能层：自动发现并包含 backtesting 目录下的所有 Python 子包
        # 设计层：find_packages() 是 setuptools 提供的工具函数，
        #          它扫描项目目录，自动定位包含 __init__.py 的子目录
        packages=find_packages(),

        # 功能层：包含包数据文件（如 CSV、JS 等非 .py 文件）
        #         这些文件在 MANIFEST.in 中声明，由本选项控制是否安装
        include_package_data=True,

        # --- 构建依赖 ---
        # 上下文层：setup_requires 指定在运行 setup.py 本身之前需要安装的包
        #          这些包不会被安装到目标环境，只在构建阶段临时使用
        setup_requires=[
            'setuptools_git',   # 从 git 仓库历史生成文件清单（替代 MANIFEST.in）
            'setuptools_scm',   # 从 git tag 生成 Python 版本号
        ],

        # --- 版本管理 ---
        # 功能层：使用 setuptools_scm 从 git tag 自动生成版本号
        # 设计层：setuptools_scm 读取最近的 git tag 和 commit 距离，
        #          自动生成如 "0.6.6.dev7+g219c22398" 的版本字符串。
        #          这样每次 commit 都有唯一可追溯的版本号，避免手动维护。
        #          write_to 参数将版本号写入 backtesting/_version.py，
        #          这样运行时无需导入 setuptools_scm。
        use_scm_version={
            'write_to': os.path.join('backtesting', '_version.py'),
        },

        # --- 核心依赖 ---
        # 上下文层：install_requires 声明运行时必须安装的包。
        #           pip install backtesting 会自动安装这些依赖。
        # 设计层：使用 PEP 440 版本规范符。
        #          != 0.25.0 排除了 pandas 的一个已知 bug 版本。
        #          != 3.0.* 和 != 3.2.* 排除了 bokeh 的不兼容版本。
        install_requires=[
            'numpy >= 1.17.0',
            'pandas >= 0.25.0, != 0.25.0',
            'bokeh >= 3.0.0, != 3.0.*, != 3.2.*',
        ],

        # --- 可选依赖组 ---
        # 上下文层：extras_require 定义可选依赖组。
        #          用户可通过 pip install backtesting[test] 安装对应组。
        # 设计层：按功能分组（文档、测试、开发），
        #          避免所有用户都被迫安装不必要的大依赖。
        extras_require={
            # 功能层：文档构建依赖（pdoc3 生成 API 文档，jupytext 转换 notebook）
            'doc': [
                'pdoc3',
                'jupytext >= 1.3',
                'nbconvert',
                'ipykernel',       # nbconvert 在后台需要的内核
                'jupyter_client',  # nbconvert 的通信客户端
            ],
            # 功能层：测试依赖（matplotlib 绘图对比，scikit-learn 数据生成，
            #          sambo 贝叶斯优化，tqdm 进度条）
            'test': [
                'matplotlib',
                'scikit-learn',
                'sambo',
                'tqdm',
                'ipywidgets',  # tqdm 在 Jupyter 中需要的交互组件
            ],
            # 功能层：开发依赖（代码检查、覆盖率、类型检查）
            'dev': [
                'flake8',
                'coverage',
                'mypy',
            ],
        },

        # --- 测试配置 ---
        # 功能层：声明测试套件的包路径
        #         运行 `python -m unittest backtesting.test` 时
        #         自动发现该包下的测试用例
        test_suite="backtesting.test",

        # 功能层：声明最低 Python 版本要求（PEP 440）
        #         PyPI 会根据此字段阻止不兼容的 Python 版本安装
        python_requires='>=3.9',

        # --- 作者信息 ---
        author='Zach Lûster',

        # --- PyPI 分类标签 ---
        # 上下文层：classifiers 是 PyPI 的标准分类体系。
        #          正确的分类标签有助于用户在 PyPI 上通过分类浏览找到本项目。
        classifiers=[
            'Intended Audience :: Financial and Insurance Industry',
            'Intended Audience :: Science/Research',
            'Framework :: Jupyter',
            'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3 :: Only',
            'Topic :: Office/Business :: Financial :: Investment',
            'Topic :: Scientific/Engineering :: Visualization',
        ],

        # --- 搜索关键词 ---
        # 功能层：PyPI 搜索索引的关键词列表
        # 设计层：覆盖广泛的相关术语（金融、交易、技术指标、加密货币等），
        #          提升在 PyPI 上的可发现性
        keywords=[
            'algo',
            'algorithmic',
            'ashi',
            'backtest',
            'backtesting',
            'bitcoin',
            'bokeh',
            'bonds',
            'candle',
            'candlestick',
            'cboe',
            'chart',
            'cme',
            'commodities',
            'crash',
            'crypto',
            'currency',
            'doji',
            'drawdown',
            'equity',
            'etf',
            'ethereum',
            'exchange',
            'finance',
            'financial',
            'forecast',
            'forex',
            'fund',
            'futures',
            'fx',
            'fxpro',
            'gold',
            'heiken',
            'historical',
            'indicator',
            'invest',
            'investing',
            'investment',
            'macd',
            'market',
            'mechanical',
            'money',
            'oanda',
            'ohlc',
            'ohlcv',
            'order',
            'price',
            'profit',
            'quant',
            'quantitative',
            'rsi',
            'silver',
            'simulation',
            'stocks',
            'strategy',
            'ticker',
            'trader',
            'trading',
            'tradingview',
            'usd',
        ],
    )
