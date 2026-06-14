# ============================================================
# setup.py — 安装和打包配置
# ============================================================
# 上下文层：pip install 会读取此文件，从中获取包名、依赖、版本号等元数据。
# 功能层：声明 backtesting 包的 install_requires（numpy/pandas/bokeh）、
#          可选依赖组（test/dev/doc）、PyPI 分类和搜索关键词。
# 设计层：版本号由 setuptools_scm 从 git tag 自动生成，无需手动维护。

import os
import sys

# 要求 Python 3.9+（PEP 585 内置泛型的最低版本）
if sys.version_info < (3, 9):
    sys.exit('ERROR: Backtesting.py requires Python 3.9+')


if __name__ == '__main__':
    from setuptools import setup, find_packages

    setup(
        name='backtesting',
        description="Backtest trading strategies in Python",
        license='AGPL-3.0',
        url='https://kernc.github.io/backtesting.py/',
        project_urls={
            'Documentation': 'https://kernc.github.io/backtesting.py/doc/backtesting/',
            'Source': 'https://github.com/kernc/backtesting.py/',
            'Tracker': 'https://github.com/kernc/backtesting.py/issues',
        },

        # PyPI 项目页的长描述从 README.md 读取
        long_description=open(os.path.join(os.path.dirname(__file__), 'README.md'),
                              encoding='utf-8').read(),
        long_description_content_type='text/markdown',

        # find_packages() 自动扫描含 __init__.py 的子目录
        packages=find_packages(),
        # 打包时也带上 CSV、JS 等非 .py 文件
        include_package_data=True,

        setup_requires=[
            'setuptools_git',
            'setuptools_scm',
        ],

        # 从 git tag 自动生成版本号，写入 backtesting/_version.py
        use_scm_version={
            'write_to': os.path.join('backtesting', '_version.py'),
        },

        # 核心运行时依赖
        install_requires=[
            'numpy >= 1.17.0',
            'pandas >= 0.25.0, != 0.25.0',            # 排除有 bug 的版本
            'bokeh >= 3.0.0, != 3.0.*, != 3.2.*',     # 排除不兼容版本
        ],

        # 可选依赖组（pip install backtesting[test] 安装）
        extras_require={
            'doc': [
                'pdoc3',
                'jupytext >= 1.3',
                'nbconvert',
                'ipykernel',
                'jupyter_client',
            ],
            'test': [
                'matplotlib',
                'scikit-learn',
                'sambo',
                'tqdm',
                'ipywidgets',
            ],
            'dev': [
                'flake8',
                'coverage',
                'mypy',
            ],
        },

        test_suite="backtesting.test",
        python_requires='>=3.9',
        author='Zach Lûster',

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

        keywords=[
            'algo', 'algorithmic', 'ashi', 'backtest', 'backtesting',
            'bitcoin', 'bokeh', 'bonds', 'candle', 'candlestick',
            'cboe', 'chart', 'cme', 'commodities', 'crash',
            'crypto', 'currency', 'doji', 'drawdown', 'equity',
            'etf', 'ethereum', 'exchange', 'finance', 'financial',
            'forecast', 'forex', 'fund', 'futures', 'fx',
            'fxpro', 'gold', 'heiken', 'historical', 'indicator',
            'invest', 'investing', 'investment', 'macd', 'market',
            'mechanical', 'money', 'oanda', 'ohlc', 'ohlcv',
            'order', 'price', 'profit', 'quant', 'quantitative',
            'rsi', 'silver', 'simulation', 'stocks', 'strategy',
            'ticker', 'trader', 'trading', 'tradingview', 'usd',
        ],
    )
