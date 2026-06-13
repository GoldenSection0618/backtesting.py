# ============================================================
# backtesting/test/__main__.py — 测试运行入口
# ============================================================
# 上下文层：当执行 `python -m backtesting.test` 时，Python 自动运行本文件。
#           它是测试套件的启动器，负责配置警告过滤并调用 unittest。
# 设计层：`python -m package` 会执行 package/__main__.py 中的代码。
#          这是 Python 标准库推荐的模块执行方式。
# ============================================================

import unittest
import warnings


# 功能层：仅当本文件被直接执行时（非 import）运行测试。
# 设计层：`if __name__ == '__main__':` 是 Python 的标准惯用法，
#          防止 import 本模块时不必要地运行整个测试套件。
if __name__ == '__main__':
    # 功能层：将所有警告升级为异常。
    # 设计层：这确保测试过程中任何意外警告（如弃用警告）都会被立即捕获，
    #          而不是在 CI 日志中被淹没。
    warnings.filterwarnings('error')

    # 功能层：运行 backtesting.test._test 模块中的全部测试用例
    #          verbosity=2 输出每个测试方法和结果（比默认的按类汇总更详细）
    unittest.main(module='backtesting.test._test', verbosity=2)
