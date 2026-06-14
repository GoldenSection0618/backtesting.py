# ============================================================
# backtesting/test/__main__.py -- 测试运行入口
# ============================================================
# 上下文层: `python -m backtesting.test` 执行本文件.
# 功能层: 将警告升级为异常, 然后跑 _test.py 里全部测试.
# 设计层: `if __name__ == '__main__'` 防止被 import 时误跑测试.

import unittest
import warnings


if __name__ == '__main__':
    warnings.filterwarnings('error')   # 意外警告直接失败, 防止被淹没
    unittest.main(module='backtesting.test._test', verbosity=2)
