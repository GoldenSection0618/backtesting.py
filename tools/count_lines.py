# ============================================================
# tools/count_lines.py —— 课程项目代码规模统计脚本
# ============================================================
# 上下文层：本脚本统计纳入逐行注释的全部文件的代码规模，
#           验证课程要求的"约 2000 行有效代码"。
#           输出终端表格并写入 line_count_report.txt。
# 运行方式：python tools/count_lines.py
# ============================================================

from pathlib import Path


# --- 统计目标文件列表 ---
# 功能层：包含所有纳入注释的 .py、.js 和配置文件
#         排除 README.md、LICENSE.md 和 CSV 数据文件
FILES = [
    ".gitignore",
    "MANIFEST.in",
    "requirements.txt",
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
    "constraints.txt",
    "run_demo.py",
    "backtesting/__init__.py",
    "backtesting/backtesting.py",
    "backtesting/lib.py",
    "backtesting/_util.py",
    "backtesting/_stats.py",
    "backtesting/_plotting.py",
    "backtesting/autoscale_cb.js",
    "backtesting/test/__init__.py",
    "backtesting/test/__main__.py",
    "backtesting/test/_test.py",
    "tools/count_lines.py",
]


def count_file(path: Path):
    """统计单个文件的总行数和非空行数"""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    nonempty = [line for line in lines if line.strip()]
    return len(lines), len(nonempty)


def main():
    total_lines = 0      # 累计总行数
    total_nonempty = 0   # 累计非空行数（即"有效代码行"）
    rows = []

    # 功能层：逐个文件统计
    for name in FILES:
        path = Path(name)
        if not path.exists():
            rows.append((name, "MISSING", "MISSING"))
            continue

        lines, nonempty = count_file(path)
        total_lines += lines
        total_nonempty += nonempty
        rows.append((name, lines, nonempty))

    # 功能层：构建 Markdown 表格格式输出
    output = []
    output.append("File | Lines | Non-empty lines")
    output.append("--- | ---: | ---:")

    for name, lines, nonempty in rows:
        output.append(f"{name} | {lines} | {nonempty}")

    output.append(f"TOTAL | {total_lines} | {total_nonempty}")

    # 功能层：终端打印 + 写入文件
    report = "\n".join(output)
    print(report)
    Path("line_count_report.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
