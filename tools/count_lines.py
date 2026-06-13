from pathlib import Path


FILES = [
    ".gitignore",
    "MANIFEST.in",
    "requirements.txt",
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
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
]


def count_file(path: Path):
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    nonempty = [line for line in lines if line.strip()]
    return len(lines), len(nonempty)


def main():
    total_lines = 0
    total_nonempty = 0
    rows = []

    for name in FILES:
        path = Path(name)
        if not path.exists():
            rows.append((name, "MISSING", "MISSING"))
            continue

        lines, nonempty = count_file(path)
        total_lines += lines
        total_nonempty += nonempty
        rows.append((name, lines, nonempty))

    output = []
    output.append("File | Lines | Non-empty lines")
    output.append("--- | ---: | ---:")

    for name, lines, nonempty in rows:
        output.append(f"{name} | {lines} | {nonempty}")

    output.append(f"TOTAL | {total_lines} | {total_nonempty}")

    report = "\n".join(output)
    print(report)
    Path("line_count_report.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
