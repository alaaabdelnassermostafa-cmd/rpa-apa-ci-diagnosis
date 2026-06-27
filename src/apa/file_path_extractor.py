import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MentionedFile:
    path: str
    line: Optional[int] = None
    column: Optional[int] = None
    context: str = ""


PY_PATTERN = re.compile(
    r'File\s+"([^"]+\.(?:py|pyi|pyw|pyc))"(?:,\s*line\s+(\d+))?'
)
JAVA_PATTERN = re.compile(
    r'\(([A-Za-z0-9_]+\.(?:java|kt|scala)):(\d+)\)'
)
GO_PATTERN = re.compile(
    r'(\.?/?[a-zA-Z0-9_\-./]+\.go):(\d+)(?::(\d+))?'
)
JS_PATTERN = re.compile(
    r'([a-zA-Z0-9_\-./]+\.(?:js|ts|jsx|tsx|mjs|cjs)):(\d+)(?::(\d+))?'
)
RUST_PATTERN = re.compile(
    r'-->\s+([a-zA-Z0-9_\-./]+\.rs):(\d+):(\d+)'
)
C_PATTERN = re.compile(
    r'([a-zA-Z0-9_\-./]+\.(?:c|cpp|cc|cxx|h|hpp)):(\d+)(?::(\d+))?:\s*(?:error|warning|fatal)'
)
RUBY_PATTERN = re.compile(
    r'([a-zA-Z0-9_\-./]+\.rb):(\d+)'
)
TEST_PATTERN = re.compile(
    r'((?:tests?|spec)/[a-zA-Z0-9_\-./]+\.(?:py|rb|js|ts|java|go|rs|php|cs))(?::(\d+))?'
)
CONFIG_PATTERN = re.compile(
    r'(\.github/workflows/[a-zA-Z0-9_\-./]+\.(?:yml|yaml))'
)
DEP_PATTERN = re.compile(
    r'((?:package\.json|package-lock\.json|yarn\.lock|pnpm-lock\.yaml|'
    r'requirements\.txt|Pipfile|Pipfile\.lock|poetry\.lock|pyproject\.toml|'
    r'setup\.py|setup\.cfg|Cargo\.toml|Cargo\.lock|pom\.xml|build\.gradle|'
    r'build\.gradle\.kts|go\.mod|go\.sum|Gemfile|Gemfile\.lock|'
    r'composer\.json|composer\.lock|Dockerfile|docker-compose\.yml|'
    r'docker-compose\.yaml|Makefile|tox\.ini|pytest\.ini))'
)
GENERIC_PATH = re.compile(
    r'(?:^|[\s"\'(])('
    r'/?[a-zA-Z0-9_\-./]+/[a-zA-Z0-9_\-./]+\.[a-zA-Z]{1,8}'
    r')(?=$|[\s)"\':,])'
)

NOISE_PATTERNS = (
    "/usr/",
    "/bin/",
    "/lib/",
    "/etc/",
    "/opt/",
    "/tmp/",
    "/var/",
    "/home/runner/",
    "/__w/",
    "/github/",
    "/root/",
    "node_modules/",
    ".cargo/registry/",
    "/nix/store/",
    "/snap/",
)


def _is_noise(path: str) -> bool:
    lowered = path.lower()
    return any(noise in lowered for noise in NOISE_PATTERNS)


def _clean_path(path: str) -> str:
    path = path.strip().strip("'\"`()[]{}<>")
    if path.startswith("./"):
        path = path[2:]
    return path


def _store_match(
    found: dict,
    path: str,
    context: str,
    line_num: Optional[int] = None,
    col_num: Optional[int] = None,
) -> None:
    cleaned = _clean_path(path)
    if not cleaned or _is_noise(cleaned):
        return
    existing = found.get(cleaned)
    if existing is None:
        found[cleaned] = MentionedFile(
            path=cleaned,
            line=line_num,
            column=col_num,
            context=context[:180],
        )
        return
    if existing.line is None and line_num is not None:
        existing.line = line_num
    if existing.column is None and col_num is not None:
        existing.column = col_num
    if len(context) > len(existing.context):
        existing.context = context[:180]


def extract_mentioned_files(log_lines: List[str]) -> List[MentionedFile]:
    found = {}

    for raw_line in log_lines:
        line = raw_line.strip()
        if not line:
            continue

        for pattern in (
            PY_PATTERN,
            JAVA_PATTERN,
            GO_PATTERN,
            JS_PATTERN,
            RUST_PATTERN,
            C_PATTERN,
            RUBY_PATTERN,
            TEST_PATTERN,
        ):
            for match in pattern.finditer(line):
                groups = match.groups()
                path = groups[0]
                line_num = int(groups[1]) if len(groups) > 1 and groups[1] else None
                col_num = int(groups[2]) if len(groups) > 2 and groups[2] else None
                _store_match(found, path, line, line_num, col_num)

        for pattern in (CONFIG_PATTERN, DEP_PATTERN):
            for match in pattern.finditer(line):
                _store_match(found, match.group(1), line)

        lowered = line.lower()
        if any(token in lowered for token in ("error", "failed", "fatal", "exception", "traceback", "undefined", "cannot find")):
            for match in GENERIC_PATH.finditer(line):
                _store_match(found, match.group(1), line)

    return sorted(found.values(), key=lambda f: (f.line is None, f.path))


def extract_from_excerpt_windows(error_windows: List[List[str]]) -> List[MentionedFile]:
    all_lines: List[str] = []
    for window in error_windows:
        all_lines.extend(window)
    return extract_mentioned_files(all_lines)


def print_mentioned_files(files: List[MentionedFile]) -> None:
    if not files:
        print("  (no file paths found)")
        return
    for mentioned in files:
        loc = mentioned.path
        if mentioned.line is not None:
            loc += f":{mentioned.line}"
        if mentioned.column is not None:
            loc += f":{mentioned.column}"
        print(f"  {loc}")
        if mentioned.context:
            print(f"    -> {mentioned.context[:100]}")


if __name__ == "__main__":
    test_lines = [
        'File "/home/runner/work/project/src/utils/parser.py", line 42, in parse',
        '  raise ValueError("invalid input")',
        'at org.apache.maven.plugin.enforcer.RequireJavaVersion(RequireJavaVersion.java:128)',
        '/__e/node20/bin/node: /lib64/libm.so.6: version `GLIBC_2.27\' not found',
        "Error: Cannot find module './src/components/App.tsx'",
        '--> src/_bcrypt/src/lib.rs:42:5',
        'error in .github/workflows/wheel-builder.yml',
        '[ERROR] Failed to execute goal on project unomi: Could not resolve dependencies in pom.xml',
        'npm ERR! peer dep missing: react@^18.0.0, required by my-package@1.0.0',
        'src/main.go:15:3: undefined: NewClient',
        'tests/test_bcrypt.py::test_hash FAILED',
        '/usr/bin/git version',
        'node_modules/.cache/something.js',
    ]

    print("=== File Path Extractor — self-test ===\n")
    files = extract_mentioned_files(test_lines)
    print(f"Found {len(files)} file paths:\n")
    print_mentioned_files(files)
