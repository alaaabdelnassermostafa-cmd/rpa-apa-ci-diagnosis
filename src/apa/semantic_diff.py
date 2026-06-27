# semantic_diff.py
# ─────────────────────────────────────────────────────────────────────
# Semantic Diff Analysis
#
# Goes beyond "pom.xml changed" to extract the actual version bump
# ("spring-boot-starter 2.7.0 → 3.0.0") and cross-reference it with
# the library name that appears in the error log
# ("ClassNotFoundException: org/springframework/core/...").
#
# No LLM call — purely deterministic regex across 7 package ecosystems
# plus a normalised token-overlap matcher.
#
# Architecture
#   extract_version_changes(files)  → list[VersionChange]
#   link_to_errors(changes, errors) → list[LinkedEvidence]
#   analyze_semantic_diff(files, error_lines) → dict   ← public API
#
# Integration: called inside inspect_commit_diff() in agent.py after
# the raw diff is fetched, before the Bayesian update, so that linked
# evidence can enrich the observation text passed to llm_generate_likelihood.
# ─────────────────────────────────────────────────────────────────────

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ─── data structures ─────────────────────────────────────────────────

@dataclass
class VersionChange:
    library: str        # normalised package/library name
    old_version: str    # "" if only an addition was seen
    new_version: str    # "" if only a deletion was seen
    file: str           # source filename
    ecosystem: str      # maven | gradle | npm | pip | cargo | go | toml


@dataclass
class LinkedEvidence:
    library: str
    old_version: str
    new_version: str
    file: str
    ecosystem: str
    matching_error_lines: list[str]
    match_strength: float       # 0.0–1.0
    match_tokens: list[str]     # tokens that triggered the match

    def summary(self) -> str:
        arrow = f"{self.old_version} → {self.new_version}" if self.old_version and self.new_version else (self.new_version or self.old_version)
        return f"{self.library} ({arrow}) [{self.ecosystem}] — strength={self.match_strength:.2f}"


# ─── shared helpers ──────────────────────────────────────────────────

_SEMVER = r'[\d]+\.[\d]+(?:\.[\d]+)?(?:[._-][A-Za-z0-9.]+)*'


def _iter_diff_lines(patch: str):
    """Yield (sign, content) for each meaningful diff line.
    sign ∈ {'+', '-', ' '}  (hunk headers and file headers are skipped).
    """
    for line in patch.splitlines():
        if not line:
            continue
        if line.startswith(('@@', '---', '+++')):
            continue
        if line[0] in ('+', '-', ' '):
            yield line[0], line[1:]


def _norm(name: str) -> str:
    """Lowercase + collapse punctuation for fuzzy matching."""
    return re.sub(r'[-_./]+', '-', name.lower()).strip('-')


# ─── per-ecosystem parsers ────────────────────────────────────────────

def _parse_pip(patch: str, filename: str) -> list[VersionChange]:
    """requirements.txt: 'package==1.2.3' or 'package>=1.0.0'"""
    seen: dict[str, dict] = {}
    for sign, content in _iter_diff_lines(patch):
        content = content.strip()
        m = re.match(
            r'^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[.*?\])?'
            r'(?:\s*[=<>!~^]+\s*)(' + _SEMVER + r')',
            content,
        )
        if not m:
            continue
        name = _norm(m.group(1))
        ver = m.group(2)
        seen.setdefault(name, {'old': '', 'new': ''})
        if sign == '-':
            seen[name]['old'] = ver
        else:
            seen[name]['new'] = ver
    return [
        VersionChange(library=n, old_version=v['old'], new_version=v['new'],
                      file=filename, ecosystem='pip')
        for n, v in seen.items() if v['old'] != v['new']
    ]


def _parse_npm(patch: str, filename: str) -> list[VersionChange]:
    """package.json: '"name": "^1.2.3"'"""
    seen: dict[str, dict] = {}
    for sign, content in _iter_diff_lines(patch):
        content = content.strip()
        m = re.match(
            r'"(@?[A-Za-z0-9][A-Za-z0-9._/@-]*)"\s*:\s*"[~^>=<v]?(' + _SEMVER + r')',
            content,
        )
        if not m:
            continue
        name = m.group(1)
        ver = m.group(2)
        seen.setdefault(name, {'old': '', 'new': ''})
        if sign == '-':
            seen[name]['old'] = ver
        else:
            seen[name]['new'] = ver
    return [
        VersionChange(library=n, old_version=v['old'], new_version=v['new'],
                      file=filename, ecosystem='npm')
        for n, v in seen.items() if v['old'] != v['new']
    ]


def _parse_cargo(patch: str, filename: str) -> list[VersionChange]:
    """Cargo.toml: 'serde = "1.0"' or 'serde = { version = "1.0" }'"""
    seen: dict[str, dict] = {}
    for sign, content in _iter_diff_lines(patch):
        content = content.strip()
        # Simple form: name = "1.2.3"
        m = re.match(r'^([a-z][a-z0-9_-]*)\s*=\s*"(' + _SEMVER + r')"', content)
        if not m:
            # Inline table: name = { version = "1.2.3", ... }
            m = re.match(
                r'^([a-z][a-z0-9_-]*)\s*=\s*\{[^}]*version\s*=\s*"(' + _SEMVER + r')"',
                content,
            )
        if not m:
            continue
        name = m.group(1)
        ver = m.group(2)
        seen.setdefault(name, {'old': '', 'new': ''})
        if sign == '-':
            seen[name]['old'] = ver
        else:
            seen[name]['new'] = ver
    return [
        VersionChange(library=n, old_version=v['old'], new_version=v['new'],
                      file=filename, ecosystem='cargo')
        for n, v in seen.items() if v['old'] != v['new']
    ]


def _parse_gomod(patch: str, filename: str) -> list[VersionChange]:
    """go.mod: 'github.com/user/repo v1.2.3'"""
    seen: dict[str, dict] = {}
    for sign, content in _iter_diff_lines(patch):
        content = content.strip()
        m = re.match(r'^([\w./-]+)\s+(v[\d][^\s]*)', content)
        if not m:
            continue
        name = m.group(1)
        ver = m.group(2)
        seen.setdefault(name, {'old': '', 'new': ''})
        if sign == '-':
            seen[name]['old'] = ver
        else:
            seen[name]['new'] = ver
    return [
        VersionChange(library=n, old_version=v['old'], new_version=v['new'],
                      file=filename, ecosystem='go')
        for n, v in seen.items() if v['old'] != v['new']
    ]


def _parse_gradle(patch: str, filename: str) -> list[VersionChange]:
    """build.gradle: 'implementation "group:artifact:1.2.3"'"""
    seen: dict[str, dict] = {}
    _dep_kw = r'(?:implementation|api|compile|testImplementation|runtimeOnly|compileOnly|annotationProcessor)'
    for sign, content in _iter_diff_lines(patch):
        content = content.strip()
        m = re.search(_dep_kw + r"""\s*[\('"]([^'")\s]+)['")]""", content)
        if not m:
            continue
        dep = m.group(1)
        parts = dep.split(':')
        if len(parts) >= 3 and re.match(r'[\d]', parts[-1]):
            name = ':'.join(parts[:-1])
            ver = parts[-1].strip()
            seen.setdefault(name, {'old': '', 'new': ''})
            if sign == '-':
                seen[name]['old'] = ver
            else:
                seen[name]['new'] = ver
    return [
        VersionChange(library=n, old_version=v['old'], new_version=v['new'],
                      file=filename, ecosystem='gradle')
        for n, v in seen.items() if v['old'] != v['new']
    ]


def _parse_maven(patch: str, filename: str) -> list[VersionChange]:
    """
    pom.xml: tricky because <artifactId> and <version> are on different lines.

    Strategy: scan the patch lines in order, keeping a small rolling window
    of recently-seen XML tag values (groupId, artifactId).  When we see a
    version change on a +/- line, attribute it to the nearest preceding
    artifactId.  Context lines (' ') update the window; hunk resets clear it.
    """
    lines = patch.splitlines()
    # Track last seen groupId / artifactId from context + changed lines
    last_group = ''
    last_artifact = ''
    seen: dict[str, dict] = {}

    def _tag_val(text: str, tag: str) -> Optional[str]:
        m = re.search(fr'<{tag}>([^<]+)</{tag}>', text)
        return m.group(1).strip() if m else None

    for line in lines:
        if not line or line.startswith(('@@', '---', '+++')):
            last_group = ''
            last_artifact = ''
            continue

        sign = line[0] if line[0] in ('+', '-', ' ') else ' '
        content = line[1:] if sign in ('+', '-') else line

        g = _tag_val(content, 'groupId')
        if g:
            last_group = g
        a = _tag_val(content, 'artifactId')
        if a:
            last_artifact = a

        ver = _tag_val(content, 'version')
        if ver and re.match(r'[\d]', ver) and sign in ('+', '-'):
            # Build canonical key
            key = f"{last_group}:{last_artifact}" if last_group or last_artifact else f"(unknown):{ver}"
            name = key if key else ver
            seen.setdefault(name, {'old': '', 'new': ''})
            if sign == '-':
                seen[name]['old'] = ver
            else:
                seen[name]['new'] = ver

    return [
        VersionChange(library=n, old_version=v['old'], new_version=v['new'],
                      file=filename, ecosystem='maven')
        for n, v in seen.items() if v['old'] != v['new']
    ]


def _parse_pyproject(patch: str, filename: str) -> list[VersionChange]:
    """pyproject.toml (Poetry style): 'requests = "^2.28"'"""
    seen: dict[str, dict] = {}
    for sign, content in _iter_diff_lines(patch):
        content = content.strip()
        m = re.match(
            r'^([A-Za-z0-9][A-Za-z0-9._-]*)\s*=\s*"[~^>=<]?(' + _SEMVER + r')',
            content,
        )
        if not m:
            continue
        name = _norm(m.group(1))
        ver = m.group(2)
        seen.setdefault(name, {'old': '', 'new': ''})
        if sign == '-':
            seen[name]['old'] = ver
        else:
            seen[name]['new'] = ver
    return [
        VersionChange(library=n, old_version=v['old'], new_version=v['new'],
                      file=filename, ecosystem='toml')
        for n, v in seen.items() if v['old'] != v['new']
    ]


# ─── dispatcher ──────────────────────────────────────────────────────

_ECOSYSTEM_MAP = [
    (['pom.xml'],                                       _parse_maven),
    (['build.gradle', 'build.gradle.kts'],              _parse_gradle),
    (['package.json'],                                  _parse_npm),
    (['requirements.txt', 'requirements-dev.txt',
      'requirements-test.txt', 'requirements-ci.txt'],  _parse_pip),
    (['pyproject.toml'],                                _parse_pyproject),
    (['cargo.toml'],                                    _parse_cargo),
    (['go.mod'],                                        _parse_gomod),
]


def extract_version_changes(files: list[dict]) -> list[VersionChange]:
    """
    Extract all version changes from a list of diff file dicts
    (as returned by _fetch_commit_diff in agent.py).

    Each dict is expected to have 'filename' and 'patch_excerpt' keys.
    """
    results: list[VersionChange] = []
    for f in files:
        fname = f.get('filename', '')
        patch = f.get('patch_excerpt', '') or ''
        if not patch:
            continue
        base = Path(fname).name.lower()
        for basenames, parser in _ECOSYSTEM_MAP:
            if base in basenames:
                results.extend(parser(patch, fname))
                break
    return results


# ─── error-log tokeniser ─────────────────────────────────────────────

# Patterns that look like library or package references in error text.
_ERROR_LIB_PATTERNS = [
    # Python: "No module named 'requests'" / "from requests import"
    re.compile(r"(?:module named|from|import)\s+'?([A-Za-z][A-Za-z0-9._-]+)'?", re.I),
    # Node: "Cannot find module 'express'"
    re.compile(r"cannot find module\s+'?([A-Za-z@][A-Za-z0-9._/@-]+)'?", re.I),
    # Java: "ClassNotFoundException: org.springframework.core.Foo"
    re.compile(r'(?:ClassNotFoundException|NoClassDefFoundError|NoSuchMethodError):\s+'
               r'([A-Za-z][A-Za-z0-9./]+)', re.I),
    # Rust: "use of unresolved item 'serde'"
    re.compile(r"(?:unresolved import|unresolved module|use of unresolved)\s+'?([a-z][a-z0-9_]+)'?", re.I),
    # Go: "cannot find package 'github.com/user/lib'"
    re.compile(r"cannot find (?:package|module)\s+['\"]?([A-Za-z][A-Za-z0-9._/-]+)['\"]?", re.I),
]


def _tokenise_library_name(name: str) -> set[str]:
    """
    Break a library name into matchable tokens.
    'org.springframework.boot:spring-boot-starter' →
    {'org', 'springframework', 'spring', 'boot', 'starter'}
    """
    tokens = set()
    # Split on common separators
    for part in re.split(r'[.:/_-]', name):
        part = part.lower().strip()
        if len(part) >= 3:
            tokens.add(part)
    return tokens


# ─── cross-reference linker ──────────────────────────────────────────

def link_to_errors(
    changes: list[VersionChange],
    error_lines: list[str],
) -> list[LinkedEvidence]:
    """
    For each VersionChange, find error lines that mention the same library.

    Matching tiers (highest wins):
      1.0  — exact normalised library name appears in the error line
      0.75 — library name appears as a substring of an error token (path/class)
      0.55 — ≥2 significant tokens of the library name overlap with error tokens
      0.35 — 1 significant token overlaps (weak, included for traceability)

    Returns only matches with strength ≥ 0.35, sorted by strength desc.
    """
    evidence: list[LinkedEvidence] = []

    for change in changes:
        lib_norm = _norm(change.library)
        lib_tokens = _tokenise_library_name(change.library)
        # Drop very short / generic tokens that match almost everything
        sig_tokens = {t for t in lib_tokens if len(t) >= 4 and t not in
                      {'java', 'core', 'main', 'util', 'test', 'impl', 'base', 'common'}}

        matching: list[str] = []
        max_strength = 0.0
        best_tokens: list[str] = []

        for eline in error_lines:
            el_norm = _norm(eline)
            el_tokens = _tokenise_library_name(eline)

            strength = 0.0
            matched_tokens: list[str] = []

            # Tier 1: exact
            if lib_norm in el_norm:
                strength = 1.0
                matched_tokens = [lib_norm]

            # Tier 2: substring inside a class path / import path
            elif any(lib_norm in t for t in re.split(r'[\s,;\'"]', eline.lower())):
                strength = 0.75
                matched_tokens = [lib_norm]

            # Tier 3+: token overlap
            else:
                overlap = sig_tokens & el_tokens
                if len(overlap) >= 2:
                    strength = 0.55
                    matched_tokens = sorted(overlap)
                elif len(overlap) == 1:
                    strength = 0.35
                    matched_tokens = sorted(overlap)

            if strength > 0:
                matching.append(eline)
                if strength > max_strength:
                    max_strength = strength
                    best_tokens = matched_tokens

        if matching:
            evidence.append(LinkedEvidence(
                library=change.library,
                old_version=change.old_version,
                new_version=change.new_version,
                file=change.file,
                ecosystem=change.ecosystem,
                matching_error_lines=matching[:4],
                match_strength=max_strength,
                match_tokens=best_tokens,
            ))

    evidence.sort(key=lambda e: -e.match_strength)
    return evidence


# ─── public API ──────────────────────────────────────────────────────

def analyze_semantic_diff(
    files: list[dict],
    error_lines: list[str],
) -> dict:
    """
    Main entry point.  Returns a dict with:
        version_changes   list[dict]   — all detected version bumps
        linked_evidence   list[dict]   — changes cross-referenced to error log
        observation_text  str          — human-readable summary for LLM prompt
        has_links         bool         — True if any evidence was linked
    """
    changes = extract_version_changes(files)
    linked = link_to_errors(changes, error_lines) if error_lines else []

    def _change_dict(c: VersionChange) -> dict:
        return {
            'library': c.library,
            'old_version': c.old_version,
            'new_version': c.new_version,
            'file': c.file,
            'ecosystem': c.ecosystem,
        }

    def _link_dict(e: LinkedEvidence) -> dict:
        return {
            'library': e.library,
            'old_version': e.old_version,
            'new_version': e.new_version,
            'file': e.file,
            'ecosystem': e.ecosystem,
            'match_strength': round(e.match_strength, 2),
            'match_tokens': e.match_tokens,
            'matching_error_lines': e.matching_error_lines,
            'summary': e.summary(),
        }

    # Build a concise observation text for the Bayesian LLM call
    obs_parts: list[str] = []

    if linked:
        obs_parts.append("SEMANTIC DIFF LINKS (version change ↔ error log):")
        for e in linked[:5]:
            arrow = f"{e.old_version} → {e.new_version}" if e.old_version and e.new_version else (e.new_version or e.old_version)
            obs_parts.append(
                f"  {e.library} bumped {arrow} in {Path(e.file).name} [{e.ecosystem}]"
                f"  (match_strength={e.match_strength:.2f})"
            )
            for el in e.matching_error_lines[:2]:
                obs_parts.append(f"    ↳ error: {el[:120]}")
    elif changes:
        obs_parts.append("VERSION CHANGES IN COMMIT (no direct error-log link found):")
        for c in changes[:6]:
            arrow = f"{c.old_version} → {c.new_version}" if c.old_version and c.new_version else (c.new_version or c.old_version)
            obs_parts.append(f"  {c.library} {arrow} in {Path(c.file).name} [{c.ecosystem}]")

    return {
        'version_changes': [_change_dict(c) for c in changes],
        'linked_evidence': [_link_dict(e) for e in linked],
        'observation_text': '\n'.join(obs_parts),
        'has_links': bool(linked),
    }


# ─── self-test ───────────────────────────────────────────────────────

if __name__ == '__main__':
    files = [
        {
            'filename': 'requirements.txt',
            'patch_excerpt': (
                '@@ -1,3 +1,3 @@\n'
                ' flask==2.2.5\n'
                '-requests==2.28.0\n'
                '+requests==2.31.0\n'
                ' boto3==1.26.0\n'
            ),
        },
        {
            'filename': 'pom.xml',
            'patch_excerpt': (
                '@@ -10,7 +10,7 @@\n'
                '  <dependency>\n'
                '    <groupId>org.springframework</groupId>\n'
                '    <artifactId>spring-core</artifactId>\n'
                '-   <version>5.3.21</version>\n'
                '+   <version>6.0.0</version>\n'
                '  </dependency>\n'
            ),
        },
    ]
    error_lines = [
        "ImportError: cannot import name 'Session' from 'requests' (2.28.0)",
        "java.lang.NoClassDefFoundError: org/springframework/core/io/Resource",
    ]

    result = analyze_semantic_diff(files, error_lines)
    print(f"Version changes found: {len(result['version_changes'])}")
    for c in result['version_changes']:
        print(f"  {c['library']}  {c['old_version']} -> {c['new_version']}  [{c['ecosystem']}]")
    print(f"\nLinked evidence: {len(result['linked_evidence'])}")
    for e in result['linked_evidence']:
        print(f"  {e['summary']}")
    print(f"\nObservation text:\n{result['observation_text']}")
