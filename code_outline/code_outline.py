#!/usr/bin/env python3
from __future__ import annotations

"""
Repo Analyzer — SECTIONED v5
- Directories as H2, files as H3 (distinct subsections)
- Ordering: README/code first, docs next, config/build last
- Always rescan (no change detection)
- Sanitizes model output (no headings/rules; strips self-titles like "App.config Summary")
- KeyError-safe when root has no files
- Repository Packages:
    • Node: recurse and list every package.json with deps
    • Python: recurse and list every requirements.txt / pyproject.toml with deps
    • Go: recurse and list every go.mod with modules
    • .NET: per-csproj TargetFramework/ProjectReference/Framework References/NuGet (PackageReference & packages.config)
    • WCF endpoints parsed from App.config/Web.config (client & service)
  Only languages present are shown.
"""

import argparse
import ast
import datetime as _dt
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import xml.etree.ElementTree as ET

import httpx
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# ================== CONFIG (loaded at runtime) ==================
# Defaults (can be overridden by config.json or environment)
PROVIDER: str = "azure"          # "openai" or "azure"
MODEL: str = "gpt-4o"

# --- OpenAI (public API) ---
OPENAI_API_KEY: Optional[str] = None
OPENAI_BASE_URL: str = "https://api.openai.com/v1"
OPENAI_ORG: Optional[str] = None

# --- Azure OpenAI ---
AZURE_API_KEY: Optional[str] = None
AZURE_ENDPOINT: Optional[str] = None
AZURE_DEPLOYMENT: Optional[str] = None
AZURE_API_VERSION: str = "2024-12-01-preview"

def load_config(cfg_path: Optional[Path]) -> None:
    """
    Load settings from JSON file (if provided) and environment variables.
    Order of precedence: explicit cfg file > env vars > existing defaults.
    """
    global PROVIDER, MODEL
    global OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_ORG
    global AZURE_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT, AZURE_API_VERSION

    # 1) File (explicit path or default ./config.json next to this script)
    config = {}
    if cfg_path is None:
        default_path = Path(__file__).parent / "config.json"
        if default_path.exists():
            cfg_path = default_path
    if cfg_path is not None:
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    # 2) Merge file values over current defaults
    PROVIDER = str(config.get("provider", PROVIDER)).lower()
    MODEL = config.get("model", MODEL)

    OPENAI_API_KEY = config.get("openai_api_key", OPENAI_API_KEY)
    OPENAI_BASE_URL = config.get("openai_base_url", OPENAI_BASE_URL)
    OPENAI_ORG = config.get("openai_org", OPENAI_ORG)

    AZURE_API_KEY = config.get("azure_api_key", AZURE_API_KEY)
    AZURE_ENDPOINT = config.get("azure_endpoint", AZURE_ENDPOINT)
    AZURE_DEPLOYMENT = config.get("azure_deployment", AZURE_DEPLOYMENT)
    AZURE_API_VERSION = config.get("azure_api_version", AZURE_API_VERSION)

    # 3) Env vars (optional, override everything if present)
    PROVIDER = os.getenv("PROVIDER", PROVIDER).lower()
    MODEL = os.getenv("MODEL", MODEL)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", OPENAI_BASE_URL)
    OPENAI_ORG = os.getenv("OPENAI_ORG", OPENAI_ORG)
    AZURE_API_KEY = os.getenv("AZURE_API_KEY", AZURE_API_KEY)
    AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", AZURE_ENDPOINT)
    AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", AZURE_DEPLOYMENT)
    AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", AZURE_API_VERSION)


# ---------- CONFIG ----------
IGNORE_DIRS = {
    ".git","node_modules",".pnpm",".yarn",".venv","venv","dist","build","out","coverage",
    ".idea",".vscode",".vs","bin","obj","target","__pycache__",".terraform",".next",".svelte-kit",".gradle"
}
TEXT_EXTS = {
    ".js",".jsx",".ts",".tsx",".mjs",".cjs",".json",".md",".mdx",".yml",".yaml",".toml",
    ".py",".cs",".go",".rs",".java",".kt",".rb",".php",".sql",".html",".css",".scss",
    ".less",".sh",".ps1",".bat",".cmd",".sln",".csproj",".fsproj",".vbproj",".proto",
    ".graphql",".gql",".env",".config",".ini",".props",".targets","Makefile"
}
DOC_FILE   = "PROJECT_OVERVIEW.md"

# ---------- DATA ----------
@dataclass
class FileInfo:
    path: str    # relative to repo root
    content: str
    lang: str

# ---------- LLM ----------
class LLM:
    def __init__(self, provider: str, model: str):
        self.provider = provider.lower()
        self.model = model
        self.timeout = 120

        if self.provider == "openai":
            if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-PASTE"):
                raise RuntimeError("Set OPENAI_API_KEY at the top of the script")
            self.oai_key  = OPENAI_API_KEY
            self.oai_base = OPENAI_BASE_URL
            self.oai_org  = OPENAI_ORG

        elif self.provider == "azure":
            if not (AZURE_API_KEY and AZURE_ENDPOINT and AZURE_DEPLOYMENT):
                raise RuntimeError("Set AZURE_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT at the top of the script")
            self.az_key        = AZURE_API_KEY
            self.az_endpoint   = AZURE_ENDPOINT.rstrip("/")
            self.az_deployment = AZURE_DEPLOYMENT
            self.az_version    = AZURE_API_VERSION
        else:
            raise RuntimeError(f"Unknown provider: {provider}")

    def chat(self, system: str, messages: List[Dict[str,str]], max_tokens: Optional[int] = None, temperature: float = 0.2) -> str:
        if self.provider == "openai":
            headers = {"Authorization": f"Bearer {self.oai_key}"}
            if self.oai_org:
                headers["OpenAI-Organization"] = self.oai_org
            payload = {
                "model": self.model,
                "messages": [{"role": "system", "content": system}] + messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            r = httpx.post(f"{self.oai_base}/chat/completions", json=payload, headers=headers, timeout=self.timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

        if self.provider == "azure":
            headers = {"api-key": self.az_key}
            payload = {
                "messages": [{"role": "system", "content": system}] + messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            url = f"{self.az_endpoint}/openai/deployments/{self.az_deployment}/chat/completions?api-version={self.az_version}"
            r = httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

        raise RuntimeError("unreachable")

def build_llm() -> LLM:
    return LLM(PROVIDER, MODEL)

# ---------- FILE SCAN ----------
def is_text_file(p: Path) -> bool:
    try:
        return b"\0" not in p.read_bytes()[:2048]
    except Exception:
        return False

def detect_lang(p: Path) -> str:
    ext = p.suffix.lower()
    MAP = {
        ".js":"JavaScript",".jsx":"JavaScript JSX",".ts":"TypeScript",".tsx":"TypeScript JSX",
        ".py":"Python",".cs":"C#",".go":"Go",".rs":"Rust",".java":"Java",".kt":"Kotlin",".rb":"Ruby",
        ".php":"PHP",".html":"HTML",".css":"CSS",".scss":"SCSS",".json":"JSON",".yml":"YAML",".yaml":"YAML",
        ".md":"Markdown",".proto":"Proto",".graphql":"GraphQL",".gql":"GraphQL",".sln":"DotNet Solution",
        ".csproj":"DotNet Project"
    }
    return MAP.get(ext, ext.upper().lstrip("."))

def read_full(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def walk_repo(root: Path) -> List[FileInfo]:
    files: List[FileInfo] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() not in TEXT_EXTS and not is_text_file(p):
                continue
            files.append(FileInfo(
                path=str(p.relative_to(root)),
                content=read_full(p),
                lang=detect_lang(p)
            ))
    return files

# ---------- IMPORT / PACKAGE EXTRACTION ----------
JS_IMPORT_RE = re.compile(r"import\s+[^;]*?from\s+['\"]([^'\"]+)['\"]|require\(\s*['\"]([^'\"]+)['\"]\s*\)")
PY_IMPORT_FROM_RE = re.compile(r"^\s*from\s+([\w\.]+)\s+import\s+", re.M)
PY_IMPORT_RE      = re.compile(r"^\s*import\s+([\w\.]+)", re.M)
CS_USING_RE       = re.compile(r"^\s*(?:global\s+)?using\s+([\w\.]+)\s*;", re.M)

def extract_imports_for_file(path: str, lang: str, content: str) -> List[str]:
    lang_l = (lang or "").lower()
    imports: Set[str] = set()
    if path.endswith((".js",".jsx",".ts",".tsx",".mjs",".cjs")) or "javascript" in lang_l or "typescript" in lang_l:
        for m in JS_IMPORT_RE.finditer(content):
            pkg = m.group(1) or m.group(2)
            if pkg and not pkg.startswith((".", "/")):
                imports.add(pkg)
    elif path.endswith(".py") or "python" in lang_l:
        imports.update(PY_IMPORT_FROM_RE.findall(content))
        imports.update(PY_IMPORT_RE.findall(content))
        imports = {i.split(".")[0] for i in imports if i}
    elif path.endswith(".cs") or "c#" in lang_l:
        imports.update(CS_USING_RE.findall(content))
    else:
        for m in re.finditer(r"\bimport\s+([\w\-\.\_]+)", content):
            imports.add(m.group(1))
    return sorted(imports)

# ---------- RECURSIVE MANIFEST SCANS ----------
def collect_node_packages_recursive(root: Path) -> Dict[str, Dict[str,str]]:
    out: Dict[str, Dict[str,str]] = {}
    for pkg in root.rglob("package.json"):
        if any(part in IGNORE_DIRS or part == "node_modules" for part in pkg.parts):
            continue
        try:
            data = json.loads(pkg.read_text(encoding="utf-8"))
            deps = {}
            for k, v in (data.get("dependencies") or {}).items():
                deps[k] = str(v)
            for k, v in (data.get("devDependencies") or {}).items():
                deps.setdefault(k, str(v))
            if deps:
                out[str(pkg.relative_to(root))] = deps
            else:
                out.setdefault(str(pkg.relative_to(root)), {})
        except Exception:
            continue
    return out

def collect_python_packages_recursive(root: Path) -> Dict[str, Dict[str,str]]:
    out: Dict[str, Dict[str,str]] = {}
    for req in root.rglob("requirements.txt"):
        if any(part in IGNORE_DIRS for part in req.parts):
            continue
        deps: Dict[str,str] = {}
        try:
            for line in req.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"): 
                    continue
                deps[line] = ""
            out[str(req.relative_to(root))] = deps
        except Exception:
            continue
    for pyproj in root.rglob("pyproject.toml"):
        if any(part in IGNORE_DIRS for part in pyproj.parts):
            continue
        deps: Dict[str,str] = {}
        try:
            txt = pyproj.read_text(encoding="utf-8")
            m = re.search(r"(?ms)^\s*\[project\]\s.*?dependencies\s*=\s*\[(.*?)\]", txt)
            block = m.group(1) if m else None
            if not block:
                # fallback to generic search
                m2 = re.search(r"dependencies\s*=\s*\[(.*?)\]", txt, re.S)
                block = m2.group(1) if m2 else None
            if block:
                for d in re.findall(r"\"([^\"]+)\"|'([^']+)'", block):
                    name = (d[0] or d[1]).strip()
                    if name:
                        deps.setdefault(name, "")
            out[str(pyproj.relative_to(root))] = deps
        except Exception:
            continue
    return out

def collect_go_packages_recursive(root: Path) -> Dict[str, Dict[str,str]]:
    out: Dict[str, Dict[str,str]] = {}
    for gomod in root.rglob("go.mod"):
        if any(part in IGNORE_DIRS for part in gomod.parts):
            continue
        mods: Dict[str,str] = {}
        try:
            txt = gomod.read_text(encoding="utf-8")
            # single-line requires
            for line in txt.splitlines():
                line = line.strip()
                if line.startswith("require ") and not line.startswith("require ("):
                    parts = line.split()
                    if len(parts) >= 3:
                        mods[parts[1]] = parts[2]
            # block requires
            block = re.search(r"require\s*\((.*?)\)", txt, re.S)
            if block:
                for line in block.group(1).splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        mods[parts[0]] = parts[1]
            out[str(gomod.relative_to(root))] = mods
        except Exception:
            continue
    return out

def _et_localname(tag: str) -> str:
    return tag.split('}', 1)[-1] if '}' in tag else tag

def collect_dotnet_details_recursive(root: Path) -> Dict[str, dict]:
    details: Dict[str, dict] = {"projects": {}, "old_packages": {}, "endpoints": []}

    # csproj parsing
    for csproj in root.rglob("*.csproj"):
        if any(part in IGNORE_DIRS for part in csproj.parts):
            continue
        info = {"target_framework": None, "project_refs": [], "framework_refs": [], "packages": {}}
        try:
            tree = ET.parse(csproj)
            root_el = tree.getroot()
            # TargetFramework / TargetFrameworkVersion
            tf = None
            for el in root_el.iter():
                name = _et_localname(el.tag).lower()
                if name in ("targetframework", "targetframeworks", "targetframeworkversion"):
                    tf = (tf or "") + (el.text or "")
            info["target_framework"] = (tf or "").strip() or None
            # ProjectReference
            for pr in root_el.iter():
                if _et_localname(pr.tag).lower() == "projectreference":
                    inc = pr.attrib.get("Include")
                    if inc:
                        try:
                            rel = str((csproj.parent / inc).resolve().relative_to(root.resolve()))
                        except Exception:
                            rel = inc
                        info["project_refs"].append(rel)
            # Framework References
            for rf in root_el.iter():
                if _et_localname(rf.tag).lower() == "reference":
                    inc = rf.attrib.get("Include")
                    if inc:
                        info["framework_refs"].append(inc.split(",")[0])
            # PackageReference (NuGet)
            for pr in root_el.iter():
                if _et_localname(pr.tag).lower() == "packagereference":
                    name = pr.attrib.get("Include") or pr.attrib.get("Update")
                    ver  = pr.attrib.get("Version") or (pr.findtext(pr.tag.replace("PackageReference","Version")) if hasattr(pr, "findtext") else "")
                    if name:
                        info["packages"][name] = ver or ""
            details["projects"][str(csproj.relative_to(root))] = info
        except Exception:
            continue

    # packages.config (old NuGet)
    for pkgcfg in root.rglob("packages.config"):
        if any(part in IGNORE_DIRS for part in pkgcfg.parts):
            continue
        pkgs: Dict[str,str] = {}
        try:
            tree = ET.parse(pkgcfg)
            for p in tree.getroot().iter():
                if _et_localname(p.tag).lower().endswith("package"):
                    name = p.attrib.get("id")
                    ver  = p.attrib.get("version") or ""
                    if name:
                        pkgs[name] = ver
            details["old_packages"][str(pkgcfg.relative_to(root))] = pkgs
        except Exception:
            continue

    # App/Web.config endpoints
    for cfg in list(root.rglob("App.config")) + list(root.rglob("Web.config")):
        if any(part in IGNORE_DIRS for part in cfg.parts):
            continue
        try:
            tree = ET.parse(cfg)
            root_el = tree.getroot()
            # traverse regardless of namespaces
            for el in root_el.iter():
                if _et_localname(el.tag).lower() == "endpoint":
                    parent = el.getparent() if hasattr(el, "getparent") else None
                    # Without lxml, ElementTree lacks getparent; infer by searching upward.
                    kind = "unknown"
                    # crude: check ancestors by string search in the file path string
                    # Better: walk for known containers
            # Manual walk for client/service
            # client endpoints
            for client in root_el.iter():
                if _et_localname(client.tag).lower() == "client":
                    for ep in client:
                        if _et_localname(ep.tag).lower() == "endpoint":
                            details["endpoints"].append({
                                "config": str(cfg.relative_to(root)),
                                "kind": "client",
                                "address": ep.attrib.get("address",""),
                                "binding": ep.attrib.get("binding",""),
                                "contract": ep.attrib.get("contract",""),
                                "name": ep.attrib.get("name",""),
                            })
                if _et_localname(client.tag).lower() == "services":
                    for service in client:
                        if _et_localname(service.tag).lower() == "service":
                            for ep in service:
                                if _et_localname(ep.tag).lower() == "endpoint":
                                    details["endpoints"].append({
                                        "config": str(cfg.relative_to(root)),
                                        "kind": "service",
                                        "address": ep.attrib.get("address",""),
                                        "binding": ep.attrib.get("binding",""),
                                        "contract": ep.attrib.get("contract",""),
                                        "name": ep.attrib.get("name",""),
                                    })
        except Exception:
            continue

    return details

# ---------- PROMPTS & SANITIZATION ----------
SYSTEM_PROMPT = (
    "You are a senior engineer updating technical documentation for a multi-service repo. "
    "Be concise, accurate, and structured."
)

FILE_SUMMARY_PROMPT = (
    """Produce a Markdown summary for this file (body content only; do NOT include markdown headings like #, ##, ###, and do NOT include horizontal rules '---').

**Summary**
• 2–4 bullets on what the file does and how it fits in the repo.

**Functions & Classes Explained**
Provide a Markdown table with columns: Name | What it does | Inputs | Outputs | Notes (side effects, errors, I/O, concurrency).
Only include items that exist in the source.

**Design & Dependencies**
Bullet list calling out key libraries/services and why they’re used.

Context:
Path: {path}
Language: {lang}

Imports detected (non-relative):
{imports_md}

Function signatures detected:
(derive from the source code above; do not invent)

Full source:
```{lang}
{content}
```
"""
)

OVERVIEW_PROMPT = (
    """Write a high-level overview of this repository (body content only; do NOT include markdown headings like #, ##, ###, and do NOT include horizontal rules '---').
Explain the main components/services, how they interact, and the primary data/control flows. Focus on what a new engineer needs to know to be productive.

Here are per-file summaries (JSON mapping path -> summary):
{file_summaries_json}
"""
)

_HEADING_OR_RULE = re.compile(r"^\s*(#{1,6}\s+|---\s*$)", re.M)

def _looks_like_self_title(line: str, path: Optional[str]) -> bool:
    if not path or not line: 
        return False
    base = Path(path).name
    stem = Path(path).stem
    def norm(s: str) -> str:
        return re.sub(r"[\s:_\-]+", "", s.strip().lower())
    l = norm(line)
    candidates = {norm(base), norm(stem), norm(base + " summary"), norm(stem + " summary")}
    return l in candidates

def sanitize_body(md: str, path: Optional[str] = None) -> str:
    md = _HEADING_OR_RULE.sub("", md)
    lines = [ln.rstrip() for ln in md.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and _looks_like_self_title(lines[0], path):
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
    md = "\n".join(lines)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()

# ---------- ORDERING HELPERS ----------
CODE_EXTS = {'.cs','.ts','.tsx','.js','.jsx','.py','.go','.rs','.java','.kt','.rb','.php'}
DOC_EXTS  = {'.md','.mdx'}
CONFIG_EXTS = {'.config','.ini','.toml','.yaml','.yml','.props','.targets','.csproj','.sln','.json'}
SPECIAL_CONFIG_NAMES = {
    'app.config','appsettings.json','appsettings.development.json','packages.config','dockerfile','makefile',
    'package.json','pyproject.toml','requirements.txt','go.mod'
}

def file_priority(path: str) -> tuple[int, str]:
    p = Path(path)
    name = p.name.lower()
    ext  = p.suffix.lower()
    if name.startswith('readme'):
        return (0, name)
    if ext in CODE_EXTS:
        return (1, name)
    if ext in DOC_EXTS:
        return (2, name)
    if name in {'app.config', 'appsettings.json', 'appsettings.development.json'}:
        return (11, name)
    if ext in CONFIG_EXTS or name in SPECIAL_CONFIG_NAMES or ext == '':
        return (10, name)
    return (5, name)

# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze a repository and generate PROJECT_OVERVIEW.md")
    ap.add_argument("repo", help="Path to repository root")
    ap.add_argument("--verbose", "-v", action="store_true", help="Print extra logs")
    ap.add_argument("--dry-run", action="store_true", help="Preview only; do not write the markdown file")
    ap.add_argument("--overview", action="store_true", help="Include the High-level Overview section")
    ap.add_argument("--config", help="Path to config.json (if omitted, looks for ./config.json next to this script)")
    return ap.parse_args()

# ---------- MAIN ----------
def main():
    args = parse_args()
    root = Path(args.repo).resolve()
    load_config(Path(args.config) if args.config else None)
    llm = build_llm()

    files = walk_repo(root)

    # Recursive package scans
    node_pkgs = collect_node_packages_recursive(root)
    py_pkgs   = collect_python_packages_recursive(root)
    go_pkgs   = collect_go_packages_recursive(root)
    net_info  = collect_dotnet_details_recursive(root)

    # Determine which languages are present
    have_node = any(node_pkgs.values()) or len(node_pkgs) > 0
    have_py   = any(py_pkgs.values())   or len(py_pkgs) > 0
    have_go   = any(go_pkgs.values())   or len(go_pkgs) > 0
    have_net  = (len(net_info["projects"]) > 0) or (len(net_info["old_packages"]) > 0) or (len(net_info["endpoints"]) > 0)

    file_summaries: Dict[str, str] = {}
    per_file_imports: Dict[str, List[str]] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.fields[status]}"),
        BarColumn(),
        TimeElapsedColumn()
    ) as prog:
        task = prog.add_task("scan", total=len(files), status="Scanning …")
        files_sorted = sorted(
            files, 
            key=lambda fi: (
                Path(fi.path).parts[0] if len(Path(fi.path).parts) > 1 else ".", 
                file_priority(fi.path)
            )
        )
        for fi in files_sorted:
            prog.update(task, status=f"Analyzing: {fi.path}")
            imports = extract_imports_for_file(fi.path, fi.lang, fi.content)
            per_file_imports[fi.path] = imports
            imports_md = "\n".join(f"- {i}" for i in imports) or "- (none detected)"
            prompt = FILE_SUMMARY_PROMPT.format(
                path=fi.path, lang=fi.lang, imports_md=imports_md,
                content=fi.content,
            )
            summary_raw = llm.chat(SYSTEM_PROMPT, [{"role": "user", "content": prompt}])
            file_summaries[fi.path] = sanitize_body(summary_raw, path=fi.path)
            prog.advance(task)

    overview_body = ""
    if args.overview:
        overview_raw = llm.chat(
            SYSTEM_PROMPT,
            [{"role": "user", "content": OVERVIEW_PROMPT.format(file_summaries_json=json.dumps(file_summaries, indent=2))}]
        )
        overview_body = sanitize_body(overview_raw)

    by_dir: Dict[str, List[Tuple[str, str]]] = {}
    for path, summ in file_summaries.items():
        parts = Path(path).parts
        top = parts[0] if len(parts) > 1 else "."
        by_dir.setdefault(top, []).append((path, summ))

    # ---------- OUTPUT ----------
    def fmt_inline_list(d: Dict[str,str]) -> str:
        if not d: return "(none)"
        items = [f"{k}{(' @ ' + v) if v else ''}" for k, v in sorted(d.items(), key=lambda kv: kv[0].lower())]
        return ", ".join(items)

    lines: List[str] = []
    lines.append(f"# Project Overview\n\n_Scanned path:_ `{root}`\n_Generated:_ {_dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')} UTC\n")

    # Repository Packages (only for present ecosystems)
    lines.append("## Repository Packages\n")

    if have_node:
        lines.append("### Node (package.json)\n")
        if node_pkgs:
            for manifest, deps in sorted(node_pkgs.items(), key=lambda kv: kv[0].lower()):
                lines.append(f"- `{manifest}`: {fmt_inline_list(deps)}")
            lines.append("")
        else:
            lines.append("- (no package.json files found)\n")

    if have_py:
        lines.append("### Python (requirements / pyproject)\n")
        if py_pkgs:
            for manifest, deps in sorted(py_pkgs.items(), key=lambda kv: kv[0].lower()):
                lines.append(f"- `{manifest}`: {fmt_inline_list(deps)}")
            lines.append("")
        else:
            lines.append("- (no Python manifests found)\n")

    if have_go:
        lines.append("### Go (go.mod)\n")
        if go_pkgs:
            for manifest, mods in sorted(go_pkgs.items(), key=lambda kv: kv[0].lower()):
                lines.append(f"- `{manifest}`: {fmt_inline_list(mods)}")
            lines.append("")
        else:
            lines.append("- (no go.mod files found)\n")

    if have_net:
        # .NET projects
        if net_info["projects"]:
            lines.append("### .NET Projects\n")
            for csproj, info in sorted(net_info["projects"].items(), key=lambda kv: kv[0].lower()):
                tfm = info.get("target_framework") or "(unspecified)"
                lines.append(f"- `{csproj}` — TargetFramework: {tfm}")
                if info.get("project_refs"):
                    joined = ", ".join(sorted(info["project_refs"], key=str.lower))
                    lines.append(f"  - ProjectReference: {joined}")
                if info.get("framework_refs"):
                    joined = ", ".join(sorted(set(info["framework_refs"]), key=str.lower))
                    lines.append(f"  - Framework References: {joined}")
                if info.get("packages"):
                    joined = fmt_inline_list(info["packages"])
                    lines.append(f"  - NuGet (PackageReference): {joined}")
            lines.append("")

        # old NuGet
        if net_info["old_packages"]:
            lines.append("### .NET (packages.config)\n")
            for cfg, pkgs in sorted(net_info["old_packages"].items(), key=lambda kv: kv[0].lower()):
                lines.append(f"- `{cfg}`: {fmt_inline_list(pkgs)}")
            lines.append("")

        # WCF endpoints
        if net_info["endpoints"]:
            lines.append("### WCF Endpoints\n")
            lines.append("| Config | Kind | Address | Binding | Contract | Name |")
            lines.append("|---|---|---|---|---|---|")
            for ep in net_info["endpoints"]:
                lines.append(f"| `{ep['config']}` | {ep['kind']} | {ep.get('address','')} | {ep.get('binding','')} | {ep.get('contract','')} | {ep.get('name','')} |")
            lines.append("")

    # Optional overview
    if overview_body:
        lines.append("## High-level Overview\n")
        lines.append(overview_body.strip() + "\n")

    # Ordered directory list; include root if present
    keys_all = list(by_dir.keys())
    dir_keys = sorted([k for k in keys_all if k != "."], key=str.lower)
    ordered_dirs = (["."] if "." in by_dir else []) + dir_keys

    for d in ordered_dirs:
        is_root = (d == ".")
        dir_label = "(root)" if is_root else d
        lines.append(f"\n## Directory: {dir_label}\n")
        entries_list = by_dir.get(d, [])
        entries = sorted(entries_list, key=lambda t: (file_priority(t[0]), t[0].lower()))
        for path, summ in entries:
            lines.append(f"### {path}\n")
            imports = per_file_imports.get(path) or []
            if imports:
                lines.append("**Libraries/Imports**\n" + "\n".join(f"- {i}" for i in imports) + "\n")
            else:
                lines.append("**Libraries/Imports**\n- (none detected)\n")
            lines.append(summ.strip() + "\n")

    out_md = "\n".join(lines) + "\n"

    if args.dry_run:
        print("\n[DRY RUN] Would write PROJECT_OVERVIEW.md with", len(file_summaries), "file sections.")
        if args.verbose:
            print(out_md[:1500] + ("\n… (truncated)" if len(out_md) > 1500 else ""))
    else:
        (root / DOC_FILE).write_text(out_md, encoding="utf-8")
        print(f"✅ Created {(root / DOC_FILE)}")

if __name__ == "__main__":
    main()
