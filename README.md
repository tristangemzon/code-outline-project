# code-outline

Generate a structured `PROJECT_OVERVIEW.md` for any repo:
- Summaries per file
- Detected package manifests (Node, Python, Go, .NET)
- Optional high-level overview section

## 📦 Local Installation (pip)

You can install this project locally before it’s published to PyPI:

1. **Clone or download** this repository:
   ```bash
   git clone https://github.com/<your-username>/code-outline.git
   cd code-outline
   ```

2. **Install in editable mode** (so changes take effect without reinstalling):
   ```bash
   pip install -e .
   ```

3. **Verify the CLI is available**:
   ```bash
   code-outline --help
   ```

---

## 🚀 Usage

Run against a folder:
```bash
code-outline /path/to/folder --overview --config config.json
```

If you have a `config.json` in the same folder as the package, you can omit `--config`.

---

## ⚙️ Configuration

Copy the example config and fill in your API keys:
```bash
cp code_outline/config.json.example code_outline/config.json
```

- Supports **OpenAI** and **Azure OpenAI** — choose provider in the config.
- Do **not** commit secrets to Git.

---