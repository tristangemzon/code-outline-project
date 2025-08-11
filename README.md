# code-outline

LLM-driven tool to generate a structured `PROJECT_OVERVIEW.md` for any repo:
- Summaries per file generated via a Large Language Model (LLM)
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

Run against a repo:
```bash
code-outline /path/to/repo --overview --config config.json
```

If you have a `config.json` in the same folder as the package, you can omit `--config`.

**Output location:**  
After scanning, a file called `PROJECT_OVERVIEW.md` will be written **inside the folder you specified** in the command.  
Example:
```bash
code-outline ~/projects/my-app
# → writes ~/projects/my-app/PROJECT_OVERVIEW.md
```

---

## ⚙️ Configuration

Copy the example config and fill in your API keys:
```bash
cp code_outline/config.json.example code_outline/config.json
```

- Supports **OpenAI** and **Azure OpenAI** — choose provider in the config.
- Uses your chosen LLM to produce summaries and overviews.
- Do **not** commit secrets to Git.

---

## 🛠 How It Works

1. **Scans your repo**  
   Recursively finds code, documentation, and config files while ignoring common build/output directories.

2. **Detects imports & dependencies**  
   Identifies dependencies from manifests like `package.json`, `requirements.txt`, `go.mod`, `.csproj`, etc.

3. **Generates per-file summaries (LLM)**  
   Sends file content, imports, and metadata to your configured LLM, which returns concise summaries.

4. **Creates an optional high-level overview (LLM)**  
   If `--overview` is passed, it compiles all file summaries and asks the LLM to produce a big-picture explanation.

5. **Writes results to Markdown**  
   Produces a structured `PROJECT_OVERVIEW.md` in the repo folder you scanned, ready for onboarding or documentation.

---

## 📄 License

[MIT](LICENSE)
