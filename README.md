# LCE-I5S Manufacturing-System Decision Support

This repository contains a Streamlit prototype for a bounded LLM-based decision-support framework for manufacturing-system and supply-chain design. The tool combines Life Cycle Engineering (LCE), Industry 5.0 I5S dimensions (Social, Sustainable, Sensing, Smart, Safe), and a frozen curated manufacturing knowledge base.

## Repository contents

```text
.
├── app.py
├── manufacturing_knowledge_base.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── secrets.example.toml
├── docs/
│   └── reproducibility_notes.md
├── examples/
│   └── benchmark_scenarios.md
└── knowledge_base/
    ├── manufacturing_knowledge_base.json
    ├── manufacturing_knowledge_base.csv
    └── manufacturing_knowledge_base.md
```

## Knowledge base

The application imports `manufacturing_knowledge_base.py`, a frozen curated knowledge base. File names are stable; the version is stored in the metadata.

The knowledge base contains manufacturing-system typologies, LCE stages, engineering analysis/synthesis/evaluation fields, representative tools and methods, expected deliverables and tollgates, I5S cues, and exclusion/human-approval rules.

This is **not** a RAG index. It performs no runtime retrieval. The knowledge base formalises the domain rules used by the prototype and is included to support reproducibility.

## Benchmark boundary

The IPPMD-based external reference framework used for procedural benchmark scoring is intentionally **not** included as a generation source. This separation helps avoid direct prompt contamination.

## Run locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` using the example file:

```toml
OPENROUTER_API_KEY = "your_openrouter_api_key_here"
```

Then run:

```bash
streamlit run app.py
```

## Decision-support boundary

The prototype generates structured recommendations for human review. It does not autonomously modify SOPs, control plans, MES/ERP records, equipment parameters, supplier status, production settings, or operational records.
