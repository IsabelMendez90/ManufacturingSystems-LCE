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
├── docs/
│   └── reproducibility_notes.md
├── examples/
│   └── benchmark_scenarios.md
└── knowledge_base/
    ├── manufacturing_knowledge_base.json
    ├── manufacturing_knowledge_base.csv
    └── manufacturing_knowledge_base.md
```


## Current package notes

This GitHub package keeps secrets out of the repository. It also includes stricter Facility Design prompting and final-output cleaning to prevent internal verifier wording from appearing in generated recommendations.

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

Configure your `OPENROUTER_API_KEY` in Streamlit secrets. For local testing, create your own `.streamlit/secrets.toml` file, but do not commit it to the repository.

Then run:

```bash
streamlit run app.py
```

## Decision-support boundary

The prototype generates structured recommendations for human review. It does not autonomously modify SOPs, control plans, MES/ERP records, equipment parameters, supplier status, production settings, or operational records.


## v14 update

This package keeps the frozen curated manufacturing knowledge base as a separate module and strengthens Facility Design outputs. For Facility Design, the raw Supply Chain Configuration & Action Plan is rebuilt from the frozen knowledge base so it remains as detailed as the Stage Views. The app also removes maturity-level explanations such as `current L1`, `L1 lacks`, and `(L1→L2)` from narrative sections, and replaces unprovided target-achievement language with design-review language.
