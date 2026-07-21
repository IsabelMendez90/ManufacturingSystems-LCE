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

This GitHub package keeps secrets out of the repository. It also includes deterministic final-output formatting for Product Transfer, Technology Transfer, and Facility Design, plus final-output cleaning to prevent internal verifier wording from appearing in generated recommendations.

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


## v16 update

This package keeps the frozen curated manufacturing knowledge base as a separate module and strengthens Facility Design outputs. For Facility Design, the raw Supply Chain Configuration & Action Plan is rebuilt from the frozen knowledge base so it remains as detailed as the Stage Views. The app also removes maturity-level explanations such as `current L1`, `L1 lacks`, and `(L1→L2)` from narrative sections, replaces unprovided target-achievement language with design-review language, normalises punctuation artifacts from composed KB fragments, and clarifies traceability wording so immutable-chain traceability is used only when explicitly required.


## v16 formatting update

The raw Supply Chain Configuration & Action Plan is rendered with preserved line breaks so each LCE stage remains visually separated in Streamlit outputs and exported/copied text.


## v17 formatting update

- Raw action-plan sections now keep each LCE stage as a separate block.
- Facility Design raw plans use a stage header followed by indented `Action`, `Key information/methods`, `Representative methods/tools`, `Deliverables/tollgates`, and `Evaluation` lines.
- I5S narrative sections are normalised to the canonical order: Social, Sustainable, Sensing, Smart, Safe.
- Markdown bullets/bold are removed from raw narrative sections to make copied benchmark outputs cleaner as `.txt` files.

## v18 Technology Transfer and rationale update

- Technology Transfer raw plans are rebuilt from the frozen knowledge base using the same detailed stage format as Facility Design: `Action`, `Key information/methods`, `Representative methods/tools`, `Deliverables/tollgates`, and `Evaluation`.
- Safe risk wording is corrected so human review is treated as a governance control, not as a deployment risk.
- Technology Transfer Safe actions are prompted to include risk reviews, FMEA/HAZOP or LOTO planning, commissioning safety checks, and incomplete validation evidence as the relevant risk.
- Social rationales distinguish human-centred technology transfer, ergonomic layout design, and workforce training instead of over-linking human movement paths to skill development.
- Expected 5S rationale polishing cannot reduce the number of deterministic evidence-grounded reasons when two reasons are available.
- Raw stage formatting and I5S order normalisation from v17 are retained.

## v19 Technology Transfer I5S formatting update

- Improvement Opportunities & Risks now preserves explicit I5S labels in the canonical order: Social, Sustainable, Sensing, Smart, Safe.
- If the LLM returns five unlabeled Opportunity/Risk pairs, the app formats them under the five I5S labels instead of leaving them as repeated unlabeled lines.
- Technology Transfer Social rationale now uses human-centred work design, SOP development, equipment ramp-up, and operator training language, avoiding Facility Design-specific “layout design” wording when the scenario is Technology Transfer.
- Raw action-plan indentation and stage separation are retained for benchmark-ready copied text.

## v20 strict formatting guard

- Adds a final post-cleaning formatting pass so the raw action-plan stage blocks keep consistent two-space indentation after all safety substitutions.
- Adds a deterministic guard for `Improvement Opportunities & Risks`: if the LLM returns unlabeled `Opportunity/Risk` pairs, the app replaces them with a typology-specific I5S section labelled `Social`, `Sustainable`, `Sensing`, `Smart`, and `Safe`.
- Keeps `Digital/AI Next Steps` as a plain action list instead of forcing I5S labels onto that section.
- Updates the cache revision so deployed Streamlit results are regenerated rather than reusing previous v19 cached outputs.


## v21 deterministic formatting update

- Product Transfer, Technology Transfer, and Facility Design raw Supply Chain Configuration & Action Plan sections are now rebuilt from the frozen knowledge base before display/export.
- Every selected LCE stage uses the same benchmark-ready block format: `Action`, `Key information/methods`, `Representative methods/tools`, `Deliverables/tollgates`, and `Evaluation`.
- `Improvement Opportunities & Risks` is now rendered deterministically in the fixed I5S order: Social, Sustainable, Sensing, Smart, Safe.
- The section no longer depends on LLM indentation or unlabeled `Opportunity/Risk` pairs.
- Product Transfer expected-rationale wording no longer refers to Technology Transfer.
- Cache revision updated to v21.


## v22 preformatted raw-section display update

- Renders raw benchmark sections with `st.code(..., language="text")` so copied outputs preserve newlines and indentation.
- Keeps deterministic raw Supply Chain Configuration & Action Plan reconstruction for Product Transfer, Technology Transfer, and Facility Design.
- Keeps deterministic Improvement Opportunities & Risks in the fixed I5S order: Social, Sustainable, Sensing, Smart, Safe.
- Cache revision updated to v22 to avoid reusing older v21 formatted outputs.
