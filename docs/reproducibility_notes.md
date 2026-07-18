# Reproducibility notes

This repository includes the Streamlit prototype, the frozen knowledge-base module, documented JSON/CSV/Markdown versions of the knowledge base, shared benchmark scenario settings, and dependency specifications.

The application uses fixed generation settings in the LLM wrapper where supported by the API. The knowledge base is frozen and imported at runtime; it is not a retrieval index and does not perform external web search or RAG.

## Knowledge-base boundary

The knowledge base formalises the domain rules embedded in the prototype: typology-specific LCE actions, I5S cues, stage-view templates, representative tools, expected deliverables, and exclusion rules. It excludes the external IPPMD-based benchmark framework to avoid direct prompt contamination.

## Human approval rule

The tool generates decision-support recommendations only. Changes to SOPs, control plans, MES/ERP records, equipment parameters, supplier status, production settings, or operational records require human review and approval.


## Internal verifier wording guard

The Plan--Verify--Refine loop uses internal drafting requirements to improve the next output, but final recommendations are cleaned to avoid exposing verifier/debug language such as “verification feedback”, “feedback gaps”, “missing cues”, or “internal checklist”.

## Secrets

Secrets are intentionally not included in the repository package. Configure `OPENROUTER_API_KEY` through Streamlit secrets or a local, uncommitted `.streamlit/secrets.toml`.
