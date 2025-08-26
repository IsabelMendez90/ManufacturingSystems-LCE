# LCE + 5S Manufacturing System & Supply Chain Decision Support — AI Agent (no‑BOM)
# Author(s): Dr. J. Isabel Méndez & Dr. Arturo Molina
# Notes:
# - This app implements a bounded AI AGENT (planner → tools → reflector) that USES an LLM.
# - It is NOT "agentic AI" (no autonomous tool selection, no long‑horizon self‑planning/memory).
# - Adds KPI inputs, KPI computation, and a KPI Gate so plans must reference/meet measurable targets.

import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
from fpdf import FPDF
from io import BytesIO
from datetime import datetime
import json, re, math

# ------------------------ App + API setup ------------------------
st.set_page_config(page_title="LCE + 5S Decision Support", layout="wide")
st.title("LCE + 5S Manufacturing System & Supply Chain Decision Support (AI Agent)")
st.markdown("Developed by: Dr. J. Isabel Méndez  & Dr. Arturo Molina")

API_KEY = st.secrets["OPENROUTER_API_KEY"]
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)

TXT_LIMIT = 12000  # keep LLM prompts bounded

# ------------------------ Helpers: file text extraction (best-effort) ------------------------
def try_extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    # Simple text/markdown/CSV/log
    if name.endswith((".txt", ".md", ".csv", ".log")):
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return data.decode("latin-1", errors="ignore")
    # DOCX (optional)
    if name.endswith(".docx"):
        try:
            import docx
            f = BytesIO(data)
            doc = docx.Document(f)
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception:
            return ""
    # PDF (very light, if PyMuPDF present)
    if name.endswith(".pdf"):
        try:
            import fitz  # PyMuPDF
            pdf = fitz.open(stream=data, filetype="pdf")
            text = []
            for i in range(min(20, pdf.page_count)):
                text.append(pdf.load_page(i).get_text())
            return "\n".join(text)
        except Exception:
            return ""
    # Fallback
    return ""

# ------------------------ LLM wrapper ------------------------
def llm(msgs, temperature=0.2, seed=42, model="mistralai/mistral-7b-instruct"):
    resp = client.chat.completions.create(model=model, temperature=temperature, seed=seed, messages=msgs)
    return resp.choices[0].message.content or ""

# ------------------------ 5S Taxonomy (condensed) ------------------------
five_s_taxonomy = {
    "Social":[
        {"desc":"No integration of social factors; minimal worker well-being consideration.","tech":[]},
        {"desc":"Compliance with basic labor laws and ergonomic guidelines.","tech":["Ergonomic checklists","Manual safety reporting"]},
        {"desc":"Workforce development / inclusion programs.","tech":["LMS (Moodle, SuccessFactors)","Engagement apps"]},
        {"desc":"Worker-centric design; real-time well-being feedback.","tech":["Wearables (fatigue)","Sentiment analysis"]},
        {"desc":"Integrated socio-technical systems (co-creation, mental health, community).","tech":["Inclusion dashboards","Psychological safety analytics","Collab platforms"]}
    ],
    "Sustainable":[
        {"desc":"No environmental consideration.","tech":[]},
        {"desc":"Basic compliance; manual resource tracking.","tech":["Spreadsheets","Basic meters"]},
        {"desc":"Energy efficiency; partial recycling.","tech":["ISO 14001 checklists","Env sensors","Recycling SW"]},
        {"desc":"LCA, closed-loop, carbon monitoring.","tech":["OpenLCA","GaBi","Carbon calculators","WMS recycling metrics"]},
        {"desc":"Circular economy; AI optimization/reporting.","tech":["AI energy optimization","Blockchain traceability","ESG analytics"]}
    ],
    "Sensing":[
        {"desc":"Human-only sensing.","tech":["Visual inspection","Manual logging"]},
        {"desc":"Basic sensors; local alarms.","tech":["Thermocouples","Limit switches","PLC indicators"]},
        {"desc":"Multi-sensor + WSN + partial fusion.","tech":["WSN","SCADA","LabVIEW","Basic DAQ"]},
        {"desc":"Embedded + IoT + real-time feedback.","tech":["Azure/AWS IoT","MCUs","OPC-UA/MQTT"]},
        {"desc":"Edge AI + IIoT + adaptive closed-loop.","tech":["Edge AI devices","IIoT platforms","Sensor fusion"]}
    ],
    "Smart":[
        {"desc":"Manual control/decisions.","tech":["Paper logs","Work instructions"]},
        {"desc":"Basic open-loop control.","tech":["Relay logic","Timers","Basic HMI"]},
        {"desc":"Closed-loop automation.","tech":["PID","PLCs","HMI/SCADA"]},
        {"desc":"Advanced automation + predictive + MES.","tech":["MES (Opcenter/Proficy)","Predictive (Maximo/Senseye)"]},
        {"desc":"AI/ANN/IIoT/twins/big data.","tech":["SageMaker/Vertex","Keras/PyTorch","Digital twins"]}
    ],
    "Safe":[
        {"desc":"Minimal safety.","tech":["Safety posters","Visual checks"]},
        {"desc":"Compliance (ISO 45001).","tech":["Safety audits","Risk matrices"]},
        {"desc":"Machine safety integrated.","tech":["Safety PLCs","Light curtains","E-stops","Interlocks"]},
        {"desc":"Smart safety (sensors + anomaly).","tech":["Vibration sensors","AI failure prediction","HSE dashboards"]},
        {"desc":"AI safety management + cyber-physical.","tech":["Safety twins","AI hazard detection","IEC 62443/NIST CSF"]}
    ]
}

# ------------------------ Supply-chain hints ------------------------
supply_chain_recommendations = {
    "Product Transfer":[
        "Agile multi-sourcing; supplier MCDM (AHP/TCO); APQP/SPC; decentralized traceability."
    ],
    "Technology Transfer":[
        "Pilot lines; technical-economic benchmarking; joint dev; predictive + MES/ERP; digital twins."
    ],
    "Facility Design":[
        "Local/resilient networks; make-or-buy; green infra; layout/OEE simulation; cyber-by-design."
    ],
}

# ------------------------ 5S regex critic ------------------------
FIVE_S_PATTERNS = {
    "Social":[r"\bergonom", r"\bhuman factors?\b", r"\bLMS\b", r"\btraining\b", r"\bco-creation\b|\bparticipatory\b"],
    "Sustainable":[r"\bLCA\b", r"\bISO\s?14001\b", r"\bcarbon (footprint|accounting|intensity)\b", r"\bcircular economy\b|\brecycl|\breuse\b", r"\brenewable\b|\benergy (audit|efficiency|management)\b"],
    "Sensing":[r"\bSCADA\b", r"\bOPC[-\s]?UA\b|\bMQTT\b", r"\bDAQ\b", r"\bIIoT\b|\bedge", r"\bcondition monitoring\b"],
    "Smart":[r"\bPLC\b|\bPID\b", r"\bMES\b|\bMOM\b", r"\bpredictive maintenance\b", r"\bdigital twin", r"\boptimization\b|\bAPS\b"],
    "Safe":[r"\bISO\s?45001\b", r"\bISO\s?13849\b|\bIEC\s?61508\b|\bIEC\s?62061\b", r"\bFMEA\b|\bHAZOP\b|\bLOPA\b", r"\bLOTO\b", r"\bIEC\s?62443\b|\bNIST\b"]
}
REQUIRED_MATCHES_BY_LEVEL = {0:0, 1:1, 2:2, 3:3, 4:4}

def find_5s_gaps(plan_text: str, target_levels: dict) -> list:
    text = plan_text or ""
    gaps = []
    for dim, target in target_levels.items():
        if target <= 0:
            continue
        pats = FIVE_S_PATTERNS.get(dim, [])
        hits = set(p for p in pats if re.search(p, text, re.IGNORECASE))
        need = REQUIRED_MATCHES_BY_LEVEL.get(target, 1)
        if len(hits) < need:
            sample = [re.sub(r"\\b", "", p) for p in pats[:5]]
            gaps.append(f"{dim}: need ≥{need} concrete references; found {len(hits)}. Add: {', '.join(sample)}")
    return gaps

# ------------------------ Evidence builder ------------------------
def retrieve_domain_evidence(system_type, industry, selected_stages, five_s_levels):
    stages = [s.split(":")[0].strip() for s in selected_stages] or []
    lines = []
    hints = supply_chain_recommendations.get(system_type, [])
    if hints:
        lines.append("SUPPLY-CHAIN HINTS:")
        lines += [f"- {h}" for h in hints]
    if stages:
        lines.append("\nSELECTED LCE STAGES:")
        lines += [f"- {s}" for s in stages]
    lines.append("\n5S TECH HINTS @ CURRENT LEVEL:")
    for dim, lvl in five_s_levels.items():
        techs = five_s_taxonomy[dim][lvl].get("tech", [])
        if techs:
            lines.append(f"- {dim} (L{lvl}): " + "; ".join(techs))
    return "\n".join(lines)

# ------------------------ Plan parsing helpers ------------------------
def parse_section(text, head):
    m = re.search(rf"\[{re.escape(head)}\](.*?)(?=\n\[[A-Z].*?\]|\Z)", text or "", re.DOTALL)
    return m.group(1).strip() if m else ""

def extract_expected_levels(text, fallback):
    out = {}
    if not isinstance(text, str):
        text = ""
    for dim in ["Social", "Sustainable", "Sensing", "Smart", "Safe"]:
        pat = rf"{dim}\s*[:\-]?\s*(?:L(?:e?vel)?\s*=?\s*)?([0-4])"
        m = re.search(pat, text, flags=re.IGNORECASE)
        if not m:
            m = re.search(rf"{dim}.{{0,30}}?([0-4])", text, flags=re.IGNORECASE | re.DOTALL)
        out[dim] = int(m.group(1)) if m else int(fallback.get(dim, 0))
    return out

# ------------------------ Agent state ------------------------
class AgentState:
    def __init__(self, objective, system_type, industry, five_s_levels, selected_stages,
                 demand_info, docs_text, kpi_inputs, kpi_targets):
        self.objective = objective
        self.system_type = system_type
        self.industry = industry
        self.five_s_levels = five_s_levels
        self.selected_stages = selected_stages
        self.demand_info = demand_info or {}
        self.docs_text = docs_text or ""
        self.kpi_inputs = kpi_inputs or {}
        self.kpis = {}
        self.kpi_targets = kpi_targets or {}
        self.spec_data = {}
        self.observations = []

# ------------------------ Planner ------------------------
def plan_with_llm(state: AgentState, evidence: str) -> str:
    lce_txt = "\n".join([f"- {s}" for s in state.selected_stages]) if state.selected_stages else "None"
    s_txt  = "\n".join([f"- {d}: L{v}" for d, v in state.five_s_levels.items()])
    dem_js = json.dumps(state.demand_info, ensure_ascii=False)
    doc_snip = (state.docs_text or "")[:TXT_LIMIT]
    prompt = f"""
You are an expert in {state.system_type} manufacturing systems for the {state.industry} industry.

Context:
Objective: {state.objective}
Selected LCE stages:
{lce_txt}
Current 5S maturity:
{s_txt}
Demand/capacity info (if any, JSON): {dem_js}
Relevant document snippets (if any):
"""{doc_snip}"""

Use the evidence strictly:
{evidence}

If capacity observations were provided, include explicit takt time and the number of parallel stations in the configuration, layout, staffing, and scheduling decisions.

Return the following sections (markdown, concise and actionable):
[Supply Chain Configuration & Action Plan]
[Improvement Opportunities & Risks]
[Digital/AI Next Steps]
[Expected 5S Maturity]  (List as: Social: X, Sustainable: Y, Sensing: Z, Smart: W, Safe: V.)
"""
    return llm([{"role":"system","content":"You are a digital manufacturing systems expert."},
                {"role":"user","content":prompt}], temperature=0.2, seed=42)

# ------------------------ Tools ------------------------
class Tool:
    name = "base"
    def run(self, state: AgentState) -> dict:
        return {"tool": self.name, "summary": "noop", "data": {}}

class SpecExtractorLLM(Tool):
    name = "spec_extractor"
    def run(self, state):
        prompt = f"""Extract structured requirements from the objective and LCE stages.

Objective: {state.objective}
LCE stages: {', '.join([s.split(':')[0] for s in state.selected_stages]) or 'None'}
Industry: {state.industry}

Return JSON ONLY:
{{
 "requirements": ["<short bullets>"],
 "constraints": ["<standards, compliance, budget, quality if implied>"],
 "success_metrics": ["<KPI targets like: OEE >= 75%, Lead time <= 10 days, CO2e <= 2 kg/unit>"]
}}"""
        txt = llm([{"role":"system","content":"You turn free text into compact JSON specs."},
                   {"role":"user","content":prompt}], temperature=0.1)
        try:
            js = json.loads(re.search(r"\{.*\}", txt, re.DOTALL).group(0))
        except Exception:
            js = {"requirements":[],"constraints":[],"success_metrics":[]}
        state.spec_data = js
        return {"tool": self.name, "summary": "Structured specs extracted", "data": js}

class DocIntakeLLM(Tool):
    name = "doc_intake"
    def run(self, state):
        if not state.docs_text.strip():
            return {"tool": self.name, "summary":"No documents provided.", "data":{}}
        prompt = f"""From the following text, extract constraints, required standards, safety/environmental notes, and any numeric parameters.
Return JSON ONLY with keys: constraints, standards, safety, parameters.
TEXT:
"""{state.docs_text[:TXT_LIMIT]}""""""
        txt = llm([{"role":"system","content":"You extract constraints from long docs with concise JSON."},
                   {"role":"user","content":prompt}], temperature=0.2)
        try:
            js = json.loads(re.search(r"\{.*\}", txt, re.DOTALL).group(0))
        except Exception:
            js = {"constraints":[],"standards":[],"safety":[],"parameters":{}}
        return {"tool": self.name, "summary":"Doc constraints extracted", "data": js}

class CapacityCalculator(Tool):
    name = "capacity_calc"
    def run(self, state):
        d = state.demand_info or {}
        if not d or not d.get("weekly_output") or not d.get("cycle_time_sec"):
            return {"tool": self.name, "summary":"Insufficient demand inputs; skipped.", "data":{}}
        weekly_output = float(d.get("weekly_output",0))
        cycle_time = float(d.get("cycle_time_sec",0))
        shifts = int(d.get("shifts_per_day",1))
        hours = float(d.get("hours_per_shift",8.0))
        days = int(d.get("days_per_week",5))
        oee = float(d.get("oee",0.7))
        available_sec = shifts*hours*3600*days*oee
        required_station_secs = weekly_output*cycle_time
        stations = math.ceil(required_station_secs/max(1,available_sec))
        takt_sec = (available_sec/max(1,weekly_output)) if weekly_output>0 else 0
        return {"tool": self.name, "summary": f"Capacity sizing → stations={stations}, takt≈{takt_sec:.1f}s",
                "data":{"stations":stations,"takt_sec":takt_sec,"available_sec":available_sec}}

class KPIEvaluator(Tool):
    name = "kpi_eval"
    def run(self, state):
        d = state.demand_info or {}
        ki = state.kpi_inputs or {}
        out = {}
        # Throughput (units/h)
        days = d.get("days_per_week", 5)
        hours = d.get("hours_per_shift", 8.0) * d.get("shifts_per_day", 1)
        weekly_output = d.get("weekly_output", 0)
        denom = (days * hours)
        out["throughput_units_per_h"] = (weekly_output / denom) if denom > 0 else None
        # OEE, FPY
        rt = (ki.get("runtime_h") or 0.0) * 3600.0
        stime = (ki.get("scheduled_time_h") or 0.0) * 3600.0
        total = ki.get("total_count") or 0
        good = ki.get("good_count") or 0
        ict = ki.get("ideal_cycle_time_s") or 0.0
        if stime>0 and rt>0 and total>0 and ict>0:
            avail = rt / stime
            perf = (ict * total) / rt
            qual = good / total if total>0 else 0
            oee = max(0.0, min(1.0, avail * perf * qual)) * 100.0
            fpy = max(0.0, min(1.0, qual)) * 100.0
            out["OEE_pct"] = oee
            out["FPY_pct"] = fpy
        else:
            out["OEE_pct"] = None
            out["FPY_pct"] = None
        # ESG per unit
        if weekly_output and weekly_output>0:
            out["energy_kWh_per_unit"] = (ki.get("energy_kwh_week") or 0.0) / weekly_output if ki.get("energy_kwh_week") else None
            out["co2e_kg_per_unit"] = (ki.get("co2e_kg_week") or 0.0) / weekly_output if ki.get("co2e_kg_week") else None
            out["water_L_per_unit"] = (ki.get("water_l_week") or 0.0) / weekly_output if ki.get("water_l_week") else None
        else:
            out["energy_kWh_per_unit"] = None
            out["co2e_kg_per_unit"] = None
            out["water_L_per_unit"] = None
        out["changeover_min"] = ki.get("changeover_min") or None
        state.kpis = out
        return {"tool": self.name, "summary": "KPI metrics computed", "data": out}

class StandardsGate(Tool):
    name = "standards_gate"
    # Core reference rules, plus a KPI reference rule when KPI data/targets exist
    RULES = [
        ("quality_system", r"ISO\s?9001"),
        ("environment", r"ISO\s?14001|LCA|GaBi|OpenLCA"),
        ("ohs", r"ISO\s?45001"),
        ("machine_safety", r"ISO\s?13849|IEC\s?61508|IEC\s?62061"),
        ("cybersecurity", r"IEC\s?62443|NIST"),
        ("traceability", r"APQP|SPC|traceability|QR|blockchain"),
        ("interoperability", r"OPC[-\s]?UA|MQTT"),
        ("capacity_ref", r"\btakt\b|\bstations?\b|\bline balancing\b|\bparallel\b"),
        ("kpi_ref", r"\b(OEE|first[-\s]?pass yield|FPY|CO2|CO2e|energy per unit|lead time|service level|fill rate)\b"),
    ]

    KPI_KEY_MAP = {
        "oee": "OEE_pct",
        "fpy": "FPY_pct",
        "service level": "service_level_pct",
        "fill rate": "service_level_pct",
        "lead time": "lead_time_days",
        "co2": "co2e_kg_per_unit",
        "co2e": "co2e_kg_per_unit",
        "energy": "energy_kWh_per_unit",
    }

    def _parse_kpi_targets(self, text: str, success_metrics:list) -> dict:
        targets = {}
        blob = (text or "") + "\n" + "\n".join(success_metrics or [])
        # Percent up-goals (>=)
        for m in re.finditer(r"\b(OEE|FPY|service level|fill rate)\b.*?(>=|≥|>|=)\s*([0-9]+(?:\.[0-9]+)?)\s*%?", blob, re.IGNORECASE):
            key = self.KPI_KEY_MAP[m.group(1).lower()]
            val = float(m.group(3))
            if val <= 1.0:  # treat as fraction
                val *= 100.0
            targets[key] = (">=", val)
        # Lead time down-goal (<= days)
        for m in re.finditer(r"\b(lead time)\b.*?(<=|≤|<|=)\s*([0-9]+(?:\.[0-9]+)?)\s*(d|day|days)\b", blob, re.IGNORECASE):
            key = self.KPI_KEY_MAP[m.group(1).lower()]
            val = float(m.group(3))
            targets[key] = ("<=", val)
        # CO2e per unit down-goal (<= kg)
        for m in re.finditer(r"\b(CO2e?|carbon)\b.*?(<=|≤|<|=)\s*([0-9]+(?:\.[0-9]+)?)\s*(kg)(?:/unit|\s*per\s*unit)?\b", blob, re.IGNORECASE):
            key = "co2e_kg_per_unit"
            val = float(m.group(3))
            targets[key] = ("<=", val)
        # Energy per unit down-goal (<= kWh)
        for m in re.finditer(r"\benergy\b.*?(<=|≤|<|=)\s*([0-9]+(?:\.[0-9]+)?)\s*(kWh)(?:/unit|\s*per\s*unit)?\b", blob, re.IGNORECASE):
            key = "energy_kWh_per_unit"
            val = float(m.group(2))
            targets[key] = ("<=", val)
        return targets

    def _eval_kpi_targets(self, kpis: dict, targets: dict) -> list:
        warns = []
        for key, (op, tval) in targets.items():
            kval = kpis.get(key)
            label = key
            if kval is None:
                warns.append(f"KPI target declared but cannot be evaluated: {label} (no data)")
                continue
            ok = True
            if op == ">=":
                ok = kval >= tval
            elif op == "<=":
                ok = kval <= tval
            elif op == "=":
                ok = abs(kval - tval) < 1e-6
            if not ok:
                warns.append(f"KPI target not met: {label} {op} {tval} (observed {kval:.3g})")
        return warns

    def run(self, state):
        text = getattr(state, "_latest_plan_text", "")
        ev   = getattr(state, "_latest_evidence", "")
        scope = (text + "\n" + ev)
        warnings = []

        has_demand = bool(state.demand_info.get("weekly_output") and state.demand_info.get("cycle_time_sec"))

        # Parse KPI targets from plan + spec success_metrics, then merge with user targets (user overrides)
        parsed_targets = self._parse_kpi_targets(text, (state.spec_data or {}).get("success_metrics", []))
        merged_targets = dict(parsed_targets)
        for k, v in (state.kpi_targets or {}).items():
            if v is None or v == 0:
                continue
            # infer direction by metric
            if k in ("OEE_pct", "FPY_pct", "service_level_pct"):
                merged_targets[k] = (">=", float(v))
            else:
                merged_targets[k] = ("<=", float(v))

        # Enable rules heuristically based on system type + levels + demand + KPI context
        enable = {
            "quality_system": True,
            "environment": state.five_s_levels.get("Sustainable",0) >= 2,
            "ohs": state.five_s_levels.get("Safe",0) >= 1,
            "machine_safety": state.five_s_levels.get("Smart",0) >= 2,
            "cybersecurity": (state.five_s_levels.get("Smart",0)>=3 or state.five_s_levels.get("Sensing",0)>=3),
            "traceability": state.system_type in ("Product Transfer","Technology Transfer"),
            "interoperability": state.five_s_levels.get("Sensing",0)>=3,
            "capacity_ref": has_demand,
            "kpi_ref": bool(state.kpis) or bool(merged_targets),
        }
        for key, regex in self.RULES:
            if enable.get(key):
                if not re.search(regex, scope, flags=re.IGNORECASE):
                    label = key.replace("_"," ").title()
                    warnings.append(f"Missing reference: {label} (expected pattern: {regex})")

        # Evaluate KPI targets if any
        warnings += self._eval_kpi_targets(state.kpis or {}, merged_targets)
        # Persist the merged targets for UI/PDF
        state.kpi_targets = {k: v[1] for k, v in merged_targets.items()}
        return {"tool": self.name, "summary": "Standards Gate: OK" if not warnings else "Standards Gate: WARN", "data": {"warnings": warnings}}

class RiskRegisterLLM(Tool):
    name = "risk_register"
    def run(self, state):
        prompt = f"""Produce a compact risk register (FMEA-lite) given:
Objective: {state.objective}
Industry: {state.industry}
System type: {state.system_type}
LCE stages: {', '.join([s.split(':')[0] for s in state.selected_stages]) or 'None'}

Return JSON ONLY:
{{
 "risks":[
   {{"name":"...","cause":"...","effect":"...","mitigation":"...","owner":"Role","stage":"<LCE>"}}
 ],
 "notes":"(optional)"
}}
Limit to 6–10 highest-impact items."""
        txt = llm([{"role":"system","content":"You generate concise JSON risk registers."},
                   {"role":"user","content":prompt}], temperature=0.2)
        try:
            js = json.loads(re.search(r"\{.*\}", txt, re.DOTALL).group(0))
        except Exception:
            js = {"risks":[],"notes":""}
        return {"tool": self.name, "summary": f"Risks generated: {len(js.get('risks',[]))}", "data": js}

TOOL_REGISTRY = {
    "spec_extractor": SpecExtractorLLM(),
    "doc_intake":     DocIntakeLLM(),
    "capacity_calc":  CapacityCalculator(),
    "kpi_eval":       KPIEvaluator(),
    "standards_gate": StandardsGate(),
    "risk_register":  RiskRegisterLLM(),
}

# ------------------------ Agent runner ------------------------
def agent_run(objective, system_type, industry, selected_stages, five_s_levels,
              demand_info: dict, docs_text: str, kpi_inputs: dict, kpi_targets: dict,
              enable_agent=True, auto_iterate=True, max_steps=3):
    state = AgentState(objective, system_type, industry, five_s_levels, selected_stages,
                       demand_info, docs_text, kpi_inputs, kpi_targets)
    evidence = retrieve_domain_evidence(system_type, industry, selected_stages, five_s_levels)
    plan_text = ""; standards_warnings = []

    steps = 1 if not enable_agent else max_steps
    observations = []
    for step in range(steps):
        # PLAN
        plan_text = plan_with_llm(state, evidence)
        state._latest_plan_text = plan_text
        state._latest_evidence = evidence

        sc_text = parse_section(plan_text, "Supply Chain Configuration & Action Plan") or plan_text
        gaps = find_5s_gaps(sc_text, five_s_levels)

        # EXECUTE TOOLS (context-aware)
        run_tools = []
        if step == 0: run_tools.append("spec_extractor")
        if docs_text.strip() and step == 0: run_tools.append("doc_intake")
        if demand_info.get("weekly_output") and demand_info.get("cycle_time_sec"): run_tools.append("capacity_calc")
        run_tools.append("kpi_eval")
        run_tools.append("standards_gate")
        if step == 0: run_tools.append("risk_register")

        observations = []
        for name in run_tools:
            out = TOOL_REGISTRY[name].run(state)
            observations.append(out)
            if name == "standards_gate":
                standards_warnings = out["data"].get("warnings", [])

        # REFLECT
        passed_standards = (len(standards_warnings) == 0)
        passed_5s = (len(gaps) == 0)
        if not enable_agent or not auto_iterate:
            evidence += "\n\n[OBSERVATIONS]\n" + "\n".join([f"- {o['tool']}: {o['summary']}" for o in observations])
            break

        if passed_standards and passed_5s:
            evidence += "\n\n[OBSERVATIONS]\n" + "\n".join([f"- {o['tool']}: {o['summary']}" for o in observations])
            break
        else:
            critic = []
            if not passed_standards:
                critic.append("STANDARDS FINDINGS:\n- " + "\n- ".join(standards_warnings[:10]))
            if not passed_5s:
                critic.append("5S COVERAGE FINDINGS:\n- " + "\n- ".join(gaps[:10]))
            # if capacity tool ran, surface its suggestion
            cap = next((o for o in observations if o["tool"] == "capacity_calc"), None)
            if cap and cap.get("data", {}).get("stations"):
                s = cap["data"]
                critic.append(f"CAPACITY SUGGESTION:\n- Target stations: {s['stations']} (takt≈{s['takt_sec']:.1f}s). Reflect in layout, staffing, and scheduling.")
            evidence += "\n\n[NEEDS FIX — ADDRESS IN NEXT PASS]\n" + "\n\n".join(critic)

    # collect outputs
    risk = next((o for o in observations if o["tool"] == "risk_register"), {"data": {"risks": []}})
    cap = next((o for o in observations if o["tool"] == "capacity_calc"), {"data": {}})

    return {
        "plan_text": plan_text,
        "standards_warnings": standards_warnings,
        "risk_register": risk["data"],
        "capacity": cap.get("data", {}),
        "kpis": state.kpis,
        "kpi_targets": state.kpi_targets,
    }

# ------------------------ UI ------------------------

st.header("1) Define Scenario")
objective = st.text_input("Objective (e.g., launch new product, adopt a new process, expand a facility):",
                          value=st.session_state.get("objective","Design and ramp a flexible small manufacturing cell."),
                          key="objective")
industry = st.selectbox("Industry:", ["Automotive","Electronics","Medical Devices","Consumer Goods","Other"],
                        index=st.session_state.get("industry_idx",1), key="industry")
st.session_state["industry_idx"] = ["Automotive","Electronics","Medical Devices","Consumer Goods","Other"].index(industry)
system_types = ["Product Transfer","Technology Transfer","Facility Design"]
system_type = st.radio("Manufacturing system type:", system_types, key="system_type")

st.header("2) Select Relevant LCE Stages/Actions")
lce_actions = {
    "Product Transfer":[
        "Ideation: Identify product BOM, materials, quality.",
        "Basic Development: Define manufacturing & QC specs.",
        "Advanced Development: Supplier evaluation/selection.",
        "Launching: Assembly/QC/training.",
        "End-of-Life: Disassembly/reuse/returns."
    ],
    "Technology Transfer":[
        "Ideation: Capture technical specs for new component/family.",
        "Basic Development: Select technology/equipment/tools.",
        "Advanced Development: Process plan, control docs, SOPs.",
        "Launching: Install/test/ramp; optimize.",
        "End-of-Life: Retire/adapt; digital-twin learnings."
    ],
    "Facility Design":[
        "Ideation: Specify product needs and processes.",
        "Basic Development: Select systems and equipment.",
        "Advanced Development: Layout + capacity strategy.",
        "Launching: Build/install/ramp; evaluate.",
        "End-of-Life: Audit for reuse/decommission/transform."
    ]
}
selected_stages=[]
for i, action in enumerate(lce_actions[system_type]):
    if st.checkbox(action, key=f"lce_{i}"):
        selected_stages.append(action)

st.header("3) Current 5S Maturity (one per S)")
five_s_levels={}
cols = st.columns(5)
for i, dim in enumerate(["Social","Sustainable","Sensing","Smart","Safe"]):
    with cols[i]:
        options=[f"Level {idx}: {lvl['desc']}" for idx,lvl in enumerate(five_s_taxonomy[dim])]
        sel = st.radio(dim, options, index=0, key=f"{dim}_radio")
        five_s_levels[dim] = options.index(sel)
        techs = five_s_taxonomy[dim][five_s_levels[dim]].get("tech", [])
        if techs:
            st.caption("Tech hints: " + "; ".join(techs))

st.header("4) Agent Controls")
enable_agent = st.toggle("Enable Agent Mode (plan → tools → reflect)", value=True)
auto_iterate   = st.toggle("Auto-iterate until pass (standards + 5S coverage)", value=True)

st.header("5) Demand / Capacity Inputs (optional but recommended)")
c1,c2,c3,c4,c5 = st.columns(5)
with c1: weekly_output = st.number_input("Target weekly output (units)", min_value=0, value=0)
with c2: cycle_time_sec = st.number_input("Cycle time per unit (sec)", min_value=0.0, value=0.0, step=1.0)
with c3: shifts_per_day = st.number_input("Shifts per day", min_value=1, max_value=4, value=1)
with c4: hours_per_shift = st.number_input("Hours per shift", min_value=1.0, max_value=12.0, value=8.0, step=0.5)
with c5: days_per_week = st.number_input("Days per week", min_value=1, max_value=7, value=5)

oee_assumed = st.slider("Assumed OEE for capacity calc (if unknown)", min_value=0.3, max_value=0.95, value=0.7, step=0.05)
demand_info = {
    "weekly_output": weekly_output,
    "cycle_time_sec": cycle_time_sec,
    "shifts_per_day": shifts_per_day,
    "hours_per_shift": hours_per_shift,
    "days_per_week": days_per_week,
    "oee": oee_assumed,
}

st.subheader("5b) Optional KPI Inputs (for OEE/FPY/ESG)")
with st.expander("Enter KPI inputs if available"):
    e1, e2, e3 = st.columns(3)
    with e1:
        scheduled_time_h = st.number_input("Scheduled Time (h/week)", 0.0, 200.0, 40.0, 0.5)
        runtime_h = st.number_input("Runtime (h/week)", 0.0, 200.0, 36.0, 0.5)
        ideal_cycle_time_s = st.number_input("Ideal Cycle Time (s/unit)", 0.0, 3600.0, 60.0, 1.0)
    with e2:
        total_count = st.number_input("Units Produced (total)", 0, 1_000_000, 0, 1)
        good_count = st.number_input("Good Units", 0, 1_000_000, 0, 1)
        changeover_min = st.number_input("Avg Changeover (min)", 0.0, 600.0, 0.0, 1.0)
    with e3:
        energy_kwh_week = st.number_input("Energy (kWh/week)", 0.0, 1e9, 0.0, 1.0)
        co2e_kg_week = st.number_input("CO₂e (kg/week)", 0.0, 1e9, 0.0, 1.0)
        water_l_week = st.number_input("Water (L/week)", 0.0, 1e12, 0.0, 1.0)

kpi_inputs = dict(
    scheduled_time_h=scheduled_time_h, runtime_h=runtime_h, ideal_cycle_time_s=ideal_cycle_time_s,
    total_count=total_count, good_count=good_count, changeover_min=changeover_min,
    energy_kwh_week=energy_kwh_week, co2e_kg_week=co2e_kg_week, water_l_week=water_l_week,
)

st.subheader("5c) Optional KPI Targets (used by KPI Gate)")
with st.expander("Set targets to enable pass/fail checks"):
    t1, t2, t3 = st.columns(3)
    with t1:
        tgt_oee = st.number_input("Target OEE (%)", 0.0, 100.0, 0.0, 0.5)
        tgt_fpy = st.number_input("Target FPY (%)", 0.0, 100.0, 0.0, 0.5)
        tgt_service = st.number_input("Target Service Level / Fill Rate (%)", 0.0, 100.0, 0.0, 0.5)
    with t2:
        tgt_lead_time = st.number_input("Target Lead Time (days) — lower is better", 0.0, 365.0, 0.0, 0.5)
        tgt_energy = st.number_input("Target Energy (kWh/unit) — lower is better", 0.0, 1e6, 0.0, 0.01)
        tgt_co2e = st.number_input("Target CO₂e (kg/unit) — lower is better", 0.0, 1e6, 0.0, 0.01)
    with t3:
        pass

kpi_targets = {
    "OEE_pct": tgt_oee if tgt_oee > 0 else None,
    "FPY_pct": tgt_fpy if tgt_fpy > 0 else None,
    "service_level_pct": tgt_service if tgt_service > 0 else None,
    "lead_time_days": tgt_lead_time if tgt_lead_time > 0 else None,
    "energy_kWh_per_unit": tgt_energy if tgt_energy > 0 else None,
    "co2e_kg_per_unit": tgt_co2e if tgt_co2e > 0 else None,
}

st.header("6) (Optional) Upload relevant docs (manuals/specs/SOPs)")
uploads = st.file_uploader("Upload .txt/.md/.csv/.log/.docx/.pdf (max a few MB). Multiple allowed.",
                           type=["txt","md","csv","log","docx","pdf"], accept_multiple_files=True)
docs_text = ""
if uploads:
    pieces=[]
    for f in uploads:
        text = try_extract_text(f)
        if text:
            pieces.append(f"# {f.name}\n{text}")
    docs_text = "\n\n".join(pieces)

# Run agent
if st.button("Generate Plan & Recommendations"):
    with st.spinner("Agent planning, executing tools, and refining..."):
        result = agent_run(
            objective=objective,
            system_type=system_type,
            industry=industry,
            selected_stages=selected_stages,
            five_s_levels=five_s_levels,
            demand_info=demand_info,
            docs_text=docs_text,
            kpi_inputs=kpi_inputs,
            kpi_targets=kpi_targets,
            enable_agent=enable_agent,
            auto_iterate=auto_iterate,
            max_steps=3,
        )
        st.session_state["agent_result"] = result

# ------------------------ Results ------------------------
res = st.session_state.get("agent_result")
if res:
    st.header("7) Results")
    plan_text = res["plan_text"]
    st.subheader("Supply Chain Configuration & Action Plan")
    st.info(parse_section(plan_text,"Supply Chain Configuration & Action Plan") or "—")
    st.subheader("Improvement Opportunities & Risks")
    st.info(parse_section(plan_text,"Improvement Opportunities & Risks") or "—")
    st.subheader("Digital/AI Next Steps")
    st.info(parse_section(plan_text,"Digital/AI Next Steps") or "—")
    st.subheader("Expected 5S Maturity")
    exp = parse_section(plan_text,"Expected 5S Maturity")
    st.info(exp or "—")
    expected_5s = extract_expected_levels(exp, five_s_levels)

    st.header("8) 5S Profiles")
    curr_df = pd.DataFrame({"Dimension": list(five_s_levels.keys()), "Level":[five_s_levels[k] for k in five_s_levels]})
    fig_curr = px.line_polar(curr_df, r="Level", theta="Dimension", line_close=True, range_r=[0,4])
    st.plotly_chart(fig_curr, use_container_width=True)
    if expected_5s:
        exp_df = pd.DataFrame({
            "Dimension": list(expected_5s.keys()),
            "Level": [expected_5s[k] for k in expected_5s]
        })
        exp_df["Level"] = pd.to_numeric(exp_df["Level"], errors="coerce").fillna(0).clip(0, 4)
        fig_exp = px.line_polar(exp_df, r="Level", theta="Dimension", line_close=True, range_r=[0, 4])
        st.plotly_chart(fig_exp, use_container_width=True)

    # Capacity
    cap = res.get("capacity", {})
    # Show section only if capacity was actually computed
    if cap and cap.get("takt_sec") is not None:
        st.header("9) Capacity Sizing (from demand inputs)")
        st.success(f"Computed takt ≈ {cap.get('takt_sec',0):.1f} s | Required parallel stations: {cap.get('stations','?')}")
        st.caption("The agent feeds these into the planner; the plan should echo them in layout/staffing.")

    # KPI Panel
    kpis = res.get("kpis", {})
    targets = res.get("kpi_targets", {})
    any_kpi_input = any([
        scheduled_time_h>0, runtime_h>0, ideal_cycle_time_s>0,
        total_count>0, good_count>0, changeover_min>0,
        energy_kwh_week>0, co2e_kg_week>0, water_l_week>0,
    ])
    any_kpi_target = any(v for v in targets.values())
    computed_has_kpis = any(v is not None for v in kpis.values())

    if any_kpi_input or any_kpi_target or computed_has_kpis:
        st.header("10) KPI Panel")
        # Build a tidy KPI table for display
        def tidy(metric_key, label, unit):
            val = kpis.get(metric_key)
            tgt = targets.get(metric_key)
            status = "—"
            if val is None and tgt:
                status = "no data"
            elif val is not None and tgt:
                if metric_key in ("OEE_pct","FPY_pct","service_level_pct"):
                    status = "OK" if val >= tgt else "NOT MET"
                else:
                    status = "OK" if val <= tgt else "NOT MET"
            return {"Metric": label, "Value": None if val is None else (f"{val:.3g} {unit}" if unit else f"{val:.3g}"),
                    "Target": None if not tgt else (f"{tgt:.3g} {unit}" if unit else f"{tgt:.3g}"),
                    "Status": status}

        rows = [
            tidy("throughput_units_per_h", "Throughput", "units/h"),
            tidy("OEE_pct", "OEE", "%"),
            tidy("FPY_pct", "FPY", "%"),
            tidy("changeover_min", "Changeover", "min"),
            tidy("energy_kWh_per_unit", "Energy / Unit", "kWh"),
            tidy("co2e_kg_per_unit", "CO₂e / Unit", "kg"),
            tidy("water_L_per_unit", "Water / Unit", "L"),
            tidy("lead_time_days", "Lead Time", "days"),
            tidy("service_level_pct", "Service Level / Fill Rate", "%"),
        ]
        df_kpi = pd.DataFrame(rows)
        st.dataframe(df_kpi, use_container_width=True)

        # Standards Gate (only if KPI context exists or warnings present)
        warns = res.get("standards_warnings", [])
        if warns or any_kpi_input or any_kpi_target:
            st.header("11) Standards Gate")
            if warns:
                msg = "Standards/compliance warnings:\n- " + "\n- ".join([str(w) for w in warns])
                st.error(msg)
            else:
                st.success("Standards Gate: OK")

    # Risk Register (only if risks exist)
    risks = res.get("risk_register",{}).get("risks",[])
    if risks:
        st.header("12) Risk Register")
        df = pd.DataFrame(risks)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download Risk Register (CSV)", df.to_csv(index=False).encode("utf-8"),
                           "risk_register.csv","text/csv")

# ------------------------ PDF export ------------------------

def to_latin1(text):
    if not isinstance(text,str): text=str(text)
    return text.encode('latin-1','replace').decode('latin-1')


def generate_pdf_report(plan_text, five_s_levels, warnings, risks, cap, kpis, targets):
    pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", size=12)
    pdf.cell(0,10,to_latin1("LCE + 5S Manufacturing Decision Support (AI Agent, no‑BOM)"), ln=True, align="C")
    pdf.set_font("Arial", size=11)
    pdf.cell(0,8,to_latin1("Developed by Dr. Juana Isabel Méndez and Dr. Arturo Molina"), ln=True, align="C")
    pdf.cell(0,8,to_latin1("Date: "+datetime.now().strftime("%Y-%m-%d")), ln=True, align="C")
    pdf.ln(6)
    for head in ["Supply Chain Configuration & Action Plan","Improvement Opportunities & Risks","Digital/AI Next Steps","Expected 5S Maturity"]:
        pdf.set_font("Arial","B",12); pdf.cell(0,8,to_latin1(head), ln=True)
        pdf.set_font("Arial", size=11); pdf.multi_cell(0,7,to_latin1(parse_section(plan_text, head) or "—")); pdf.ln(2)

    # Capacity
    if cap and cap.get("takt_sec") is not None:
        pdf.set_font("Arial","B",12); pdf.cell(0,8,to_latin1("Capacity Sizing"), ln=True)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0,7,to_latin1(f"Takt ≈ {cap.get('takt_sec',0):.1f} s; Required stations: {cap.get('stations','?')}"))

    # KPI table
    if kpis:
        pdf.ln(4); pdf.set_font("Arial","B",12); pdf.cell(0,8,to_latin1("KPI Summary"), ln=True)
        pdf.set_font("Arial", size=11)
        def line(label, key, unit):
            v = kpis.get(key); t = targets.get(key)
            vtxt = "—" if v is None else (f"{v:.3g} {unit}" if unit else f"{v:.3g}")
            ttxt = "—" if not t else (f"{t:.3g} {unit}" if unit else f"{t:.3g}")
            pdf.multi_cell(0,6,to_latin1(f"- {label}: {vtxt} | Target: {ttxt}"))
        line("Throughput","throughput_units_per_h","units/h")
        line("OEE","OEE_pct","%")
        line("FPY","FPY_pct","%")
        line("Changeover","changeover_min","min")
        line("Energy / Unit","energy_kWh_per_unit","kWh")
        line("CO₂e / Unit","co2e_kg_per_unit","kg")
        line("Water / Unit","water_L_per_unit","L")
        line("Lead Time","lead_time_days","days")
        line("Service Level","service_level_pct","%")

    if warnings:
        pdf.ln(4); pdf.set_font("Arial","B",12); pdf.cell(0,8,"Standards Gate Warnings", ln=True)
        pdf.set_font("Arial", size=11)
        for w in warnings:
            pdf.multi_cell(0,6,to_latin1(f"- {w}"))

    if risks:
        pdf.ln(4); pdf.set_font("Arial","B",12); pdf.cell(0,8,"Risk Register (FMEA-lite)", ln=True)
        pdf.set_font("Arial", size=11)
        for r in risks[:12]:
            line = f"- {r.get('name','')} | cause: {r.get('cause','')} | mitigation: {r.get('mitigation','')} | owner: {r.get('owner','')} | stage: {r.get('stage','')}"
            pdf.multi_cell(0,6,to_latin1(line[:220]))

    buf = BytesIO(); pdf_bytes = pdf.output(dest='S').encode('latin1')
    buf.write(pdf_bytes); buf.seek(0); return buf

if res:
    pdf_buf = generate_pdf_report(
        res.get("plan_text",""),
        five_s_levels,
        res.get("standards_warnings",[]),
        res.get("risk_register",{}).get("risks",[]),
        res.get("capacity",{}),
        res.get("kpis",{}),
        res.get("kpi_targets",{}),
    )
    st.download_button("Download Full Report (PDF)", data=pdf_buf,
                       file_name=f"{datetime.now():%Y-%m-%d}_AI_Agent_NoBOM_Report.pdf", mime="application/pdf")
