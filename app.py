# LCE + 5S Manufacturing System & Supply Chain Decision Support — AI Agent (no-BOM)
# Authors: Dr. J. Isabel Méndez & Dr. Arturo Molina
# Notes:
# - Bounded AI AGENT (planner → tools → reflector) that USES an LLM.
# - Not "agentic AI" (no autonomous long-horizon self-planning/memory).
# - Option B: KPI inputs are used ONLY if user ticks "Use KPI inputs".
# - Restores System Type → LCE Stage selection and Stage Views (Function/Org/Info/Resource/Performance).

import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
from fpdf import FPDF
from io import BytesIO
from datetime import datetime
import json, re, math

# ------------------------ App + API setup ------------------------
st.set_page_config(page_title="LCE + 5S Manufacturing Decision Support (AI Agent)", layout="wide")
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

# --------------- LCE Actions Taxonomy ---------------
lce_actions_taxonomy = {
    "Product Transfer": [
        "Ideation: Identify product BOM, materials, and quality requirements.",
        "Basic Development: Define manufacturing and quality control specs for cost, volume, and delivery.",
        "Advanced Development: Evaluate/select suppliers based on capabilities, capacity, compliance.",
        "Launching: Manage in-house assembly, document and check quality controls, train workforce.",
        "End-of-Life: Plan for disassembly, recycling, reuse; manage contract closure and reverse logistics."
    ],
    "Technology Transfer": [
        "Ideation: Capture technical specs (geometry, materials, batch size) for new component/family.",
        "Basic Development: Specify/select new manufacturing technology, equipment, and supporting tools.",
        "Advanced Development: Develop process plan, define control documentation, SOPs.",
        "Launching: Set up, test, and ramp-up equipment; optimize production.",
        "End-of-Life: Retire/adapt equipment; document performance and failures; involve digital twins for lifecycle learning."
    ],
    "Facility Design": [
        "Ideation: Specify new product requirements and associated manufacturing processes.",
        "Basic Development: Select manufacturing systems and equipment for facility.",
        "Advanced Development: Design shop-floor layout, define capacity and manufacturing strategy.",
        "Launching: Build/install/ramp; evaluate.",
        "End-of-Life: Audit facility for reuse/decommission/transform."
    ]
}

# --------------- Supply Chain Recommendations ---------------
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
        if target <= 0:  # show nothing if current level is 0
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

# ------------------------ Stage Views helpers ------------------------
def build_context_block(objective, system_type, industry, five_s_levels, selected_stages, plan_text=""):
    stages_list = [s.split(":")[0].strip() for s in selected_stages]
    five_s_str = ", ".join([f"{k}: {v}" for k, v in five_s_levels.items()])
    return (
        "PROJECT CONTEXT\n"
        f"- Objective: {objective}\n"
        f"- System Type: {system_type}\n"
        f"- Industry: {industry}\n"
        f"- 5S Levels: {five_s_str}\n"
        f"- Selected LCE Stages: {', '.join(stages_list) if stages_list else 'None'}\n\n"
        "Recent plan (if any):\n"
        f"{plan_text}\n"
    )

def build_stage_views_prompt_json(context_block, selected_stages):
    stages = [s.split(":")[0].strip() for s in selected_stages]
    stages_str = ", ".join(stages) if stages else "None"
    return (
        f"{context_block}\n\n"
        "Return ONLY a JSON object (no prose) with this schema:\n\n"
        "{\n"
        '  "stages": [\n'
        "    {\n"
        f'      "name": "<one of: {stages_str}>",\n'
        '      "Analysis": { "Function": "..." },\n'
        '      "Synthesis": {\n'
        '        "Organization": "...",\n'
        '        "Information":  "...",\n'
        '        "Resource":     "..."\n'
        "      },\n"
        '      "Evaluation": { "Performance": "..." }\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        f"- Include EVERY stage exactly once (for: {stages_str}).\n"
        "- Use plain strings (no markdown).\n"
        "- Keep fields concise but concrete.\n"
    )

def parse_stage_views_json(llm_response, selected_stages):
    m = re.search(r"\{.*\}", llm_response, flags=re.DOTALL)
    payload = m.group(0) if m else llm_response
    data = json.loads(payload)
    stage_keys = [s.split(":")[0].strip() for s in selected_stages]
    out = {}
    for item in data.get("stages", []):
        name = (item.get("name") or "").strip()
        if name in stage_keys:
            A = item.get("Analysis", {}) or {}
            S = item.get("Synthesis", {}) or {}
            E = item.get("Evaluation", {}) or {}
            out[name] = {
                "Function":     (A.get("Function")     or "").strip(),
                "Organization": (S.get("Organization") or "").strip(),
                "Information":  (S.get("Information")  or "").strip(),
                "Resource":     (S.get("Resource")     or "").strip(),
                "Performance":  (E.get("Performance")  or "").strip(),
            }
    return out

def parse_stage_views_from_plan(plan_text, selected_stages):
    stage_keys = [s.split(":")[0].strip() for s in selected_stages]
    out = {k: {"Function":"","Organization":"","Information":"","Resource":"","Performance":""} for k in stage_keys}
    union = "|".join(re.escape(k) for k in stage_keys) if stage_keys else ""
    if not union:
        return out
    parts = re.split(rf"(?:^\[({union})\]\s*|^({union})\s*:)", plan_text or "", flags=re.MULTILINE)
    i = 1
    while i < len(parts):
        name = (parts[i] or parts[i+1] or "").strip()
        body = parts[i+2] if i+2 < len(parts) else ""
        if name in out:
            def grab(label):
                m = re.search(rf"{label}\s*:\s*(.*?)(?:(?:\n[A-Z][A-Za-z \-/]+?\s*:)|\Z)", body, flags=re.DOTALL)
                return (m.group(1).strip() if m else "")
            out[name]["Function"]     = grab("Function")
            out[name]["Organization"] = grab("Organization")
            out[name]["Information"]  = grab("Information")
            out[name]["Resource"]     = grab("Resource")
            out[name]["Performance"]  = grab("Performance")
        i += 3
    return out

# ------------------------ Agent state ------------------------
class AgentState:
    def __init__(self, objective, system_type, industry, role, five_s_levels, selected_stages,
                 demand_info, docs_text, kpi_inputs, kpi_targets, use_kpi_inputs: bool):
        self.objective = objective
        self.system_type = system_type
        self.industry = industry
        self.role = role
        self.five_s_levels = five_s_levels
        self.selected_stages = selected_stages
        self.demand_info = demand_info or {}
        self.docs_text = docs_text or ""
        self.kpi_inputs = kpi_inputs or {}
        self.kpis = {}
        self.kpi_targets = kpi_targets or {}
        self.use_kpi_inputs = bool(use_kpi_inputs)
        self.spec_data = {}
        self.observations = []

# ------------------------ Planner ------------------------
def plan_with_llm(state: AgentState, evidence: str) -> str:
    lce_txt = "\n".join([f"- {s}" for s in state.selected_stages]) if state.selected_stages else "None"
    s_txt  = "\n".join([f"- {d}: L{v}" for d, v in state.five_s_levels.items()])
    dem_js = json.dumps(state.demand_info, ensure_ascii=False)
    doc_snip = (state.docs_text or "")[:TXT_LIMIT]
    prompt = (
        f"You are an expert in {state.system_type} manufacturing systems for the {state.industry} industry.\n\n"
        "IMPORTANT INSTRUCTIONS:\n"
        "- Scope: manufacturing system typology, LCE stages, 5S (Social, Sustainable, Sensing, Smart, Safe), with supply chain & Industry 5.0.\n"
        "- Never invent data; use the evidence only.\n"
        "- Keep responses concise, precise, and actionable.\n\n"
        f"Company objective: {state.objective}\n"
        f"User role: {state.role}\n\n"
        "Selected LCE stages:\n"
        f"{lce_txt}\n"
        "Current 5S maturity:\n"
        f"{s_txt}\n"
        f"Demand/capacity info (if any, JSON): {dem_js}\n"
        "Relevant document snippets (if any):\n"
        "<<<DOC>>>\n"
        f"{doc_snip}\n"
        "<<<END-DOC>>>\n\n"
        "Use the evidence strictly:\n"
        f"{evidence}\n\n"
        "If capacity observations were provided, include explicit takt time and the number of parallel stations "
        "in the configuration, layout, staffing, and scheduling decisions.\n\n"
        "Return the following sections (markdown, concise and actionable):\n"
        "[Supply Chain Configuration & Action Plan]\n"
        "[Improvement Opportunities & Risks]\n"
        "[Digital/AI Next Steps]\n"
        "[Expected 5S Maturity]\n"
        "Provide EXACTLY these five lines using integers 0–4 (no prose, no letters):\n"
        "Social: <0-4>\n"
        "Sustainable: <0-4>\n"
        "Sensing: <0-4>\n"
        "Smart: <0-4>\n"
        "Safe: <0-4>\n"
    )
    return llm(
        [
            {"role": "system", "content": "You are a digital manufacturing systems expert."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        seed=42,
    )

# ------------------------ Tools ------------------------
class Tool:
    name = "base"
    def run(self, state: AgentState) -> dict:
        return {"tool": self.name, "summary": "noop", "data": {}}

class SpecExtractorLLM(Tool):
    name = "spec_extractor"
    def run(self, state):
        prompt = (
            "Extract structured requirements from the objective and LCE stages.\n\n"
            f"Objective: {state.objective}\n"
            f"LCE stages: {', '.join([s.split(':')[0] for s in state.selected_stages]) or 'None'}\n"
            f"Industry: {state.industry}\n\n"
            "Return JSON ONLY:\n"
            "{\n"
            ' "requirements": ["<short bullets>"],\n'
            ' "constraints": ["<standards, compliance, budget, quality if implied>"],\n'
            ' "success_metrics": ["<KPI targets like: OEE >= 75%, Lead time <= 10 days, CO2e <= 2 kg/unit>"]\n'
            "}\n"
        )
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
        text_block = state.docs_text[:TXT_LIMIT]
        prompt = (
            "From the following text, extract constraints, required standards, safety/environmental notes, "
            "and any numeric parameters.\n"
            "Return JSON ONLY with keys: constraints, standards, safety, parameters.\n"
            "TEXT:\n"
            "<<<DOC>>>\n" + text_block + "\n<<<END-DOC>>>"
        )
        txt = llm(
            [
                {"role":"system","content":"You extract constraints from long docs with concise JSON."},
                {"role":"user","content":prompt}
            ],
            temperature=0.2
        )
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
        # Throughput (units/h) — only when there is demand
        days = d.get("days_per_week", 5)
        hours = d.get("hours_per_shift", 8.0) * d.get("shifts_per_day", 1)
        weekly_output = d.get("weekly_output", 0)
        denom = (days * hours)
        if weekly_output and weekly_output > 0 and denom > 0:
            out["throughput_units_per_h"] = weekly_output / denom
        else:
            out["throughput_units_per_h"] = None

        # If KPI inputs not enabled, skip details
        if not state.use_kpi_inputs:
            out["OEE_pct"] = None; out["FPY_pct"] = None
            out["energy_kWh_per_unit"] = None; out["co2e_kg_per_unit"] = None; out["water_L_per_unit"] = None
            out["changeover_min"] = None
            state.kpis = out
            return {"tool": self.name, "summary": "KPI metrics skipped (user did not enable KPI inputs)", "data": out}

        # OEE/FPY
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
            out["OEE_pct"] = oee; out["FPY_pct"] = fpy
        else:
            out["OEE_pct"] = None; out["FPY_pct"] = None

        # ESG/unit
        if weekly_output and weekly_output>0:
            out["energy_kWh_per_unit"] = (ki.get("energy_kwh_week") or 0.0) / weekly_output if ki.get("energy_kwh_week") else None
            out["co2e_kg_per_unit"] = (ki.get("co2e_kg_week") or 0.0) / weekly_output if ki.get("co2e_kg_week") else None
            out["water_L_per_unit"] = (ki.get("water_l_week") or 0.0) / weekly_output if ki.get("water_l_week") else None
        else:
            out["energy_kWh_per_unit"] = None; out["co2e_kg_per_unit"] = None; out["water_L_per_unit"] = None

        out["changeover_min"] = ki.get("changeover_min") or None
        state.kpis = out
        return {"tool": self.name, "summary": "KPI metrics computed", "data": out}

class StandardsGate(Tool):
    name = "standards_gate"
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
    KPI_KEY_MAP = {"oee":"OEE_pct","fpy":"FPY_pct","service level":"service_level_pct","fill rate":"service_level_pct",
                   "lead time":"lead_time_days","co2":"co2e_kg_per_unit","co2e":"co2e_kg_per_unit","energy":"energy_kWh_per_unit"}

    def _parse_kpi_targets(self, text: str, success_metrics:list) -> dict:
        targets = {}
        blob = (text or "") + "\n" + "\n".join(success_metrics or [])
        for m in re.finditer(r"\b(OEE|FPY|service level|fill rate)\b.*?(>=|≥|>|=)\s*([0-9]+(?:\.[0-9]+)?)\s*%?", blob, re.IGNORECASE):
            key = self.KPI_KEY_MAP[m.group(1).lower()]; val = float(m.group(3))
            if val <= 1.0: val *= 100.0
            targets[key] = (">=", val)
        for m in re.finditer(r"\b(lead time)\b.*?(<=|≤|<|=)\s*([0-9]+(?:\.[0-9]+)?)\s*(d|day|days)\b", blob, re.IGNORECASE):
            targets["lead_time_days"] = ("<=", float(m.group(3)))
        for m in re.finditer(r"\b(CO2e?|carbon)\b.*?(<=|≤|<|=)\s*([0-9]+(?:\.[0-9]+)?)\s*kg(?:/unit|\s*per\s*unit)?\b", blob, re.IGNORECASE):
            targets["co2e_kg_per_unit"] = ("<=", float(m.group(3)))
        for m in re.finditer(r"\benergy\b.*?(<=|≤|<|=)\s*([0-9]+(?:\.[0-9]+)?)\s*kWh(?:/unit|\s*per\s*unit)?\b", blob, re.IGNORECASE):
            targets["energy_kWh_per_unit"] = ("<=", float(m.group(2)))
        return targets

    def _eval_kpi_targets(self, kpis: dict, targets: dict) -> list:
        warns = []
        for key, (op, tval) in targets.items():
            kval = kpis.get(key)
            if kval is None:
                warns.append(f"KPI target declared but cannot be evaluated: {key} (no data)")
                continue
            ok = (kval >= tval) if op==">=" else (kval <= tval) if op=="<=" else abs(kval-tval)<1e-6
            if not ok:
                warns.append(f"KPI target not met: {key} {op} {tval} (observed {kval:.3g})")
        return warns

    def run(self, state):
        text = getattr(state, "_latest_plan_text", ""); ev = getattr(state, "_latest_evidence", "")
        scope = (text + "\n" + ev); warnings = []
        has_demand = bool(state.demand_info.get("weekly_output") and state.demand_info.get("cycle_time_sec"))
        parsed_targets = self._parse_kpi_targets(text, (state.spec_data or {}).get("success_metrics", []))
        merged_targets = dict(parsed_targets)
        for k, v in (state.kpi_targets or {}).items():
            if v: merged_targets[k] = ((">=" if k in ("OEE_pct","FPY_pct","service_level_pct") else "<="), float(v))
        has_kpi_values = any(v is not None for v in (state.kpis or {}).values())
        enable = {
            "quality_system": True,
            "environment": state.five_s_levels.get("Sustainable",0) >= 2,
            "ohs": state.five_s_levels.get("Safe",0) >= 1,
            "machine_safety": state.five_s_levels.get("Smart",0) >= 2,
            "cybersecurity": (state.five_s_levels.get("Smart",0)>=3 or state.five_s_levels.get("Sensing",0)>=3),
            "traceability": state.system_type in ("Product Transfer","Technology Transfer"),
            "interoperability": state.five_s_levels.get("Sensing",0)>=3,
            "capacity_ref": has_demand,
            "kpi_ref": has_kpi_values or bool(merged_targets),
        }
        for key, regex in self.RULES:
            if enable.get(key) and not re.search(regex, scope, flags=re.IGNORECASE):
                warnings.append(f"Missing reference: {key.replace('_',' ').title()} (expected pattern: {regex})")
        warnings += self._eval_kpi_targets(state.kpis or {}, merged_targets)
        state.kpi_targets = {k: v[1] for k, v in merged_targets.items()}
        return {"tool": self.name, "summary": "Standards Gate: OK" if not warnings else "Standards Gate: WARN", "data": {"warnings": warnings}}

class RiskRegisterLLM(Tool):
    name = "risk_register"
    def run(self, state):
        prompt = (
            "Produce a compact risk register (FMEA-lite) given:\n"
            f"Objective: {state.objective}\n"
            f"Industry: {state.industry}\n"
            f"System type: {state.system_type}\n"
            f"LCE stages: {', '.join([s.split(':')[0] for s in state.selected_stages]) or 'None'}\n\n"
            "Return JSON ONLY:\n"
            "{\n"
            ' "risks":[{"name":"...","cause":"...","effect":"...","mitigation":"...","owner":"Role","stage":"<LCE>"}],\n'
            ' "notes":""\n'
            "}\n"
            "Limit to 6–10 highest-impact items.\n"
        )
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
def agent_run(objective, system_type, industry, role, selected_stages, five_s_levels,
              demand_info: dict, docs_text: str, kpi_inputs: dict, kpi_targets: dict, use_kpi_inputs: bool,
              enable_agent=True, auto_iterate=True, max_steps=3):
    state = AgentState(objective, system_type, industry, role, five_s_levels, selected_stages,
                       demand_info, docs_text, kpi_inputs, kpi_targets, use_kpi_inputs)
    evidence = retrieve_domain_evidence(system_type, industry, selected_stages, five_s_levels)
    plan_text = ""; standards_warnings = []

    steps = 1 if not enable_agent else max_steps
    observations = []
    for step in range(steps):
        plan_text = plan_with_llm(state, evidence)
        state._latest_plan_text = plan_text; state._latest_evidence = evidence
        sc_text = parse_section(plan_text, "Supply Chain Configuration & Action Plan") or plan_text
        gaps = find_5s_gaps(sc_text, five_s_levels)

        run_tools = []
        if step == 0: run_tools.append("spec_extractor")
        if docs_text.strip() and step == 0: run_tools.append("doc_intake")
        if demand_info.get("weekly_output") and demand_info.get("cycle_time_sec"): run_tools.append("capacity_calc")
        run_tools.append("kpi_eval"); run_tools.append("standards_gate")
        if step == 0: run_tools.append("risk_register")

        observations = []
        for name in run_tools:
            out = TOOL_REGISTRY[name].run(state)
            observations.append(out)
            if name == "standards_gate":
                standards_warnings = out["data"].get("warnings", [])

        passed_standards = (len(standards_warnings) == 0); passed_5s = (len(gaps) == 0)
        if not enable_agent or not auto_iterate:
            evidence += "\n\n[OBSERVATIONS]\n" + "\n".join([f"- {o['tool']}: {o['summary']}" for o in observations]); break
        if passed_standards and passed_5s:
            evidence += "\n\n[OBSERVATIONS]\n" + "\n".join([f"- {o['tool']}: {o['summary']}" for o in observations]); break
        critic = []
        if not passed_standards: critic.append("STANDARDS FINDINGS:\n- " + "\n- ".join(standards_warnings[:10]))
        if not passed_5s: critic.append("5S COVERAGE FINDINGS:\n- " + "\n- ".join(gaps[:10]))
        cap = next((o for o in observations if o["tool"] == "capacity_calc"), None)
        if cap and cap.get("data", {}).get("stations"):
            s = cap["data"]
            critic.append(f"CAPACITY SUGGESTION:\n- Target stations: {s['stations']} (takt≈{s['takt_sec']:.1f}s). Reflect in layout, staffing, and scheduling.")
        evidence += "\n\n[NEEDS FIX — ADDRESS IN NEXT PASS]\n" + "\n\n".join(critic)

    # ---- Stage Views (second LLM pass, using plan_text as context) ----
    context_block = build_context_block(objective, system_type, industry, five_s_levels, selected_stages,
                                        parse_section(plan_text, "Supply Chain Configuration & Action Plan"))
    stage_prompt = build_stage_views_prompt_json(context_block, selected_stages)
    stage_views_raw = llm([{"role":"system","content":"You are a digital manufacturing systems expert."},
                           {"role":"user","content":stage_prompt}], temperature=0.1)
    stage_views = {}
    try:
        stage_views = parse_stage_views_json(stage_views_raw, selected_stages)
    except Exception:
        stage_views = {}
    if (not stage_views or not any(v for v in stage_views.values())) and plan_text:
        stage_views = parse_stage_views_from_plan(plan_text, selected_stages)

    risk = next((o for o in observations if o["tool"] == "risk_register"), {"data": {"risks": []}})
    cap = next((o for o in observations if o["tool"] == "capacity_calc"), {"data": {}})

    return {
        "plan_text": plan_text,
        "stage_views": stage_views,
        "standards_warnings": standards_warnings,
        "risk_register": risk["data"],
        "capacity": cap.get("data", {}),
        "kpis": state.kpis,
        "kpi_targets": state.kpi_targets,
    }

# ------------------------ UI ------------------------

# 1) Objective / Industry / Role
st.header("Define Scenario")
objective = st.text_input("Objective (e.g., launch new product, adopt a new process, expand a facility):",
                          value=st.session_state.get("objective","Design and ramp a flexible small manufacturing cell."),
                          key="objective")
industry = st.selectbox("Industry:", ["Automotive","Electronics","Medical Devices","Consumer Goods","Other"],
                        index=st.session_state.get("industry_idx",1), key="industry")
st.session_state["industry_idx"] = ["Automotive","Electronics","Medical Devices","Consumer Goods","Other"].index(industry)
role_options = ["Design Engineer","Process Engineer","Manufacturing Engineer","Safety Supervisor",
                "Sustainability Manager","Supply Chain Analyst","Manager/Decision Maker","Other"]
role_selected = st.selectbox("Your role:", role_options,
                             index=st.session_state.get("role_idx",2), key="user_role")
if role_selected == "Other":
    role_selected = st.text_input("Please specify your role:", value=st.session_state.get("custom_role",""), key="custom_role") or "Other"
st.session_state["role_idx"] = role_options.index("Other") if role_selected=="Other" else role_options.index(role_selected)

# 2) Manufacturing System Type
st.header("Select Manufacturing System Type")
system_types = ["Product Transfer","Technology Transfer","Facility Design"]
system_type = st.radio("Manufacturing system type:", system_types, key="system_type")

# 3) Select LCE Stages/Actions
st.header("Select Relevant LCE Stages/Actions")
lce_global_keys = ["Ideation","Basic Development","Advanced Development","Launching","End-of-Life"]
if "lce_global_checked" not in st.session_state:
    st.session_state["lce_global_checked"] = [False]*len(lce_global_keys)

lce_actions = lce_actions_taxonomy[system_type]
selected_stages=[]
for i, action in enumerate(lce_actions):
    action_key = action.split(":")[0].strip()
    idx = lce_global_keys.index(action_key)
    checked = st.checkbox(action, value=st.session_state["lce_global_checked"][idx], key=f"lce_{i}")
    st.session_state["lce_global_checked"][idx] = checked
    if checked: selected_stages.append(action)
st.session_state["selected_stages"] = selected_stages

# 4) 5S Maturity
st.header("Current 5S Maturity (one per S)")
five_s_levels={}
cols = st.columns(5)
for i, dim in enumerate(["Social","Sustainable","Sensing","Smart","Safe"]):
    with cols[i]:
        options=[f"Level {idx}: {lvl['desc']}" for idx,lvl in enumerate(five_s_taxonomy[dim])]
        sel = st.radio(dim, options, index=0, key=f"{dim}_radio")
        five_s_levels[dim] = options.index(sel)
        techs = five_s_taxonomy[dim][five_s_levels[dim]].get("tech", [])
        if techs: st.caption("Tech hints: " + "; ".join(techs))

# ---------------- MASTER OPS TOGGLE (single, before any Demand/KPI UI) ----------------
ops_enabled = st.toggle(
    "Enable operations inputs (Demand/Capacity + KPI)",
    value=False,
    help="Turn on to provide demand/capacity and (optional) KPI inputs & targets.",
    key="ops_toggle",
)

# Safe defaults so later code never breaks when the module is OFF
use_kpi_inputs = False
scheduled_time_h = runtime_h = ideal_cycle_time_s = 0.0
total_count = good_count = 0
changeover_min = 0.0
energy_kwh_week = co2e_kg_week = water_l_week = 0.0
tgt_oee = tgt_fpy = tgt_service = 0.0
tgt_lead_time = tgt_energy = tgt_co2e = 0.0

# Defaults for demand (used by capacity tool even when hidden)
demand_info = {
    "weekly_output": 0,
    "cycle_time_sec": 0.0,
    "shifts_per_day": 1,
    "hours_per_shift": 8.0,
    "days_per_week": 5,
    "oee": 0.70,
}

# ================== Only render Demand/KPI UI when enabled ==================
if ops_enabled:
    # 5a) Demand / Capacity
    st.header("Demand / Capacity Inputs (optional but recommended)")
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: demand_info["weekly_output"]   = st.number_input("Target weekly output (units)", min_value=0, value=0)
    with c2: demand_info["cycle_time_sec"]  = st.number_input("Cycle time per unit (sec)", min_value=0.0, value=0.0, step=1.0)
    with c3: demand_info["shifts_per_day"]  = st.number_input("Shifts per day", min_value=1, max_value=4, value=1)
    with c4: demand_info["hours_per_shift"] = st.number_input("Hours per shift", min_value=1.0, max_value=12.0, value=8.0, step=0.5)
    with c5: demand_info["days_per_week"]   = st.number_input("Days per week", min_value=1, max_value=7, value=5)
    demand_info["oee"] = st.slider("Assumed OEE for capacity calc (if unknown)", min_value=0.3, max_value=0.95, value=0.7, step=0.05)

    # 5b) KPI Inputs
    st.subheader("Optional KPI Inputs (for OEE/FPY/ESG)")
    with st.expander("Enter KPI inputs if available", expanded=False):
        use_kpi_inputs = st.checkbox("Use KPI inputs", value=False, key="use_kpi_inputs")
        e1, e2, e3 = st.columns(3)
        with e1:
            scheduled_time_h   = st.number_input("Scheduled Time (h/week)", 0.0, 200.0, 40.0, 0.5)
            runtime_h          = st.number_input("Runtime (h/week)",        0.0, 200.0, 36.0, 0.5)
            ideal_cycle_time_s = st.number_input("Ideal Cycle Time (s/unit)", 0.0, 3600.0, 60.0, 1.0)
        with e2:
            total_count    = st.number_input("Units Produced (total)", 0, 1_000_000, 0, 1)
            good_count     = st.number_input("Good Units",             0, 1_000_000, 0, 1)
            changeover_min = st.number_input("Avg Changeover (min)",   0.0, 600.0, 0.0, 1.0)
        with e3:
            energy_kwh_week = st.number_input("Energy (kWh/week)", 0.0, 1e9, 0.0, 1.0)
            co2e_kg_week    = st.number_input("CO₂e (kg/week)",    0.0, 1e9, 0.0, 1.0)
            water_l_week    = st.number_input("Water (L/week)",    0.0, 1e12, 0.0, 1.0)

    # 5c) KPI Targets
    st.subheader("Optional KPI Targets (used by KPI Gate)")
    with st.expander("Set targets to enable pass/fail checks", expanded=False):
        t1, t2, t3 = st.columns(3)
        with t1:
            tgt_oee     = st.number_input("Target OEE (%)", 0.0, 100.0, 0.0, 0.5)
            tgt_fpy     = st.number_input("Target FPY (%)", 0.0, 100.0, 0.0, 0.5)
            tgt_service = st.number_input("Target Service Level / Fill Rate (%)", 0.0, 100.0, 0.0, 0.5)
        with t2:
            tgt_lead_time = st.number_input("Target Lead Time (days) — lower is better", 0.0, 365.0, 0.0, 0.5)
            tgt_energy    = st.number_input("Target Energy (kWh/unit) — lower is better", 0.0, 1e6, 0.0, 0.01)
            tgt_co2e      = st.number_input("Target CO₂e (kg/unit) — lower is better", 0.0, 1e6, 0.0, 0.01)
        with t3:
            pass  # reserved

# Build dicts used by tools (exist even if ops_enabled=False)
kpi_inputs = dict(
    scheduled_time_h=scheduled_time_h, runtime_h=runtime_h, ideal_cycle_time_s=ideal_cycle_time_s,
    total_count=total_count, good_count=good_count, changeover_min=changeover_min,
    energy_kwh_week=energy_kwh_week, co2e_kg_week=co2e_kg_week, water_l_week=water_l_week,
)
kpi_targets = {
    "OEE_pct":             tgt_oee if tgt_oee > 0 else None,
    "FPY_pct":             tgt_fpy if tgt_fpy > 0 else None,
    "service_level_pct":   tgt_service if tgt_service > 0 else None,
    "lead_time_days":      tgt_lead_time if tgt_lead_time > 0 else None,
    "energy_kWh_per_unit": tgt_energy if tgt_energy > 0 else None,
    "co2e_kg_per_unit":    tgt_co2e if tgt_co2e > 0 else None,
}

st.header("Optional Upload relevant docs (manuals/specs/SOPs)")
uploads = st.file_uploader("Upload .txt/.md/.csv/.log/.docx/.pdf (max a few MB). Multiple allowed.",
                           type=["txt","md","csv","log","docx","pdf"], accept_multiple_files=True)
docs_text = ""
if uploads:
    pieces=[]
    for f in uploads:
        text = try_extract_text(f)
        if text: pieces.append(f"# {f.name}\n{text}")
    docs_text = "\n\n".join(pieces)

# --------------- Run agent ---------------
# Agent is always on; no UI toggles
enable_agent = True
auto_iterate = True

if st.button("Generate Plan & Recommendations"):
    with st.spinner("Agent planning, executing tools, stage views, and refining..."):
        result = agent_run(
            objective=objective,
            system_type=system_type,
            industry=industry,
            role=role_selected,
            selected_stages=selected_stages,
            five_s_levels=five_s_levels,
            demand_info=demand_info,
            docs_text=docs_text,
            kpi_inputs=kpi_inputs,
            kpi_targets=kpi_targets,
            use_kpi_inputs=use_kpi_inputs,
            enable_agent=enable_agent,   # always True
            auto_iterate=auto_iterate,   # always True
            max_steps=3,
        )
        st.session_state["agent_result"] = result

# ------------------------ Results ------------------------
res = st.session_state.get("agent_result")
ops_enabled_state = st.session_state.get("ops_toggle", False)   # read the master toggle

if res:
    st.header("Results")
    plan_text = res["plan_text"]

    # 7a) Stage Views — System Type + Selected LCE Stages rendered
    st.subheader("Supply Chain Configuration & Action Plan (by LCE Stage)")
    displayed_stages = st.session_state.get("selected_stages", [])
    stage_views = res.get("stage_views", {}) or {}
    if displayed_stages:
        st.markdown("<ul style='margin-top:0; margin-bottom:0;'>", unsafe_allow_html=True)
        for action in displayed_stages:
            stage, desc = (action.split(":",1)+[""])[:2]
            stage_key = stage.strip(); desc = desc.strip()
            views = stage_views.get(stage_key, {})
            st.markdown(f"<li><b>Stage → {stage_key}</b>: {desc}", unsafe_allow_html=True)
            st.markdown("<ul style='margin-top:0; margin-bottom:0;'>", unsafe_allow_html=True)
            groups = [
                ("Engineering Analysis", [("Function","Function")]),
                ("Engineering Synthesis", [("Organization","Organization"),("Information","Information"),("Resource","Resource")]),
                ("Engineering Evaluation", [("Performance","Performance")]),
            ]
            for heading, items in groups:
                present=[]
                for label,key in items:
                    value = (views.get(key) or "").strip()
                    if value: present.append((label,value))
                if present:
                    st.markdown(f"<li><b>{heading}</b>", unsafe_allow_html=True)
                    st.markdown("<ul style='margin-top:0; margin-bottom:0;'>", unsafe_allow_html=True)
                    for label,value in present:
                        st.markdown(f"<li><i>{label}:</i> {value}</li>", unsafe_allow_html=True)
                    st.markdown("</ul></li>", unsafe_allow_html=True)
            st.markdown("</ul></li>", unsafe_allow_html=True)
        st.markdown("</ul>", unsafe_allow_html=True)
    else:
        st.info("No LCE stage activities selected.")

    # 7b) Also show the LLM sections verbatim (for completeness)
    st.subheader("Plan (raw sections)")
    st.markdown("**Supply Chain Configuration & Action Plan**")
    st.info(parse_section(plan_text,"Supply Chain Configuration & Action Plan") or "—")
    st.markdown("**Improvement Opportunities & Risks**")
    st.info(parse_section(plan_text,"Improvement Opportunities & Risks") or "—")
    st.markdown("**Digital/AI Next Steps**")
    st.info(parse_section(plan_text,"Digital/AI Next Steps") or "—")
    st.markdown("**Expected 5S Maturity**")
    exp_raw = parse_section(plan_text, "Expected 5S Maturity")
    expected_5s = extract_expected_levels(exp_raw, five_s_levels)  # clamps 0–4
    exp_lines = "\n".join([f"{d}: {expected_5s[d]}" for d in ["Social","Sustainable","Sensing","Smart","Safe"]])
    st.info(exp_lines)

    # 8) 5S Profiles
    st.header("5S Profiles")
    curr_df = pd.DataFrame({"Dimension": list(five_s_levels.keys()), "Level":[five_s_levels[k] for k in five_s_levels]})
    fig_curr = px.line_polar(curr_df, r="Level", theta="Dimension", line_close=True, range_r=[0,4])
    st.plotly_chart(fig_curr, use_container_width=True)
    if expected_5s:
        exp_df = pd.DataFrame({"Dimension": list(expected_5s.keys()), "Level": [expected_5s[k] for k in expected_5s]})
        fig_exp = px.line_polar(exp_df, r="Level", theta="Dimension", line_close=True, range_r=[0, 4])
        st.plotly_chart(fig_exp, use_container_width=True)

    # 9) Capacity (hide unless ops are enabled)
    cap = res.get("capacity", {})
    if ops_enabled_state and cap and cap.get("takt_sec") is not None:
        st.header("Capacity Sizing (from demand inputs)")
        st.success(f"Computed takt ≈ {cap.get('takt_sec',0):.1f} s | Required parallel stations: {cap.get('stations','?')}")
        st.caption("The agent feeds these into the planner; the plan should echo them in layout/staffing.")

    # 10) KPI Panel (only if operations module is enabled)
    if ops_enabled_state:
        kpis = res.get("kpis", {}); targets = res.get("kpi_targets", {})
        any_kpi_input = use_kpi_inputs and any([
            scheduled_time_h>0, runtime_h>0, ideal_cycle_time_s>0,
            total_count>0, good_count>0, changeover_min>0,
            energy_kwh_week>0, co2e_kg_week>0, water_l_week>0,
        ])
        any_kpi_target = any(v for v in (targets or {}).values())
        computed_has_kpis = any(v is not None for v in (kpis or {}).values())

        if any_kpi_input or any_kpi_target or computed_has_kpis:
            st.header("KPI Panel")
            def tidy(metric_key, label, unit):
                val = kpis.get(metric_key); tgt = targets.get(metric_key); status = "—"
                if val is None and tgt: status = "no data"
                elif val is not None and tgt:
                    up_or_down = metric_key in ("OEE_pct","FPY_pct","service_level_pct")
                    status = "OK" if ((val >= tgt) if up_or_down else (val <= tgt)) else "NOT MET"
                return {"Metric": label,
                        "Value": None if val is None else (f"{val:.3g} {unit}" if unit else f"{val:.3g}"),
                        "Target": None if not tgt else (f"{tgt:.3g} {unit}" if unit else f"{tgt:.3g}"),
                        "Status": status}
            rows = [
                tidy("throughput_units_per_h","Throughput","units/h"),
                tidy("OEE_pct","OEE","%"),
                tidy("FPY_pct","FPY","%"),
                tidy("changeover_min","Changeover","min"),
                tidy("energy_kWh_per_unit","Energy / Unit","kWh"),
                tidy("co2e_kg_per_unit","CO₂e / Unit","kg"),
                tidy("water_L_per_unit","Water / Unit","L"),
                tidy("lead_time_days","Lead Time","days"),
                tidy("service_level_pct","Service Level / Fill Rate","%"),
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # 11) Standards Gate (only when ops are enabled)
            warns = res.get("standards_warnings", [])
            if warns or any_kpi_input or any_kpi_target:
                st.header("Standards Gate")
                if warns:
                    st.error("Standards/compliance warnings:\n- " + "\n- ".join([str(w) for w in warns]))
                else:
                    st.success("Standards Gate: OK")
    # 12) Risk Register (only if exists)
    risks = res.get("risk_register",{}).get("risks",[])
    if risks:
        st.header("Risk Register")
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
    pdf.cell(0,10,to_latin1("LCE + 5S Manufacturing Decision Support (AI Agent, no-BOM)"), ln=True, align="C")
    pdf.set_font("Arial", size=11)
    pdf.cell(0,8,to_latin1("Developed by Dr. Juana Isabel Méndez and Dr. Arturo Molina"), ln=True, align="C")
    pdf.cell(0,8,to_latin1("Date: "+datetime.now().strftime("%Y-%m-%d")), ln=True, align="C")
    pdf.ln(6)

    for head in ["Supply Chain Configuration & Action Plan","Improvement Opportunities & Risks","Digital/AI Next Steps","Expected 5S Maturity"]:
        pdf.set_font("Arial","B",12); pdf.cell(0,8,to_latin1(head), ln=True)
        pdf.set_font("Arial", size=11)
        if head == "Expected 5S Maturity":
            exp_raw = parse_section(plan_text, head)
            exp_levels = extract_expected_levels(exp_raw, five_s_levels)
            text = "\n".join([f"{d}: {exp_levels[d]}" for d in ["Social","Sustainable","Sensing","Smart","Safe"]])
        else:
            text = parse_section(plan_text, head) or "—"
        pdf.multi_cell(0,7,to_latin1(text)); pdf.ln(2)
    if cap and cap.get("takt_sec") is not None:
        pdf.set_font("Arial","B",12); pdf.cell(0,8,to_latin1("Capacity Sizing"), ln=True)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0,7,to_latin1(f"Takt ≈ {cap.get('takt_sec',0):.1f} s; Required stations: {cap.get('stations','?')}"))
    if kpis:
        pdf.ln(4); pdf.set_font("Arial","B",12); pdf.cell(0,8,to_latin1("KPI Summary"), ln=True)
        pdf.set_font("Arial", size=11)
        def line(label, key, unit):
            v = kpis.get(key); t = targets.get(key)
            vtxt = "—" if v is None else (f"{v:.3g} {unit}" if unit else f"{v:.3g}")
            ttxt = "—" if not t else (f"{t:.3g} {unit}" if unit else f"{t:.3g}")
            pdf.multi_cell(0,6,to_latin1(f"- {label}: {vtxt} | Target: {ttxt}"))
        line("Throughput","throughput_units_per_h","units/h")
        line("OEE","OEE_pct","%"); line("FPY","FPY_pct","%"); line("Changeover","changeover_min","min")
        line("Energy / Unit","energy_kWh_per_unit","kWh"); line("CO₂e / Unit","co2e_kg_per_unit","kg")
        line("Water / Unit","water_L_per_unit","L"); line("Lead Time","lead_time_days","days")
        line("Service Level","service_level_pct","%")
    if warnings:
        pdf.ln(4); pdf.set_font("Arial","B",12); pdf.cell(0,8,"Standards Gate Warnings", ln=True)
        pdf.set_font("Arial", size=11)
        for w in warnings: pdf.multi_cell(0,6,to_latin1(f"- {w}"))
    if risks:
        pdf.ln(4); pdf.set_font("Arial","B",12); pdf.cell(0,8,"Risk Register (FMEA-lite)", ln=True)
        pdf.set_font("Arial", size=11)
        for r in risks[:12]:
            line = f"- {r.get('name','')} | cause: {r.get('cause','')} | mitigation: {r.get('mitigation','')} | owner: {r.get('owner','')} | stage: {r.get('stage','')}"
            pdf.multi_cell(0,6,to_latin1(line[:220]))
    buf = BytesIO(); pdf_bytes = pdf.output(dest='S').encode('latin1')
    buf.write(pdf_bytes); buf.seek(0); return buf

res = st.session_state.get("agent_result")
if res:
    pdf_buf = generate_pdf_report(
        res.get("plan_text",""), five_s_levels, res.get("standards_warnings",[]),
        res.get("risk_register",{}).get("risks",[]), res.get("capacity",{}),
        res.get("kpis",{}), res.get("kpi_targets",{}),
    )
    st.download_button("Download Full Report (PDF)", data=pdf_buf,
                       file_name=f"{datetime.now():%Y-%m-%d}_AI_Agent_NoBOM_Report.pdf", mime="application/pdf")
