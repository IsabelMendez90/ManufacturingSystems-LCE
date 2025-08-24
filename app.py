# --- LCE + 5S Decision Support ---
# Planner → Tools → Reflector loop with standards gate, capacity sizing, doc-intake, risk register

import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
from fpdf import FPDF
from io import BytesIO
from datetime import datetime
import json, re, math

# ---- Optional docx/pdf text extractors (best-effort, safe fallbacks) ----
TXT_LIMIT = 12000  # keep LLM prompts bounded

def try_extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    # Simple text/markdown
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

# ------------------------ App + API setup ------------------------
st.set_page_config(page_title="LCE + 5S Decision Support (Agentic, no-BOM)", layout="wide")
st.title("LCE + 5S Manufacturing System & Supply Chain Decision Support — Agentic (no-BOM)")
st.markdown("Developed by: Dr. J. Isabel Méndez  & Dr. Arturo Molina")

API_KEY = st.secrets["OPENROUTER_API_KEY"]
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)

def llm(msgs, temperature=0.2, seed=42, model="mistralai/mistral-7b-instruct"):
    resp = client.chat.completions.create(model=model, temperature=temperature, seed=seed, messages=msgs)
    return resp.choices[0].message.content or ""

# ------------------------ 5S Taxonomy (same content, condensed text) ------------------------
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

# ------------------------ Supply-chain hints (kept short) ------------------------
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
REQUIRED_MATCHES_BY_LEVEL = {0:0,1:1,2:2,3:3,4:4}

def find_5s_gaps(plan_text: str, target_levels: dict) -> list[str]:
    text = plan_text or ""
    gaps=[]
    for dim,target in target_levels.items():
        if target<=0: continue
        pats = FIVE_S_PATTERNS.get(dim,[])
        hits = set(p for p in pats if re.search(p, text, re.IGNORECASE))
        need = REQUIRED_MATCHES_BY_LEVEL.get(target,1)
        if len(hits) < need:
            sample = [re.sub(r"\\b","",p) for p in pats[:5]]
            gaps.append(f"{dim}: need ≥{need} concrete references; found {len(hits)}. Add: {', '.join(sample)}")
    return gaps

# ------------------------ Evidence builder ------------------------
def retrieve_domain_evidence(system_type, industry, selected_stages, five_s_levels):
    stages = [s.split(":")[0].strip() for s in selected_stages] or []
    lines=[]
    hints = supply_chain_recommendations.get(system_type, [])
    if hints:
        lines.append("SUPPLY-CHAIN HINTS:")
        lines += [f"- {h}" for h in hints]
    if stages:
        lines.append("\nSELECTED LCE STAGES:")
        lines += [f"- {s}" for s in stages]
    lines.append("\n5S TECH HINTS @ CURRENT LEVEL:")
    for dim,lvl in five_s_levels.items():
        techs = five_s_taxonomy[dim][lvl].get("tech", [])
        if techs:
            lines.append(f"- {dim} (L{lvl}): " + "; ".join(techs))
    return "\n".join(lines)

# ------------------------ Planner / Tools / Reflector ------------------------
def parse_section(text, head):
    m = re.search(rf"\[{re.escape(head)}\](.*?)(?=\n\[[A-Z].*?\]|\Z)", text, re.DOTALL)
    return m.group(1).strip() if m else ""

def extract_expected_levels(text, fallback):
    out={}
    for dim in ["Social","Sustainable","Sensing","Smart","Safe"]:
        m = re.search(rf"{dim}\s*[:\-]?\s*(?:Level\s*)?(\d)", text, re.IGNORECASE)
        out[dim] = int(m.group(1)) if m else fallback.get(dim,0)
    return out

class AgentState:
    def __init__(self, objective, system_type, industry, five_s_levels, selected_stages, demand_info, docs_text):
        self.objective = objective
        self.system_type = system_type
        self.industry = industry
        self.five_s_levels = five_s_levels
        self.selected_stages = selected_stages
        self.demand_info = demand_info or {}
        self.docs_text = docs_text or ""
        self.observations=[]

def plan_with_llm(state: AgentState, evidence: str) -> str:
    lce_txt = "\n".join([f"- {s}" for s in state.selected_stages]) if state.selected_stages else "None"
    s_txt  = "\n".join([f"- {d}: L{v}" for d,v in state.five_s_levels.items()])
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
\"\"\"{doc_snip}\"\"\"

Use the evidence strictly:
{evidence}

Return the following sections (markdown, concise and actionable):
[Supply Chain Configuration & Action Plan]
[Improvement Opportunities & Risks]
[Digital/AI Next Steps]
[Expected 5S Maturity]  (List as: Social: X, Sustainable: Y, Sensing: Z, Smart: W, Safe: V.)
"""
    return llm([{"role":"system","content":"You are a digital manufacturing systems expert."},
                {"role":"user","content":prompt}], temperature=0.2, seed=42)

# ----- Tools -----
class Tool:
    name="base"
    def run(self, state: AgentState) -> dict:
        return {"tool": self.name, "summary": "noop", "data": {}}

class SpecExtractorLLM(Tool):
    name="spec_extractor"
    def run(self, state):
        prompt = f"""Extract structured requirements from the objective and LCE stages.

Objective: {state.objective}
LCE stages: {', '.join([s.split(':')[0] for s in state.selected_stages]) or 'None'}
Industry: {state.industry}

Return JSON ONLY:
{{
 "requirements": ["<short bullets>"],
 "constraints": ["<standards, compliance, budget, quality if implied>"],
 "success_metrics": ["<throughput/quality/sustainability/safety KPIs>"]
}}"""
        txt = llm([{"role":"system","content":"You turn free text into compact JSON specs."},
                   {"role":"user","content":prompt}], temperature=0.1)
        try:
            js = json.loads(re.search(r"\{.*\}", txt, re.DOTALL).group(0))
        except Exception:
            js = {"requirements":[],"constraints":[],"success_metrics":[]}
        return {"tool": self.name, "summary": "Structured specs extracted", "data": js}

class DocIntakeLLM(Tool):
    name="doc_intake"
    def run(self, state):
        if not state.docs_text.strip():
            return {"tool": self.name, "summary":"No documents provided.", "data":{}}
        prompt = f"""From the following text, extract constraints, required standards, safety/environmental notes, and any numeric parameters.
Return JSON ONLY with keys: constraints, standards, safety, parameters.
TEXT:
\"\"\"{state.docs_text[:TXT_LIMIT]}\"\"\""""
        txt = llm([{"role":"system","content":"You extract constraints from long docs with concise JSON."},
                   {"role":"user","content":prompt}], temperature=0.2)
        try:
            js = json.loads(re.search(r"\{.*\}", txt, re.DOTALL).group(0))
        except Exception:
            js = {"constraints":[],"standards":[],"safety":[],"parameters":{}}
        return {"tool": self.name, "summary":"Doc constraints extracted", "data": js}

class CapacityCalculator(Tool):
    name="capacity_calc"
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

class StandardsGate(Tool):
    name="standards_gate"
    # required if context suggests it; we check presence in plan/evidence
    RULES = [
        ("quality_system", r"ISO\s?9001"),
        ("environment", r"ISO\s?14001|LCA|GaBi|OpenLCA"),
        ("ohs", r"ISO\s?45001"),
        ("machine_safety", r"ISO\s?13849|IEC\s?61508|IEC\s?62061"),
        ("cybersecurity", r"IEC\s?62443|NIST"),
        ("traceability", r"APQP|SPC|traceability|QR|blockchain"),
        ("interoperability", r"OPC[-\s]?UA|MQTT"),
    ]
    def run(self, state):
        text = state._latest_plan_text if hasattr(state, "_latest_plan_text") else ""
        ev = state._latest_evidence if hasattr(state,"_latest_evidence") else ""
        scope = (text + "\n" + ev).lower()
        warnings=[]
        # Enable rules heuristically based on system type + levels
        enable = {
            "quality_system": True,
            "environment": state.five_s_levels.get("Sustainable",0) >= 2,
            "ohs": state.five_s_levels.get("Safe",0) >= 1,
            "machine_safety": state.five_s_levels.get("Smart",0) >= 2,
            "cybersecurity": (state.five_s_levels.get("Smart",0)>=3 or state.five_s_levels.get("Sensing",0)>=3),
            "traceability": state.system_type in ("Product Transfer","Technology Transfer"),
            "interoperability": state.five_s_levels.get("Sensing",0)>=3,
        }
        for key,regex in self.RULES:
            if enable.get(key):
                if not re.search(regex, scope, flags=re.IGNORECASE):
                    label = key.replace("_"," ").title()
                    warnings.append(f"Missing reference: {label} (expected pattern: {regex})")
        return {"tool": self.name, "summary": "Standards Gate: OK" if not warnings else "Standards Gate: WARN", "data":{"warnings":warnings}}

class RiskRegisterLLM(Tool):
    name="risk_register"
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
    "standards_gate": StandardsGate(),
    "risk_register":  RiskRegisterLLM(),
}

def agentic_run(objective, system_type, industry, selected_stages, five_s_levels,
                demand_info: dict, docs_text: str,
                enable_agentic=True, auto_iterate=True, max_steps=3):
    state = AgentState(objective, system_type, industry, five_s_levels, selected_stages, demand_info, docs_text)
    evidence = retrieve_domain_evidence(system_type, industry, selected_stages, five_s_levels)
    plan_text=""; standards_warnings=[]

    steps = 1 if not enable_agentic else max_steps
    for step in range(steps):
        # PLAN
        plan_text = plan_with_llm(state, evidence)
        state._latest_plan_text = plan_text
        state._latest_evidence = evidence

        sc_text = parse_section(plan_text,"Supply Chain Configuration & Action Plan") or plan_text
        gaps = find_5s_gaps(sc_text, five_s_levels)

        # EXECUTE TOOLS (context-aware)
        run_tools=[]
        if step==0: run_tools.append("spec_extractor")
        if docs_text.strip() and step==0: run_tools.append("doc_intake")
        if demand_info.get("weekly_output") and demand_info.get("cycle_time_sec"): run_tools.append("capacity_calc")
        run_tools.append("standards_gate")
        if step==0: run_tools.append("risk_register")

        observations=[]
        for name in run_tools:
            out = TOOL_REGISTRY[name].run(state)
            observations.append(out)
            if name=="standards_gate":
                standards_warnings = out["data"].get("warnings",[])

        # REFLECT
        passed_standards = (len(standards_warnings)==0)
        passed_5s = (len(gaps)==0)
        if not enable_agentic or not auto_iterate:
            evidence += "\n\n[OBSERVATIONS]\n" + "\n".join([f"- {o['tool']}: {o['summary']}" for o in observations])
            break

        if passed_standards and passed_5s:
            evidence += "\n\n[OBSERVATIONS]\n" + "\n".join([f"- {o['tool']}: {o['summary']}" for o in observations])
            break
        else:
            critic=[]
            if not passed_standards:
                critic.append("STANDARDS FINDINGS:\n- " + "\n- ".join(standards_warnings[:10]))
            if not passed_5s:
                critic.append("5S COVERAGE FINDINGS:\n- " + "\n- ".join(gaps[:10]))
            # if capacity tool ran, surface its suggestion to tighten plan
            cap = next((o for o in observations if o["tool"]=="capacity_calc"), None)
            if cap and cap.get("data",{}).get("stations"):
                s = cap["data"]
                critic.append(f"CAPACITY SUGGESTION:\n- Target stations: {s['stations']} (takt≈{s['takt_sec']:.1f}s). Reflect in layout, staffing, and scheduling.")
            evidence += "\n\n[NEEDS FIX — ADDRESS IN NEXT PASS]\n" + "\n\n".join(critic)

    # collect risk register (if generated)
    risk = next((o for o in observations if o["tool"]=="risk_register"), {"data":{"risks":[]}})
    return {"plan_text": plan_text, "standards_warnings": standards_warnings, "risk_register": risk["data"]}

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
        if techs: st.caption("Tech hints: " + "; ".join(techs))

st.header("4) Agentic Controls")
enable_agentic = st.toggle("Enable Agentic Mode (plan → tools → reflect)", value=True)
auto_iterate   = st.toggle("Auto-iterate until pass (standards + 5S coverage)", value=True)

st.header("5) Optional Demand / Capacity Inputs")
c1,c2,c3,c4,c5 = st.columns(5)
with c1: weekly_output = st.number_input("Target weekly output (units)", min_value=0, value=0)
with c2: cycle_time_sec = st.number_input("Cycle time per unit (sec)", min_value=0.0, value=0.0, step=1.0)
with c3: shifts_per_day = st.number_input("Shifts per day", min_value=1, max_value=4, value=1)
with c4: hours_per_shift = st.number_input("Hours per shift", min_value=1.0, max_value=12.0, value=8.0, step=0.5)
with c5: days_per_week = st.number_input("Days per week", min_value=1, max_value=7, value=5)
oee = st.slider("Overall Equipment Effectiveness (OEE)", min_value=0.3, max_value=0.95, value=0.7, step=0.05)
demand_info = {
    "weekly_output": weekly_output,
    "cycle_time_sec": cycle_time_sec,
    "shifts_per_day": shifts_per_day,
    "hours_per_shift": hours_per_shift,
    "days_per_week": days_per_week,
    "oee": oee,
}

st.header("6) (Optional) Upload any relevant docs (manuals/specs/SOPs)")
uploads = st.file_uploader("Upload .txt/.md/.csv/.log/.docx/.pdf (max a few MB). Multiple allowed.",
                           type=["txt","md","csv","log","docx","pdf"], accept_multiple_files=True)
docs_text = ""
if uploads:
    pieces=[]
    for f in uploads:
        text = try_extract_text(f)
        if text: pieces.append(f"# {f.name}\n{text}")
    docs_text = "\n\n".join(pieces)

# Run agent
if st.button("Generate Plan & Recommendations"):
    with st.spinner("Agent planning, executing tools, and refining..."):
        result = agentic_run(
            objective=objective,
            system_type=system_type,
            industry=industry,
            selected_stages=selected_stages,
            five_s_levels=five_s_levels,
            demand_info=demand_info,
            docs_text=docs_text,
            enable_agentic=enable_agentic,
            auto_iterate=auto_iterate,
            max_steps=3,
        )
        st.session_state["agent_result"]=result

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
    if expected_5s != five_s_levels:
        exp_df = pd.DataFrame({"Dimension": list(expected_5s.keys()), "Level":[expected_5s[k] for k in expected_5s]})
        fig_exp = px.line_polar(exp_df, r="Level", theta="Dimension", line_close=True)
        st.plotly_chart(fig_exp, use_container_width=True)

    st.header("9) Standards Gate")
    warns = res.get("standards_warnings", [])
    if warns:
        st.error("Standards/compliance warnings:\n- " + "\n- ".join(warns))
    else:
        st.success("Standards Gate: OK")

    st.header("10) Risk Register")
    risks = res.get("risk_register",{}).get("risks",[])
    if risks:
        df = pd.DataFrame(risks)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download Risk Register (CSV)", df.to_csv(index=False).encode("utf-8"),
                           "risk_register.csv","text/csv")
    else:
        st.info("No risks extracted/generated.")

# ------------------------ PDF export ------------------------
def to_latin1(text):
    if not isinstance(text,str): text=str(text)
    return text.encode('latin-1','replace').decode('latin-1')

def generate_pdf_report(plan_text, five_s_levels, warnings, risks):
    pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", size=12)
    pdf.cell(0,10,to_latin1("LCE + 5S Manufacturing Decision Support (Agentic, no-BOM)"), ln=True, align="C")
    pdf.set_font("Arial", size=11)
    pdf.cell(0,8,to_latin1("Developed by Dr. Juana Isabel Méndez and Dr. Arturo Molina"), ln=True, align="C")
    pdf.cell(0,8,to_latin1("Date: "+datetime.now().strftime("%Y-%m-%d")), ln=True, align="C")
    pdf.ln(6)
    for head in ["Supply Chain Configuration & Action Plan","Improvement Opportunities & Risks","Digital/AI Next Steps","Expected 5S Maturity"]:
        pdf.set_font("Arial","B",12); pdf.cell(0,8,to_latin1(head), ln=True)
        pdf.set_font("Arial", size=11); pdf.multi_cell(0,7,to_latin1(parse_section(plan_text, head) or "—")); pdf.ln(2)
    if warnings:
        pdf.set_font("Arial","B",12); pdf.cell(0,8,"Standards Gate Warnings", ln=True)
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

if res:
    pdf_buf = generate_pdf_report(
        res.get("plan_text",""),
        five_s_levels,
        res.get("standards_warnings",[]),
        res.get("risk_register",{}).get("risks",[])
    )
    st.download_button("Download Full Report (PDF)", data=pdf_buf,
                       file_name=f"{datetime.now():%Y-%m-%d}_Agentic_NoBOM_Report.pdf", mime="application/pdf")
