# --- LCE + 5S Decision Support Tool (Agentic AI edition: planner → tools → reflector) ---

import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
from fpdf import FPDF
from io import BytesIO
from datetime import datetime
import json, re

# ------------------------ App + API setup ------------------------
st.set_page_config(page_title="LCE + 5S Decision Support (Agentic)", layout="wide")
st.title("LCE + 5S Manufacturing System & Supply Chain Decision Support — Agentic AI")
st.markdown("Developed by: Dr. J. Isabel Méndez  & Dr. Arturo Molina")

API_KEY = st.secrets["OPENROUTER_API_KEY"]
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)

# ------------------------ 5S Taxonomy (concise) ------------------------
five_s_taxonomy = {
    "Social": [
        {"desc": "No integration of social factors; minimal worker well-being consideration.", "tech": []},
        {"desc": "Compliance with basic labor laws and ergonomic guidelines.", "tech": ["Ergonomic checklists", "Manual safety reporting"]},
        {"desc": "Workforce development and participatory work systems; inclusion programs.", "tech": ["LMS platforms (Moodle, SAP SuccessFactors)", "Employee engagement apps"]},
        {"desc": "Worker-centric design; real-time operator feedback/well-being.", "tech": ["Wearables (fatigue monitoring)", "Sentiment analysis tools"]},
        {"desc": "Fully integrated socio-technical systems (co-creation, mental health, community).", "tech": ["Inclusion dashboards", "Psychological safety analytics", "Collaboration platforms (Miro/Teams+AI)"]}
    ],
    "Sustainable": [
        {"desc": "Unsustainable practices with no environmental consideration.", "tech": []},
        {"desc": "Basic compliance; manual tracking of resource usage.", "tech": ["Spreadsheets for energy/water", "Basic meters"]},
        {"desc": "Energy-efficient processes; partial recycling.", "tech": ["ISO 14001 checklists", "Environmental sensors", "Material recycling software"]},
        {"desc": "LCA, closed-loop material systems, carbon monitoring.", "tech": ["OpenLCA", "GaBi", "Carbon calculators", "WMS recycling metrics"]},
        {"desc": "Circular economy; AI-enabled optimization & reporting.", "tech": ["AI energy optimization", "Blockchain traceability", "ESG analytics"]}
    ],
    "Sensing": [
        {"desc": "Human-based sensing only.", "tech": ["Visual inspection", "Manual logging"]},
        {"desc": "Basic sensors; local display and alarms.", "tech": ["Thermocouples", "Limit switches", "PLC indicators"]},
        {"desc": "Multi-sensor + WSN + partial data fusion.", "tech": ["WSNs", "SCADA", "LabVIEW", "Basic DAQ"]},
        {"desc": "Embedded computing + IoT + real-time feedback.", "tech": ["IoT platforms (Azure/AWS)", "Embedded MCUs", "OPC-UA/MQTT"]},
        {"desc": "Edge AI + IIoT + adaptive closed-loop control.", "tech": ["Edge AI devices", "IIoT platforms", "TensorFlow Lite", "Sensor fusion"]}
    ],
    "Smart": [
        {"desc": "Manual systems with human decision-making.", "tech": ["Paper logs", "Manual work instructions"]},
        {"desc": "Basic control (open loop).", "tech": ["Relay logic", "Timers", "Basic HMI"]},
        {"desc": "Closed-loop automation.", "tech": ["PID", "PLCs", "HMI/SCADA"]},
        {"desc": "Advanced automation + predictive analytics; MES.", "tech": ["MES (Opcenter/Proficy)", "Predictive maintenance (Maximo, Senseye)"]},
        {"desc": "AI modules, ANN, IIoT, twins, big data.", "tech": ["SageMaker/Vertex", "Keras/PyTorch", "Digital twins"]}
    ],
    "Safe": [
        {"desc": "Manual risk control, minimal safety.", "tech": ["Safety posters", "Visual checks"]},
        {"desc": "Compliance (e.g., ISO 45001).", "tech": ["Safety audits", "Risk matrices"]},
        {"desc": "Machine safety systems integrated.", "tech": ["Safety PLCs", "Light curtains", "E-stops", "Interlocks"]},
        {"desc": "Smart safety: sensors + anomaly detection.", "tech": ["Vibration sensors", "AI failure prediction", "HSE dashboards"]},
        {"desc": "AI-powered safety management; cyber-physical protection.", "tech": ["Safety twins", "AI hazard detection", "Cybersecurity (IEC 62443/NIST)"]}
    ]
}

# ------------------------ Supply chain recommendations (by system type) ------------------------
supply_chain_recommendations = {
    "Product Transfer": [
        "Agile sourcing; multi-supplier qualification; modular packaging.",
        "MCDM for supplier selection (AHP/TCO), APQP/SPC for quality.",
        "Traceability (QR/blockchain), decentralized/regionalized sourcing.",
        "ERP/SRM dashboards, VMI with key partners.",
    ],
    "Technology Transfer": [
        "Pilot lines and technical-economic benchmarking for process adoption.",
        "Joint development with suppliers; rapid qualification cycles; IP controls.",
        "Predictive maintenance + MES/ERP integration; digital twins for ramp-up.",
    ],
    "Facility Design": [
        "Local, resilient networks; make-or-buy with deep BoM decomposition.",
        "Green infra (LEED, PV, water reuse), cyber-by-design (IEC 62443).",
        "Layout simulation and OEE optimization with twins.",
    ],
}

# ------------------------ 5S regex critic ------------------------
FIVE_S_PATTERNS = {
    "Social": [
        r"\bergonom(?:ic|ics|[íi]a)\b", r"\bhuman factors?\b",
        r"\bDEI\b|\bdiversity\b|\bequity\b|\binclusion\b|\binclusi[oó]n\b",
        r"\bpsychological safety\b", r"\bfatigue (monitoring|management)|\bfatiga\b",
        r"\bwearables?\b|\bexoskeletons?\b", r"\bLMS\b|\blearning management\b",
        r"\btraining\b|\bupskilling\b|\breskilling\b", r"\bco-creation\b|\bparticipatory\b",
        r"\bworkforce\b|\bworker[- ]centric\b",
    ],
    "Sustainable": [
        r"\bLCA\b|\blife[- ]cycle assessment\b|\bACV\b",
        r"\bISO\s?14001\b|\bISO\s?50001\b", r"\bScope\s?[123]\b",
        r"\bcarbon (footprint|accounting|intensity)\b",
        r"\bcircular economy\b|\bremanufactur|\brecycl|\breuse\b|\bclosed[- ]loop\b",
        r"\brenewable\b|\bPPA\b|\bREC\b",
        r"\benergy (audit|efficiency|management)\b|\bwaste heat recovery\b",
        r"\bwater (reuse|recycling|ZLD)\b",
        r"\bESG\b|\bEPD\b|\bcarbon neutrality\b"
    ],
    "Sensing": [
        r"\bSCADA\b", r"\bOPC[-\s]?UA\b|\bMQTT\b|\bModbus\b",
        r"\bhistorian\b|\bPI System\b", r"\bRFID\b|\bbarcode\b|\bRTLS\b",
        r"\bcomputer vision\b|\bcameras?\b", r"\bDAQ\b|\bdata acquisition\b",
        r"\bWSN\b|\bwireless sensor network\b", r"\bedge (device|gateway|computing)\b",
        r"\bIIoT\b", r"\bsensor fusion\b", r"\bcondition monitoring\b|\bvibration\b"
    ],
    "Smart": [
        r"\bPLC\b|\bDCS\b", r"\bISA[- ]?(95|88)\b", r"\bMES\b|\bMOM\b",
        r"\bAPS\b|\badvanced planning\b", r"\bSPC\b|\bAPC\b|\bOEE\b",
        r"\bpredictive maintenance\b|\banomaly detection\b|\bsoft sensors?\b",
        r"\bdigital twins?\b|\bdigital thread\b", r"\boptimization\b|\bMILP\b|\bCP-?SAT\b",
        r"\b(machine learning|ML)\b|\breinforcement learning\b", r"\bAGVs?\b|\bAMRs?\b|\bcobots?\b",
    ],
    "Safe": [
        r"\bISO\s?45001\b", r"\bIEC\s?61508\b|\bIEC\s?62061\b|\bISO\s?13849\b",
        r"\bSIL\b|\bPL\b[cd]?\b|\bfunctional safety\b",
        r"\brisk assessment\b|\bFMEA\b|\bHAZOP\b|\bLOPA\b",
        r"\bLOTO\b|\blockout[- ]tagout\b",
        r"\bsafety PLC\b|\bsafety relay\b|\blight curtain\b|\binterlock\b|\bE-?Stop\b",
        r"\bIEC\s?62443\b|\bNIST\s?(800-82|CSF)\b|\bindustrial cyber(security)?\b",
    ],
}
REQUIRED_MATCHES_BY_LEVEL = {0:0,1:1,2:2,3:3,4:4}

def find_5s_gaps(plan_text: str, target_levels: dict) -> list[str]:
    text = plan_text or ""
    gaps = []
    for dim, target in target_levels.items():
        if target <= 0: 
            continue
        pats = FIVE_S_PATTERNS.get(dim, [])
        hits = set(p for p in pats if re.search(p, text, re.IGNORECASE))
        need = REQUIRED_MATCHES_BY_LEVEL.get(target, 1)
        if len(hits) < need:
            readable = [re.sub(r"\\b","",p) for p in list(pats)[:6]]
            gaps.append(f"{dim}: need ≥{need} concrete signals; found {len(hits)}. Add: {', '.join(readable)}")
    return gaps

# ------------------------ Evidence builder ------------------------
def retrieve_domain_evidence(system_type, industry, selected_stages, five_s_levels, five_s_taxonomy, supply_chain_recommendations):
    stages = [s.split(":")[0].strip() for s in selected_stages] or []
    recs = supply_chain_recommendations.get(system_type, [])
    lines = []
    if recs:
        lines.append("SUPPLY-CHAIN HINTS:")
        lines += [f"- {r}" for r in recs]
    if stages:
        lines.append("\nSELECTED LCE STAGES:")
        lines += [f"- {s}" for s in stages]
    lines.append("\n5S TECH HINTS @ CURRENT LEVEL:")
    for dim, lvl in five_s_levels.items():
        techs = five_s_taxonomy[dim][lvl].get("tech", [])
        if techs:
            lines.append(f"- {dim} (L{lvl}): " + "; ".join(techs))
    return "\n".join(lines)

# ------------------------ Planner / Executor / Reflector ------------------------
class AgentState:
    def __init__(self, objective, system_type, industry, five_s_levels, selected_stages, bom_df):
        self.objective = objective
        self.system_type = system_type
        self.industry = industry
        self.five_s_levels = five_s_levels
        self.selected_stages = selected_stages
        self.bom_df = bom_df if isinstance(bom_df, pd.DataFrame) else pd.DataFrame()
        self.observations = []     # tool outputs
        self.generated_bom = None  # if the agent synthesizes one

def llm(msgs, temperature=0.2, seed=42, model="mistralai/mistral-7b-instruct"):
    resp = client.chat.completions.create(
        model=model, temperature=temperature, seed=seed, messages=msgs
    )
    return resp.choices[0].message.content or ""

def plan_with_llm(state: AgentState, evidence: str) -> str:
    # Core planning prompt (Generative AI)
    lce_txt = "\n".join([f"- {s}" for s in state.selected_stages]) if state.selected_stages else "None"
    s_txt = "\n".join([f"- {d}: L{v}" for d,v in state.five_s_levels.items()])
    prompt = f"""
You are an expert in {state.system_type} manufacturing systems for the {state.industry} industry.

Context:
Objective: {state.objective}
Selected LCE stages:
{lce_txt}
Current 5S maturity:
{s_txt}

Use the following evidence strictly:
{evidence}

Return the following sections exactly (markdown):
[Supply Chain Configuration & Action Plan]
(Concise, actionable and aligned to LCE + 5S.)

[Improvement Opportunities & Risks]
(Bullets or short paragraphs.)

[Digital/AI Next Steps]
(Cross-map relevant tools/tech per S.)

[Expected 5S Maturity]
(List as: Social: X, Sustainable: Y, Sensing: Z, Smart: W, Safe: V.)
"""
    return llm([{"role":"system","content":"You are a digital manufacturing systems expert."},
                {"role":"user","content":prompt}])

def parse_section(text, head):
    m = re.search(rf"\[{re.escape(head)}\](.*?)(?=\n\[[A-Z].*?\]|\Z)", text, re.DOTALL)
    return m.group(1).strip() if m else ""

def extract_expected_levels(text, fallback):
    out = {}
    for dim in ["Social","Sustainable","Sensing","Smart","Safe"]:
        m = re.search(rf"{dim}\s*[:\-]?\s*(?:Level\s*)?(\d)", text, re.IGNORECASE)
        out[dim] = int(m.group(1)) if m else fallback.get(dim, 0)
    return out

# ------------------------ Tooling (generic, not micromachine-specific) ------------------------
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

Return JSON ONLY with:
{{
 "requirements": ["<short bullets>"],
 "constraints": ["<standards, capacity, budget, compliance if stated or implied>"],
 "success_metrics": ["<throughput/quality/sustainability/safety/KPIs>"]
}}"""
        txt = llm([{"role":"system","content":"You turn free text into compact JSON specs."},
                   {"role":"user","content":prompt}], temperature=0.1)
        try:
            js = json.loads(re.search(r"\{.*\}", txt, re.DOTALL).group(0))
        except Exception:
            js = {"requirements":[],"constraints":[],"success_metrics":[]}
        return {"tool": self.name, "summary": "Structured specs extracted", "data": js}

class BomSynthesizerLLM(Tool):
    name = "bom_synth"
    def run(self, state):
        if isinstance(state.bom_df, pd.DataFrame) and not state.bom_df.empty:
            return {"tool": self.name, "summary": "User BOM provided; synthesis skipped.", "data": {}}
        prompt = f"""Synthesize a minimalist starter Bill of Materials for a manufacturing system aligned to:

Objective: {state.objective}
System type: {state.system_type}
Industry: {state.industry}

Return JSON ONLY:
{{
 "bom":[
   {{"Category":"Structure","Item":"...","Purpose":"...","Qty":1,"SafetyCritical":false}},
   {{"Category":"Control","Item":"...","Purpose":"...","Qty":1,"SafetyCritical":true}}
 ]
}} 
Rules: 6–14 lines, general-purpose items (not brand-specific), include at least one safety-critical item."""
        txt = llm([{"role":"system","content":"You design minimal, general BOMs."},
                   {"role":"user","content":prompt}], temperature=0.3)
        try:
            js = json.loads(re.search(r"\{.*\}", txt, re.DOTALL).group(0))
            bom_rows = js.get("bom", [])
        except Exception:
            bom_rows = []
        # Convert to DF for app use
        df = pd.DataFrame(bom_rows) if bom_rows else pd.DataFrame()
        state.generated_bom = df
        return {"tool": self.name, "summary": f"Synthesized BOM with {len(df)} lines." if not df.empty else "No BOM generated.", "data": {"rows": bom_rows}}

class SafetyGate(Tool):
    name = "safety_gate"
    # simple keyword rules that are cross-domain
    RULES = [
        # (if keyword seen, require these dependencies)
        ("laser", ["Enclosure", "Interlock/E-Stop", "Eye Protection Policy"]),
        ("spindle|milling|cutting", ["E-Stop", "Guarding", "Chip Extraction"]),
        ("robot|cobot|amr|agv", ["Emergency Stop", "Speed & Separation Monitoring", "Risk Assessment ISO 10218/TS 15066"]),
        ("furnace|kiln|heater", ["Thermal PPE", "Overtemp Cutoff", "Ventilation"]),
        ("edm|electro-discharge", ["Dielectric Handling", "Fluid Containment", "Electrical Isolation"]),
        ("laser|uv|blue", ["IEC 60825 labelling", "Class-1 enclosure"]),
        ("saw|shear|press", ["Two-hand control or guarding", "LOTO procedure"]),
        ("chemical|solvent|resin", ["MSDS available", "Spill kit", "Ventilation"]),
    ]
    def run(self, state):
        df = state.bom_df if (isinstance(state.bom_df, pd.DataFrame) and not state.bom_df.empty) else state.generated_bom
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {"tool": self.name, "summary": "No BOM to evaluate.", "data": {"warnings":[]}}
        text = " ".join(map(lambda x: str(x).lower(), df.get("Item", pd.Series([], dtype=str))))
        warnings = []
        for pattern, deps in self.RULES:
            if re.search(pattern, text):
                for d in deps:
                    # if dependency word not present anywhere in BOM Items, warn:
                    if not any(d.lower().split()[0] in str(x).lower() for x in df.get("Item", [])):
                        warnings.append(f"Detected '{pattern}' → missing dependency: {d}")
        status = "OK" if not warnings else "WARNINGS"
        return {"tool": self.name, "summary": f"Safety Gate: {status}", "data": {"warnings":warnings}}

TOOL_REGISTRY = {
    "spec_extractor": SpecExtractorLLM(),
    "bom_synth":      BomSynthesizerLLM(),
    "safety_gate":    SafetyGate(),
}

# ------------------------ Agent loop ------------------------
def agentic_run(objective, system_type, industry, selected_stages, five_s_levels,
                bom_df: pd.DataFrame | None,
                enable_agentic=True, auto_iterate=True, max_steps=3):
    state = AgentState(objective, system_type, industry, five_s_levels, selected_stages, bom_df)
    evidence = retrieve_domain_evidence(system_type, industry, selected_stages, five_s_levels, five_s_taxonomy, supply_chain_recommendations)
    plan_text = ""
    safety_warnings = []

    steps = 1 if not enable_agentic else max_steps
    for step in range(steps):
        # 1) PLAN
        plan_text = plan_with_llm(state, evidence)

        # Extract the core plan to evaluate coverage
        sc_text = parse_section(plan_text, "Supply Chain Configuration & Action Plan") or plan_text
        gaps = find_5s_gaps(sc_text, five_s_levels)

        # 2) EXECUTE TOOLS (decide which to run)
        run_tools = []
        # Always extract specs once; synthesize BOM only if user didn't provide one; always run safety gate
        if step == 0: run_tools.append("spec_extractor")
        if (state.bom_df is None or state.bom_df.empty) and step == 0:
            run_tools.append("bom_synth")
        run_tools.append("safety_gate")

        observations = []
        for t in run_tools:
            out = TOOL_REGISTRY[t].run(state)
            observations.append(out)
            # if BOM was synthesized, prefer it going forward
            if t == "bom_synth" and isinstance(state.generated_bom, pd.DataFrame) and not state.generated_bom.empty:
                state.bom_df = state.generated_bom.copy()

        # 3) REFLECT
        safety_warnings = []
        for obs in observations:
            if obs["tool"] == "safety_gate":
                safety_warnings = obs["data"].get("warnings", [])
        passed_safety = (len(safety_warnings) == 0)
        passed_5s = (len(gaps) == 0)

        if (not enable_agentic) or (not auto_iterate):
            # Just one pass or manual iterate
            evidence += "\n\n[OBSERVATIONS]\n" + "\n".join([f"- {o['tool']}: {o['summary']}" for o in observations])
            break

        if passed_safety and passed_5s:
            # Yay — done
            evidence += "\n\n[OBSERVATIONS]\n" + "\n".join([f"- {o['tool']}: {o['summary']}" for o in observations])
            break
        else:
            # Add critic findings to evidence and iterate
            critic = []
            if not passed_safety:
                critic.append("SAFETY FINDINGS:\n- " + "\n- ".join(safety_warnings[:10]))
            if not passed_5s:
                critic.append("5S COVERAGE FINDINGS:\n- " + "\n- ".join(gaps[:10]))
            evidence += "\n\n[NEEDS FIX — ADDRESS IN NEXT PASS]\n" + "\n\n".join(critic)

    # Return final artifacts
    return {
        "plan_text": plan_text,
        "bom_df": state.bom_df if isinstance(state.bom_df, pd.DataFrame) else pd.DataFrame(),
        "safety_warnings": safety_warnings,
    }

# ------------------------ UI ------------------------

# 1) Objective, industry, role
st.header("1) Define Scenario")
objective = st.text_input("Objective (e.g., launch new product, adopt new process, expand facility):",
                          value=st.session_state.get("objective","Design and ramp a new small-form-factor manufacturing cell."),
                          key="objective")
industry = st.selectbox("Industry:", ["Automotive","Electronics","Medical Devices","Consumer Goods","Other"],
                        index=st.session_state.get("industry_idx", 1), key="industry")
st.session_state["industry_idx"] = ["Automotive","Electronics","Medical Devices","Consumer Goods","Other"].index(industry)

system_types = ["Product Transfer","Technology Transfer","Facility Design"]
system_type = st.radio("Manufacturing system type:", system_types, key="system_type")

role_options = ["Design Engineer","Process Engineer","Manufacturing Engineer","Safety Supervisor","Sustainability Manager","Supply Chain Analyst","Manager/Decision Maker","Other"]
role_selected = st.selectbox("Your role:", role_options, index=2, key="user_role")

# 2) LCE stages
st.header("2) Select Relevant LCE Stages/Actions")
lce_actions_taxonomy = {
    "Product Transfer": [
        "Ideation: Identify product BOM, materials, and quality requirements.",
        "Basic Development: Define manufacturing and quality control specs.",
        "Advanced Development: Supplier evaluation/selection.",
        "Launching: In-house assembly, QC checks, workforce training.",
        "End-of-Life: Disassembly/recycling/reuse planning; reverse logistics."
    ],
    "Technology Transfer": [
        "Ideation: Capture technical specs for new component/family.",
        "Basic Development: Specify/select new technology, equipment, tools.",
        "Advanced Development: Process plan, control docs, SOPs.",
        "Launching: Install/test/ramp; optimize production.",
        "End-of-Life: Retire/adapt; document learnings; digital twins."
    ],
    "Facility Design": [
        "Ideation: Specify product needs and processes.",
        "Basic Development: Select systems and equipment.",
        "Advanced Development: Layout design, capacity and strategy.",
        "Launching: Build/install/ramp; evaluate performance.",
        "End-of-Life: Audit for reuse, decommission, transform."
    ]
}
actions = lce_actions_taxonomy[system_type]
selected_stages = []
for i, action in enumerate(actions):
    checked = st.checkbox(action, key=f"lce_{i}")
    if checked: selected_stages.append(action)

# 3) 5S maturity
st.header("3) Current 5S Maturity (pick one per S)")
five_s_levels = {}
cols = st.columns(5)
for i, dim in enumerate(["Social","Sustainable","Sensing","Smart","Safe"]):
    with cols[i]:
        options = [f"Level {idx}: {lvl['desc']}" for idx, lvl in enumerate(five_s_taxonomy[dim])]
        level = st.radio(dim, options, index=0, key=f"{dim}_radio")
        five_s_levels[dim] = options.index(level)
        techs = five_s_taxonomy[dim][five_s_levels[dim]].get("tech", [])
        if techs:
            st.caption("Tech hints: " + "; ".join(techs))

# 4) Agentic controls
st.header("4) Agentic Controls")
enable_agentic = st.toggle("Enable Agentic Mode (plan → tools → reflect)", value=True)
auto_iterate   = st.toggle("Auto-iterate until pass (safety + 5S coverage)", value=True)

# 5) Optional BOM (user-provided). If empty, the agent can synthesize one.
st.header("5) (Optional) Provide a BOM — or let the agent synthesize")
if "bom_df" not in st.session_state:
    st.session_state["bom_df"] = pd.DataFrame(columns=["Category","Item","Purpose","Qty","SafetyCritical"])
bom_editor = st.data_editor(st.session_state["bom_df"], num_rows="dynamic", use_container_width=True, key="bom_editor")
st.session_state["bom_df"] = bom_editor

# 6) Run Agent
if st.button("Generate Plan & Recommendations"):
    with st.spinner("Agent planning, executing tools, and refining..."):
        result = agentic_run(
            objective=objective,
            system_type=system_type,
            industry=industry,
            selected_stages=selected_stages,
            five_s_levels=five_s_levels,
            bom_df=st.session_state["bom_df"],
            enable_agentic=enable_agentic,
            auto_iterate=auto_iterate,
            max_steps=3
        )
        st.session_state["agent_result"] = result

# 7) Show results
res = st.session_state.get("agent_result")
if res:
    st.header("6) Results")
    plan_text = res["plan_text"]
    st.subheader("Supply Chain Configuration & Action Plan")
    st.info(parse_section(plan_text, "Supply Chain Configuration & Action Plan") or "No plan text.")

    st.subheader("Improvement Opportunities & Risks")
    st.info(parse_section(plan_text, "Improvement Opportunities & Risks") or "—")

    st.subheader("Digital/AI Next Steps")
    st.info(parse_section(plan_text, "Digital/AI Next Steps") or "—")

    st.subheader("Expected 5S Maturity")
    exp = parse_section(plan_text, "Expected 5S Maturity")
    st.info(exp or "—")
    expected_5s = extract_expected_levels(exp, five_s_levels)

    # Radar charts
    st.header("7) 5S Profiles")
    curr_df = pd.DataFrame({"Dimension": list(five_s_levels.keys()), "Level":[five_s_levels[k] for k in five_s_levels]})
    fig_curr = px.line_polar(curr_df, r="Level", theta="Dimension", line_close=True, range_r=[0,4])
    st.plotly_chart(fig_curr, use_container_width=True)
    if expected_5s != five_s_levels:
        exp_df = pd.DataFrame({"Dimension": list(expected_5s.keys()), "Level":[expected_5s[k] for k in expected_5s]})
        fig_exp = px.line_polar(exp_df, r="Level", theta="Dimension", line_close=True, range_r=[0,4])
        st.plotly_chart(fig_exp, use_container_width=True)

    # BOM outcome (provided or synthesized)
    st.header("8) Bill of Materials (BOM)")
    bom_df = res.get("bom_df", pd.DataFrame())
    if isinstance(bom_df, pd.DataFrame) and not bom_df.empty:
        st.dataframe(bom_df, use_container_width=True)
        st.download_button("Download BOM (CSV)", bom_df.to_csv(index=False).encode("utf-8"),
                           "bom.csv", "text/csv")
    else:
        st.info("No BOM available (none provided and synthesis may have failed).")

    # Safety gate findings
    st.header("9) Safety Gate")
    warns = res.get("safety_warnings", [])
    if warns:
        st.error("Safety warnings:\n- " + "\n- ".join(warns))
    else:
        st.success("Safety Gate: OK")

# 8) Export PDF
def to_latin1(text):
    if not isinstance(text, str): text = str(text)
    return text.encode('latin-1','replace').decode('latin-1')

def generate_pdf_report(plan_text, five_s_levels, bom_df):
    pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", size=12)
    pdf.cell(0,10,to_latin1("LCE + 5S Decision Support Report (Agentic)"), ln=True, align="C")
    pdf.set_font("Arial", size=11)
    pdf.cell(0,8,to_latin1("Developed by Dr. Juana Isabel Méndez and Dr. Arturo Molina"), ln=True, align="C")
    pdf.cell(0,8,to_latin1("Date: "+datetime.now().strftime("%Y-%m-%d")), ln=True, align="C")
    pdf.ln(6)

    # Sections
    pdf.set_font("Arial","B",12); pdf.cell(0,8,"Supply Chain Configuration & Action Plan", ln=True)
    pdf.set_font("Arial", size=11); pdf.multi_cell(0,7,to_latin1(parse_section(plan_text,"Supply Chain Configuration & Action Plan")))
    pdf.ln(3); pdf.set_font("Arial","B",12); pdf.cell(0,8,"Improvement Opportunities & Risks", ln=True)
    pdf.set_font("Arial", size=11); pdf.multi_cell(0,7,to_latin1(parse_section(plan_text,"Improvement Opportunities & Risks")))
    pdf.ln(3); pdf.set_font("Arial","B",12); pdf.cell(0,8,"Digital/AI Next Steps", ln=True)
    pdf.set_font("Arial", size=11); pdf.multi_cell(0,7,to_latin1(parse_section(plan_text,"Digital/AI Next Steps")))
    pdf.ln(3); pdf.set_font("Arial","B",12); pdf.cell(0,8,"Expected 5S Maturity", ln=True)
    pdf.set_font("Arial", size=11); pdf.multi_cell(0,7,to_latin1(parse_section(plan_text,"Expected 5S Maturity")))

    if isinstance(bom_df, pd.DataFrame) and not bom_df.empty:
        pdf.ln(6); pdf.set_font("Arial","B",12); pdf.cell(0,8,"Bill of Materials (BOM)", ln=True)
        pdf.set_font("Arial", size=11)
        for _,row in bom_df.iterrows():
            line = f"- [{row.get('Category','')}] {row.get('Item','')} x{row.get('Qty','')}"
            pdf.multi_cell(0,6,to_latin1(line[:200]))

    buf = BytesIO(); pdf_bytes = pdf.output(dest='S').encode('latin1')
    buf.write(pdf_bytes); buf.seek(0); return buf

if res:
    pdf_buf = generate_pdf_report(res.get("plan_text",""), five_s_levels, res.get("bom_df", pd.DataFrame()))
    st.download_button("Download Full Report (PDF)", data=pdf_buf, file_name=f"{datetime.now():%Y-%m-%d}_Agentic_Report.pdf", mime="application/pdf")
