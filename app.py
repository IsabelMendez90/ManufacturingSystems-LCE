import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
import zlib

import re

API_KEY = st.secrets["OPENROUTER_API_KEY"]
st.set_page_config(page_title="LCE + 5S Decision Support Tool", layout="wide")
st.title("LCE + 5S Manufacturing System & Supply Chain Decision Support")
# ----- 5S Taxonomy with Technologies -----
five_s_taxonomy = {
    "Social": [
        {
            "desc": "No integration of social factors; minimal worker well-being consideration.",
            "tech": []
        },
        {
            "desc": "Compliance with basic labor laws and ergonomic guidelines.",
            "tech": ["Ergonomic checklists", "Manual safety reporting"]
        },
        {
            "desc": "Workforce development and participatory work systems; inclusion programs.",
            "tech": ["LMS platforms (Moodle, SAP SuccessFactors)", "Employee engagement apps"]
        },
        {
            "desc": "Worker-centric design of systems; real-time feedback for operator experience and well-being.",
            "tech": ["Wearables (fatigue monitoring)", "Sentiment analysis tools"]
        },
        {
            "desc": "Fully integrated socio-technical systems with co-creation, leadership development, mental health support, and community engagement metrics.",
            "tech": ["Digital inclusion dashboards", "Psychological safety analytics", "Collaboration platforms (Miro, Teams with sentiment AI)"]
        }
    ],
    "Sustainable": [
        {
            "desc": "Unsustainable practices with no environmental consideration.",
            "tech": []
        },
        {
            "desc": "Compliance with basic environmental regulations; manual tracking of resource usage.",
            "tech": ["Spreadsheets for energy/water tracking", "Basic energy/water meters"]
        },
        {
            "desc": "Implementation of energy-efficient processes; partial material recycling.",
            "tech": ["ISO 14001 checklists", "Environmental sensors", "Material recycling software"]
        },
        {
            "desc": "Life Cycle Assessment (LCA), closed-loop material systems, carbon monitoring.",
            "tech": ["OpenLCA", "GaBi", "Carbon calculators", "WMS with recycling metrics"]
        },
        {
            "desc": "Circular economy integration; regenerative systems; AI-enabled sustainability optimization and reporting.",
            "tech": ["AI for energy optimization", "Blockchain for circular tracking", "ESG analytics platforms"]
        }
    ],
    "Sensing": [
        {
            "desc": "Human-based sensing: perception by human operators without sensor support.",
            "tech": ["Visual inspection", "Manual data logging"]
        },
        {
            "desc": "Basic analog/digital sensors integrated into machines; local display and alarms.",
            "tech": ["Thermocouples", "Limit switches", "PLC-based indicators"]
        },
        {
            "desc": "Multi-sensor systems with basic processing units; wireless sensor networks (WSN); partial data fusion.",
            "tech": ["WSNs", "SCADA systems", "LabVIEW", "Basic DAQ systems"]
        },
        {
            "desc": "Integrated sensing-based processes with embedded computing, IoT connectivity, and real-time feedback loops for decision support.",
            "tech": ["IoT platforms (Azure IoT, AWS IoT)", "Embedded microcontrollers", "OPC-UA protocols"]
        },
        {
            "desc": "Smart sensing systems with AI-enabled smart sensors, IIoT, continuous measurements, edge computing, and adaptive closed-loop control.",
            "tech": ["Edge AI devices", "IIoT platforms (MindSphere, Lumada)", "TensorFlow Lite", "Sensor fusion algorithms"]
        }
    ],
    "Smart": [
        {
            "desc": "Manual systems with human control and decision-making.",
            "tech": ["Paper-based logs", "Manual work instructions"]
        },
        {
            "desc": "Basic control systems using open-loop logic and minimal digital feedback.",
            "tech": ["Relay logic", "Timers", "Basic HMI"]
        },
        {
            "desc": "Semi-automated systems with closed-loop feedback; basic process logic and automatic correction mechanisms.",
            "tech": ["PID controllers", "PLCs", "HMI/SCADA integration"]
        },
        {
            "desc": "Advanced automation using intelligent modules and predictive analytics; MES integration.",
            "tech": ["MES systems (Siemens Opcenter, GE Proficy)", "Predictive maintenance platforms (IBM Maximo, Senseye)"]
        },
        {
            "desc": "Intelligent automation using AI modules, neural networks, IIoT platforms, and big data analytics.",
            "tech": ["AI/ML platforms (SageMaker, Vertex AI)", "ANN frameworks (Keras, PyTorch)", "Digital twins"]
        }
    ],
    "Safe": [
        {
            "desc": "Manual risk control, minimal safety measures.",
            "tech": ["Safety posters", "Visual inspections"]
        },
        {
            "desc": "Compliance with occupational safety standards (e.g., ISO 45001).",
            "tech": ["Safety audits", "Risk matrices"]
        },
        {
            "desc": "Integration of machine safety systems (e.g., interlocks, PLCs).",
            "tech": ["Safety PLCs", "Light curtains", "Emergency stops"]
        },
        {
            "desc": "Smart safety systems with sensors, anomaly detection, and predictive alerts.",
            "tech": ["Vibration sensors", "AI for failure prediction", "Integrated HSE dashboards"]
        },
        {
            "desc": "AI-powered safety management with cyber-physical protection, incident prediction, and adaptive control systems.",
            "tech": ["Digital twins for safety", "AI-driven hazard detection", "Cybersecurity software (Darktrace, Fortinet)"]
        }
    ]
}

# ----- 5S Checklists for Self-Assessment -----
five_s_checklists = {
    "Social": [
        ("Basic ergonomic training and labor law compliance", 1),
        ("Workforce development/LMS/inclusion programs", 2),
        ("Real-time operator feedback/monitoring (wearables, sentiment, fatigue)", 3),
        ("Mental health, co-creation, or community engagement programs", 4)
    ],
    "Sustainable": [
        ("Manual tracking of energy, water, or resource use", 1),
        ("Energy-efficient processes/machines, partial recycling", 2),
        ("LCA, closed-loop systems, carbon monitoring", 3),
        ("Circular economy, AI for sustainability, ESG analytics", 4)
    ],
    "Sensing": [
        ("Visual/manual inspection and data logging", 0),
        ("Basic sensors or PLC indicators on machines", 1),
        ("Multi-sensor networks (WSN, SCADA, LabVIEW, basic DAQ)", 2),
        ("IoT/embedded devices, real-time feedback, OPC-UA/MQTT", 3),
        ("Edge AI, IIoT, sensor fusion, adaptive closed-loop control", 4)
    ],
    "Smart": [
        ("Manual work instructions, paper logs", 0),
        ("Basic HMI or relay/timer logic", 1),
        ("Closed-loop automation (PLCs, PID, SCADA)", 2),
        ("MES, advanced automation, predictive analytics", 3),
        ("AI/ML, neural networks, IIoT, digital twins, big data", 4)
    ],
    "Safe": [
        ("Visual safety checks, posters", 0),
        ("Safety audits, risk matrices, ISO 45001 compliance", 1),
        ("Machine safety interlocks, safety PLCs, light curtains", 2),
        ("Sensors/anomaly detection, AI for failure prediction, HSE dashboards", 3),
        ("Digital twins for safety, AI-driven hazard detection, cybersecurity", 4)
    ]
}

def assess_level(checks):
    levels_checked = [lvl for checked, lvl in checks if checked]
    if not levels_checked:
        return 0
    levels_checked = sorted(levels_checked)
    max_level = max(levels_checked)
    for lvl in range(max_level + 1):
        if lvl not in levels_checked:
            return lvl - 1
    return max_level
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
        "End-of-Life: Retire/adapt equipment; document performance and failures; involve HDTs for lifecycle learning."
    ],
    "Facility Design": [
        "Ideation: Specify new product requirements and associated manufacturing processes.",
        "Basic Development: Select manufacturing systems and equipment for facility.",
        "Advanced Development: Design shop-floor layout, define capacity and manufacturing strategy.",
        "Launching: Construct, install, ramp-up production; evaluate and adjust facility performance.",
        "End-of-Life: Audit facility for adaptive reuse, decommission, or transform to new functions."
    ]
}

# --------------- Supply Chain Recommendations ---------------
supply_chain_recommendations = {
    "Product Transfer": [
        "Global and distributed sourcing focused on cost, speed, and modularity.",
        "Supplier selection through audits and multi-criteria decision models (AHP, TCO analysis).",
        "Digital quality management: SPC, APQP, and blockchain/QR traceability systems.",
        "Contracts with penalties/incentives for delivery and quality; decentralized/regionalized sourcing.",
        "Integration of ERP, SRM, digital dashboards for real-time order and inventory management.",
        "Vendor-managed inventory (VMI) and joint development agreements with key suppliers.",
        "Key risks: High dependence on global suppliers, exposure to disruptions, need for buffer stock.",
        "AI-driven tools: Supplier risk scoring (SAP Ariba), AI logistics (FourKites), blockchain compliance."
    ],
    "Technology Transfer": [
        "Hybrid sourcing with high collaboration for process innovation and tech transfer.",
        "Technology selection via pilot testing and technical-economic benchmarking.",
        "Process planning with early lifecycle impact assessment, SOPs, and simulation (digital twins).",
        "Supplier partnerships for rapid qualification cycles and IP sharing.",
        "Integrated MES/ERP, predictive maintenance, and real-time analytics.",
        "Risks: Tech maturity misalignment, capacity gaps, need for operator upskilling.",
        "AI-driven tools: Digital twin modeling, predictive analytics, automated quality inspection."
    ],
    "Facility Design": [
        "Vertically integrated, resilient, and locally anchored supply chain; emphasis on autonomy.",
        "Make-or-buy analysis with deep BoM decomposition (cost, IP, sustainability factors).",
        "Regional ecosystem development: local suppliers, workforce training, community engagement.",
        "Lean layout and green infrastructure (LEED, renewable energy, water reuse).",
        "Simulation for layout, flow, and OEE optimization; full digital twin integration.",
        "Built-in safety/cybersecurity, regulatory compliance, adaptive reuse planning.",
        "Risks: High CAPEX, regulatory complexity, but maximal resilience and control.",
        "AI-driven tools: Digital twins, IIoT, automated maintenance, energy simulation, cybersecurity."
    ]
}


def build_llm_prompt(system_type, industry, lce_stages, five_s_levels, objective):
    # Convert 5S levels to readable text
    s_txt = "\n".join([f"- {dim}: Level {level}" for dim, level in five_s_levels.items()])
    lce_txt = "\n".join([f"- {stage}" for stage in lce_stages]) if lce_stages else "None selected"
    
    # Get supply chain recommendations for this system type
    sc_recs = supply_chain_recommendations.get(system_type, [])
    sc_recs_txt = "\n".join([f"- {rec}" for rec in sc_recs])
    
    prompt = f"""
You are an expert in {system_type} manufacturing systems for the {industry} industry.

The company's stated objective is:
{objective}

Current LCE stages/actions:
{lce_txt}

5S maturity levels:
{s_txt}

When designing the supply chain configuration and action plan, use the following best practices as a starting point, and adapt them as needed to the user's objective, selected LCE stages, and 5S maturity:
{sc_recs_txt}

Please respond in the following format:

[Supply Chain Configuration & Action Plan]
(Provide a concise supply chain configuration and action plan tailored to the above scenario, integrating LCE, 5S, and supply chain management. Reference/adapt the listed best practices where relevant.)

[Improvement Opportunities & Risks]
(Identify the main improvement opportunities and risks for this configuration, considering the user's objective.)

[Digital/AI Next Steps]
(Recommend concrete next steps and digital/AI technologies—drawn from each S in the 5S taxonomy—to help progress toward Industry 5.0 and the goal.)
"""
    return prompt





# ========== Streamlit App Layout ==========



# --- Step 1: Define Objective ---
st.header("1. Define Your Manufacturing Objective")
objective = st.text_input("Describe your main goal (e.g., launch new product, adopt new technology, expand facility):", key="objective", value="Automate assembly of micro-machine cells.")
industry = st.selectbox("Select your industry:", ["Automotive", "Electronics", "Medical Devices", "Consumer Goods", "Other"], key="industry")

# --- Step 2: Select Manufacturing System Type ---
st.header("2. Select Manufacturing System Type")
system_type = st.radio(
    "Choose a system type:",
    ["Product Transfer", "Technology Transfer", "Facility Design"], key="system_type"
)
st.markdown(f"**Selected system type:** {system_type}")
# Reset LCE checkboxes if the system type changes
if "last_system_type" not in st.session_state:
    st.session_state["last_system_type"] = system_type
elif st.session_state["last_system_type"] != system_type:
    # Uncheck all LCE checkboxes
    for act in sum(lce_actions_taxonomy.values(), []):
        key = f"lce_{act}"
        if key in st.session_state:
            st.session_state[key] = False
    st.session_state["last_system_type"] = system_type
# --- Step 3: Select LCE Stage Activities (Checkboxes) ---
st.header("3. Select Relevant LCE Stages/Actions")
lce_actions = lce_actions_taxonomy[system_type]
selected_stages = []
for action in lce_actions:
    checked = st.checkbox(action, key=f"lce_{action}")
    if checked:
        selected_stages.append(action)
st.session_state["selected_stages"] = selected_stages

# --- Step 4: 5S Maturity Assessment (Radio Buttons, not Checkboxes) ---
st.header("4. Assess 5S Maturity (Single Option for Each S)")
five_s_levels = {}
cols = st.columns(5)
for i, dim in enumerate(five_s_taxonomy):
    with cols[i]:
        st.subheader(dim)
        options = []
        for idx, lvl in enumerate(five_s_taxonomy[dim]):
            label = f"Level {idx}: {lvl['desc']}"
            options.append(label)
        level = st.radio(
            f"Select the most accurate description for {dim}:",
            options,
            index=0,
            key=f"{dim}_radio"
        )
        level_idx = options.index(level)
        five_s_levels[dim] = level_idx
        st.markdown(f"**Level assigned:** {level_idx}")
        st.markdown(f"_{five_s_taxonomy[dim][level_idx]['desc']}_")
        techs = five_s_taxonomy[dim][level_idx].get('tech', [])
        if techs:
            st.markdown("**Technologies & Tools:**")
            for t in techs:
                st.write(f"- {t}")
st.session_state["five_s_levels"] = five_s_levels
# --- Optional: Radar Chart Visualization ---
if st.checkbox("Show 5S Profile Radar Chart"):
    radar_df = pd.DataFrame({
        "Dimension": list(five_s_taxonomy.keys()),
        "Level": [five_s_levels[s] for s in five_s_taxonomy]
    })
    radar_fig = px.line_polar(radar_df, r='Level', theta='Dimension', line_close=True, range_r=[0,4])
    st.plotly_chart(radar_fig, use_container_width=True)



def parse_stage_views(llm_response):
    stage_pattern = re.compile(
        r"([A-Za-z \-]+)\nEngineering Analysis:\s*Function:\s*(.*?)\nEngineering Synthesis:\s*Organization:\s*(.*?)\n\s*Information:\s*(.*?)\n\s*Resource:\s*(.*?)\nEngineering Evaluation:\s*Performance:\s*(.*?)(?=\n[A-Za-z \-]+\n|$)", 
        re.DOTALL
    )
    views_dict = {}
    for match in stage_pattern.finditer(llm_response):
        stage = match.group(1).strip()
        views_dict[stage] = {
            "Function": match.group(2).strip(),
            "Organization": match.group(3).strip(),
            "Information": match.group(4).strip(),
            "Resource": match.group(5).strip(),
            "Performance": match.group(6).strip(),
        }
    return views_dict

def build_supply_chain_activity_plantuml_with_views(
    system_type,
    lce_stages,
    five_s_levels,
    stage_views
):
    sys_type_color = {
        "Product Transfer": "#8FD3F4",
        "Technology Transfer": "#A1E3B1",
        "Facility Design": "#F4E6A1"
    }
    system_color = sys_type_color.get(system_type, "#D9D9D9")

    code = [
        "@startuml",
        f"title Supply Chain Activity Diagram: {system_type}",
        f"skinparam backgroundColor {system_color}",
        "skinparam activity {",
        "BackgroundColor<<Ideation>> #F2D7D5",
        "BackgroundColor<<BasicDev>> #D5F2E3",
        "BackgroundColor<<AdvDev>> #D5E1F2",
        "BackgroundColor<<Launching>> #F2E6D5",
        "BackgroundColor<<EoL>> #F2F2D5",
        "}"
    ]
    code.append(f"note right\nSystem Type: {system_type}\nend note")
    stage_map = [
        ("Ideation", "Ideation"),
        ("Basic Development", "BasicDev"),
        ("Advanced Development", "AdvDev"),
        ("Launching", "Launching"),
        ("End-of-Life", "EoL"),
    ]
    swimlanes = [b for a, b in stage_map if any(a in s for s in lce_stages)]
    for stage_label, stereotype in stage_map:
        if not any(stage_label in s for s in lce_stages):
            continue
        code.append(f'partition "{stage_label}" <<{stereotype}>> {{')
        if stage_label in stage_views:
            v = stage_views[stage_label]
            code.append(f':Analysis/Function: {v.get("Function","")};')
            code.append(f':Synthesis/Organization: {v.get("Organization","")};')
            code.append(f':Synthesis/Information: {v.get("Information","")};')
            code.append(f':Synthesis/Resource: {v.get("Resource","")};')
            code.append(f':Evaluation/Performance: {v.get("Performance","")};')
        else:
            code.append(f":{stage_label} Key Activities;")
        code.append("}")

    for i in range(len(swimlanes) - 1):
        code.append(f":{stage_map[i][0]} Key Activities; --> :{stage_map[i+1][0]} Key Activities;")

    code.append("@enduml")
    return "\n".join(code)


def plantuml_encode(text):
    import base64
    def deflate_and_encode(data):
        zlibbed_str = zlib.compress(data.encode('utf-8'))
        compressed_string = zlibbed_str[2:-4]
        return encode64(compressed_string)
    def encode6bit(b):
        if b < 10: return chr(48 + b)
        b -= 10
        if b < 26: return chr(65 + b)
        b -= 26
        if b < 26: return chr(97 + b)
        b -= 26
        if b == 0: return '-'
        if b == 1: return '_'
        return '?'
    def encode64(data):
        res = ''
        length = len(data)
        i = 0
        while i < length:
            b1 = data[i] & 0xFF
            b2 = data[i+1] & 0xFF if i+1 < length else 0
            b3 = data[i+2] & 0xFF if i+2 < length else 0
            res += encode6bit(b1 >> 2)
            res += encode6bit(((b1 & 0x3) << 4) | (b2 >> 4))
            res += encode6bit(((b2 & 0xF) << 2) | (b3 >> 6))
            res += encode6bit(b3 & 0x3F)
            i += 3
        return res
    return deflate_and_encode(text)

def build_stage_views_prompt(selected_stages):
    stages_str = "\n".join([f"- {stage}" for stage in selected_stages])
    context = """
For each selected LCE stage, provide a structured description of the following engineering activity views.

Engineering activities are classified as:
- **Analysis**: Diagnosing, defining, and preparing information.
    - **Function**: Main system functionality, core processes, and activities.
- **Synthesis**: Arranging elements to create new effects and system order.
    - **Organization**: Key human roles, teams, partners, and their structure.
    - **Information**: Key data, documents, and knowledge required or generated.
    - **Resource**: Tools, systems, methodologies, and infrastructure used.
- **Evaluation**: Testing solutions against goals and requirements.
    - **Performance**: Main KPIs or outcomes tracked, aligned to company goals.

**Respond strictly in this markdown format for each stage (use only this format, no explanations):**

[STAGE NAME]
Engineering Analysis: 
    Function: [main process or activity]
Engineering Synthesis:
    Organization: [main roles, teams, partners]
    Information: [main data, docs, or info used/generated]
    Resource: [main tools, systems, infra]
Engineering Evaluation:
    Performance: [main KPIs or outcomes tracked]

Stages to describe:
""" + stages_str
    return context


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY)

# --- Step 5 & 6: LLM Supply Chain Action Plan and Other Advice ---
if st.button("Generate Plan and Recommendations"):
    # 1. Get the PLAN from LLM
    prompt = build_llm_prompt(system_type, industry, selected_stages, five_s_levels, objective)
    with st.spinner("Consulting LLM for plan and recommendations..."):
        plan_completion = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct",
            messages=[
                {"role": "system", "content": "You are a digital manufacturing systems expert."},
                {"role": "user", "content": prompt}
            ]
        )
        llm_response = plan_completion.choices[0].message.content

        # Parse the response by section
        sc_match = re.search(r"\[Supply Chain Configuration & Action Plan\](.*?)(?=\[Improvement Opportunities & Risks\]|\[Digital/AI Next Steps\]|$)", llm_response, re.DOTALL)
        imp_match = re.search(r"\[Improvement Opportunities & Risks\](.*?)(?=\[Digital/AI Next Steps\]|\Z)", llm_response, re.DOTALL)
        ai_match = re.search(r"\[Digital/AI Next Steps\](.*)", llm_response, re.DOTALL)

        st.session_state["llm_response"] = llm_response
        st.session_state["supply_chain_section"] = sc_match.group(1).strip() if sc_match else ""
        st.session_state["improvement_section"] = imp_match.group(1).strip() if imp_match else ""
        st.session_state["ai_section"] = ai_match.group(1).strip() if ai_match else ""

    # 2. Get the STAGE VIEWS from LLM
    stage_views_prompt = build_stage_views_prompt(selected_stages)
    with st.spinner("Consulting LLM for detailed stage views..."):
        stage_views_completion = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct",
            messages=[
                {"role": "system", "content": "You are a digital manufacturing systems expert."},
                {"role": "user", "content": stage_views_prompt}
            ]
        )
        views_response = stage_views_completion.choices[0].message.content
        stage_views = parse_stage_views(views_response)

    # 3. Show Plan/Recommendations
    st.header("5. Supply Chain Configuration & Action Plan")
    st.markdown("**Supply Chain Strategy:**")
    st.info(st.session_state["supply_chain_section"] or "No tailored supply chain plan was generated.")

    # 4. Generate and show the PlantUML diagram
    plantuml_code = build_supply_chain_activity_plantuml_with_views(
        system_type=system_type,
        lce_stages=selected_stages,
        five_s_levels=five_s_levels,
        stage_views=stage_views
    )
    st.session_state["plantuml_code"] = plantuml_code





    plantuml_url_code = plantuml_encode(plantuml_code)
    plantuml_svg_url = f"http://www.plantuml.com/plantuml/svg/{plantuml_url_code}"

    st.markdown(f"[Abrir diagrama en nueva pestaña]({plantuml_svg_url})")

    # Mostrar el diagrama SVG en la app
    st.markdown(f"![Supply Chain UML Diagram]({plantuml_svg_url})", unsafe_allow_html=True)

    # Always show Step 6 if previous responses exist in session_state
if "supply_chain_section" in st.session_state:
    st.header("6. LLM Advisor: Improvement Opportunities & Digital Next Steps")
    
    st.subheader("Supply Chain Configuration & Action Plan")
    st.info(st.session_state.get("supply_chain_section", "No tailored supply chain plan was generated."))

    st.subheader("Improvement Opportunities & Risks")
    st.info(st.session_state.get("improvement_section", "No improvement opportunities provided."))

    st.subheader("Digital/AI Next Steps")
    st.info(st.session_state.get("ai_section", "No digital/AI next steps provided."))
# --- Step 7: Ask the Project LLM Assistant ---
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

st.header("7. Ask the Project LLM Assistant")
st.markdown("""
Type your questions about this manufacturing system scenario, supply chain, or digital strategy.
This assistant only answers questions related to your current project and scenario.
""")
user_question = st.text_input("Ask the LLM Assistant (project-specific questions only):", key="user_project_chat")

if st.button("Send Question"):
    context_block = f"""
Current project objective: {objective}
System type: {system_type}
Industry: {industry}
Selected LCE stages/actions: {', '.join(selected_stages) if selected_stages else 'None'}
5S maturity levels: {', '.join([f"{dim}: {lvl}" for dim, lvl in five_s_levels.items()])}
Recent LLM Supply Chain Action Plan: {st.session_state.get('supply_chain_section','')}
Recent LLM Improvement Opportunities: {st.session_state.get('improvement_section','')}
Recent LLM Digital/AI Next Steps: {st.session_state.get('ai_section','')}
"""
    system_prompt = (
        "You are a digital manufacturing systems advisor. "
        "Only answer questions related to this specific project scenario. "
        "If the question is unrelated (e.g., about general knowledge, or non-manufacturing topics), "
        "politely remind the user that this assistant only answers manufacturing system and supply chain questions related to their current project. "
        "If the user insists on unrelated topics, just reply: "
        "'This system is only for manufacturing system and supply chain project advice.'"
    )
    chat_msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context_block}
    ]
    for entry in st.session_state["chat_history"]:
        chat_msgs.append(entry)
    chat_msgs.append({"role": "user", "content": user_question})

    with st.spinner("The assistant is thinking..."):
        chat_response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct",
            messages=chat_msgs
        )
        answer = chat_response.choices[0].message.content
        st.session_state["chat_history"].append({"role": "user", "content": user_question})
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})
        st.markdown(f"**Assistant:** {answer}")

if st.session_state["chat_history"]:
    st.markdown("#### Conversation History")
    for entry in st.session_state["chat_history"]:
        if entry["role"] == "user":
            st.markdown(f"**You:** {entry['content']}")
        elif entry["role"] == "assistant":
            st.markdown(f"**Assistant:** {entry['content']}")

