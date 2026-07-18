# Frozen Manufacturing Knowledge Base for LCE-I5S Decision Support

**Knowledge base ID:** `manufacturing_knowledge_base`  
**Version:** `1.0.0`  
**Date frozen:** `2026-07-18`  
**Runtime retrieval:** No. This is a frozen curated knowledge base, not a RAG index.

## Purpose

Frozen, manually curated procedural knowledge base for a controlled LLM-based decision-support framework in manufacturing-system design. It supports generation for Product Transfer, Technology Transfer, and Facility Design scenarios, organised by LCE stage and Industry 5.0 I5S dimension.

## Boundary conditions
- This knowledge base is used for generation in the full-framework configuration only.
- It is not a RAG index and does not perform runtime retrieval.
- It excludes Ramírez, Cortés, and Molina's IPPMD-based lightweight manufacturing-system design framework because that framework is reserved as an external procedural benchmark.
- It does not provide empirical validation, plant-floor performance evidence, or automatic implementation authority.
- AI outputs must remain decision support; changes to SOPs, control plans, MES/ERP records, equipment parameters, supplier status, production settings, or operational records require human review and approval.

## Excluded generation sources
- Ramírez, Cortés, and Molina IPPMD-based lightweight manufacturing-system design framework (reserved for external benchmark scoring, not generation).
- Reviewer reports or benchmark scoring rubrics.
- Unverified public web sources at runtime.
- User-unprovided plant-floor numerical targets, deadlines, defect rates, recovery rates, or certified performance claims.

## Source basis
- **Digital Technologies for Sustainable Manufacturing, Chapter 1**: LCE framing; lifecycle scope from materials/products/processes/systems/supply chains to end-of-life; integration of I5S principles.
- **Digital Technologies for Sustainable Manufacturing, Chapter 2**: Definitions of Social, Sustainable, Sensing, Smart, and Safe systems and their lifecycle use from ideation to end-of-life.
- **Digital Technologies for Sustainable Manufacturing, Chapter 6**: Manufacturing-system entity; Product Transfer, Technology Transfer, Facility Design; LCE stage activities, inputs, and outputs.
- **Digital Technologies for Sustainable Manufacturing, Chapter 7**: Supply-chain development logic; resilience, supplier engagement, transparency, real-time decision support, and end-of-life recovery.
- **Standard manufacturing engineering methods and safety/environmental management practice**: Representative tools and methods such as APQP, SPC, FMEA, HAZOP, LOTO, ISO 14001, ISO 45001, ISO 13849, IEC 62443, MES/MOM, SCADA, OPC-UA/MQTT, APS, digital twins, DES, CAD/BIM.

## I5S dimensions
### Social
Human-centred and socially responsible manufacturing-system design, including ergonomics, training, inclusion, operator participation, fair work, and stakeholder engagement.
Typical cues: ergonomics, human factors, operator training, LMS, participatory design, co-creation, workforce skills

### Sustainable
Environmentally responsible manufacturing-system design across lifecycle stages, including resource efficiency, energy monitoring, LCA, circularity, reuse, recycling, and reverse logistics.
Typical cues: LCA, energy monitoring, carbon accounting, resource efficiency, waste reduction, reuse, recycling, reverse logistics, circular economy

### Sensing
Data capture and monitoring capability for manufacturing assets, processes, quality, safety, and lifecycle states.
Typical cues: SCADA, DAQ, OPC-UA, MQTT, IIoT, edge sensors, condition monitoring, real-time monitoring

### Smart
Decision-support, automation, analytics, simulation, optimization, and digital-twin capability that improves planning, control, diagnosis, and lifecycle learning without autonomous unapproved actuation.
Typical cues: PLC, PID, MES, MOM, APS, optimization, predictive maintenance, digital twin, simulation, analytics

### Safe
Occupational, machine, process, environmental, and cyber-physical safety controls across the manufacturing-system lifecycle.
Typical cues: ISO 45001, ISO 13849, IEC 61508, IEC 62061, FMEA, HAZOP, LOPA, LOTO, IEC 62443, cybersecurity


## Typology × LCE-stage records
### Product Transfer — Ideation
- **Core activity:** Define product information, BOM, materials, standards, quality requirements, and delivery expectations.
- **Engineering analysis:** Capture market/customer requirements, preliminary design concepts, existing product data, and product-transfer constraints.
- **Engineering synthesis: organisation:** Product engineering, manufacturing engineering, quality, procurement, and supply-chain teams.
- **Engineering synthesis: information:** BOM, material specification sheet, component standards, quality requirement matrix, estimated delivery needs, traceability needs.
- **Engineering synthesis: resource:** BOM tools, material databases, quality templates, stakeholder input, and supplier capability information.
- **Engineering evaluation:** BOM and quality requirements approved as the basis for manufacturing and supplier planning.
- **Representative tools/methods:** BOM management; material database; quality requirement matrix; traceability template
- **Expected deliverables/tollgates:** BOM draft; material specification sheet; quality requirement matrix; product-transfer requirement brief
- **I5S cues:** {'Social': ['stakeholder input'], 'Sustainable': ['durability', 'recyclability'], 'Sensing': ['traceability needs'], 'Smart': ['BOM tools'], 'Safe': ['quality/safety requirements']}
- **Source basis:** Book ch.6, Life Cycle diagram flow, Product Transfer description; Book ch.6, Idea | Imagination: Individual Component Specification
- **Prompt priority:** Define product information, BOM, materials, standards, quality requirements, and delivery expectations.

### Product Transfer — Basic Development
- **Core activity:** Define manufacturing and quality-control requirements for volume, cost, time, product attributes, tolerances, and acceptance variables.
- **Engineering analysis:** Translate BOM and product attributes into manufacturing specifications and quality-control parameters.
- **Engineering synthesis: organisation:** Manufacturing engineering, process planning, quality assurance, supply-chain planning, and metrology support.
- **Engineering synthesis: information:** Manufacturing specification, quality-control parameters, inspection plan, acceptance criteria, control-variable definitions.
- **Engineering synthesis: resource:** APQP/SPC tools, process flow templates, metrology resources, inspection templates, and training materials.
- **Engineering evaluation:** Manufacturing and inspection plans reviewed for feasibility before supplier qualification and launch preparation.
- **Representative tools/methods:** APQP; SPC; inspection planning; metrology; process flow diagram
- **Expected deliverables/tollgates:** manufacturing specification; control plan; inspection plan; acceptance criteria; process flow diagram
- **I5S cues:** {'Social': ['workforce skill matrix'], 'Sustainable': ['waste prevention via quality control'], 'Sensing': ['inspection variables'], 'Smart': ['SPC analysis'], 'Safe': ['acceptance/rejection criteria']}
- **Source basis:** Book ch.6, Basic Development: Manufacturing and Quality Control Requirements
- **Prompt priority:** Define manufacturing and quality-control requirements for volume, cost, time, product attributes, tolerances, and acceptance variables.

### Product Transfer — Advanced Development
- **Core activity:** Select manufacturing suppliers and partners based on capabilities, capacity, compliance, cost, quality, and risk.
- **Engineering analysis:** Evaluate supplier capabilities and qualification evidence against manufacturing specifications.
- **Engineering synthesis: organisation:** Procurement, supply chain, quality, product engineering, and technical evaluation teams.
- **Engineering synthesis: information:** Supplier shortlist, qualification reports, audit evidence, supplier capability/capacity reports, risk assessment.
- **Engineering synthesis: resource:** AHP/TCO or similar MCDM tools, supplier data, audit checklists, capability matrix, and compliance screening.
- **Engineering evaluation:** Supplier selection supported by qualification evidence and documented sourcing decision.
- **Representative tools/methods:** supplier MCDM; AHP; TCO; supplier audit; risk matrix; supplier capability assessment
- **Expected deliverables/tollgates:** supplier shortlist; qualification report; supplier agreement/contract; supplier capability and capacity report
- **I5S cues:** {'Social': ['supplier engagement'], 'Sustainable': ['supplier sustainability screening'], 'Sensing': ['supplier data'], 'Smart': ['MCDM scoring'], 'Safe': ['compliance checks']}
- **Source basis:** Book ch.6, Advanced Development: Manufacturing Supplier Selection; Book ch.7, supplier engagement and resilience
- **Prompt priority:** Select manufacturing suppliers and partners based on capabilities, capacity, compliance, cost, quality, and risk.

### Product Transfer — Launching
- **Core activity:** Manufacture, assemble, check quality controls, document delivery/inventory, train workforce, and manage ramp-up readiness.
- **Engineering analysis:** Execute manufacturing and assembly instructions, quality checks, training, and launch-readiness verification.
- **Engineering synthesis: organisation:** Production team, quality manager, training coordinator, manufacturing engineering, and supply-chain team.
- **Engineering synthesis: information:** Assembly instructions, quality-control reports, acceptance criteria, training records, traceability log, delivery/inventory records.
- **Engineering synthesis: resource:** Assembly equipment, training materials, inspection tools, APQP/SPC records, and digital traceability system.
- **Engineering evaluation:** Assembly readiness verified through acceptance criteria, training completion, quality-control evidence, traceability readiness, and ramp-up approval.
- **Representative tools/methods:** APQP; SPC; quality inspection; training matrix; traceability log; inventory documentation
- **Expected deliverables/tollgates:** assembled components; quality-control report; delivery/inventory documentation; ramp-up checklist; traceability log
- **I5S cues:** {'Social': ['training'], 'Sustainable': ['waste/rework reduction'], 'Sensing': ['quality-control data'], 'Smart': ['traceability system'], 'Safe': ['safe assembly procedures']}
- **Source basis:** Book ch.6, Launching: Manufacturing and Assembly of Components
- **Prompt priority:** Manufacture, assemble, check quality controls, document delivery/inventory, train workforce, and manage ramp-up readiness.

### Product Transfer — End-of-Life
- **Core activity:** Plan disassembly, reuse, recycling, reverse logistics, contract closure, and product-transfer lifecycle documentation.
- **Engineering analysis:** Assess end-of-life options for components/products and define recovery, reuse, and reverse-logistics pathways.
- **Engineering synthesis: organisation:** Sustainability, logistics, procurement, supplier management, product engineering, and quality teams.
- **Engineering synthesis: information:** Reverse-logistics plan, contract-closure checklist, disassembly instructions, recycling pathways, lessons learned.
- **Engineering synthesis: resource:** Disassembly tools, recycling contracts, supplier records, lifecycle-data repository, and traceability system.
- **Engineering evaluation:** End-of-life and contract-closure plan approved with reuse, recycling, and reverse-logistics pathways documented.
- **Representative tools/methods:** reverse logistics; disassembly planning; recycling pathway; contract closure; lessons learned
- **Expected deliverables/tollgates:** reverse-logistics plan; contract-closure checklist; disassembly instruction; recycling/reuse pathway
- **I5S cues:** {'Social': ['responsible decommissioning'], 'Sustainable': ['reuse/recycling'], 'Sensing': ['lifecycle records'], 'Smart': ['reconfiguration/upgrade records'], 'Safe': ['safe disposal']}
- **Source basis:** Book ch.6, End of Life: 5S; Book ch.2, End-of-life 10R and 5S
- **Prompt priority:** Plan disassembly, reuse, recycling, reverse logistics, contract closure, and product-transfer lifecycle documentation.

### Technology Transfer — Ideation
- **Core activity:** Capture component or component-family specifications: geometry, drawings, materials, and production rates/batch size.
- **Engineering analysis:** Capture technical specifications, geometry, material data, production-rate assumptions, and feasibility constraints.
- **Engineering synthesis: organisation:** Product engineering, manufacturing engineering, materials, quality, and technology specialists.
- **Engineering synthesis: information:** Technical specification sheet, geometric data/drawings, material data sheets, production batch-size documentation.
- **Engineering synthesis: resource:** CAD models, material databases, supplier capability data, and technical-economic benchmarking inputs.
- **Engineering evaluation:** Technical specification sheet approved by product and manufacturing engineering for downstream technology-selection decisions.
- **Representative tools/methods:** CAD; material data sheet; technical specification; benchmarking input
- **Expected deliverables/tollgates:** technical specification sheet; geometry/drawing package; material data sheet; batch-size documentation
- **I5S cues:** {'Social': ['engineering collaboration'], 'Sustainable': ['material selection'], 'Sensing': ['data requirements'], 'Smart': ['CAD/CAM data'], 'Safe': ['technical constraints']}
- **Source basis:** Book ch.6, Idea | Imagination: Technology transfer Individual Component Specification
- **Prompt priority:** Capture component or component-family specifications: geometry, drawings, materials, and production rates/batch size.

### Technology Transfer — Basic Development
- **Core activity:** Evaluate machines, material-handling equipment, tooling, jigs/fixtures, and supporting tools to select manufacturing resources.
- **Engineering analysis:** Compare manufacturing technologies and equipment using technical-economic, feasibility, capacity, and sustainability criteria.
- **Engineering synthesis: organisation:** Manufacturing engineering, procurement, process engineering, quality, maintenance, and equipment suppliers.
- **Engineering synthesis: information:** Benchmark report, equipment/technology shortlist, supplier data, feasibility assumptions, pilot-line or joint-development trial plan.
- **Engineering synthesis: resource:** Equipment catalogues, supplier trials, pilot-line resources, AHP/TCO or similar decision tools, feasibility-analysis templates.
- **Engineering evaluation:** Technology-selection decision supported by benchmark evidence, equipment shortlist, and human-approved feasibility review.
- **Representative tools/methods:** technical-economic benchmarking; equipment shortlist; AHP/TCO; pilot-line trial; joint-development trial; resource-selection matrix
- **Expected deliverables/tollgates:** technology/equipment selection report; equipment specification sheet; preliminary process flowchart; pilot-trial plan
- **I5S cues:** {'Social': ['cross-functional review'], 'Sustainable': ['resource efficiency'], 'Sensing': ['equipment monitoring requirements'], 'Smart': ['resource optimization'], 'Safe': ['equipment safety criteria']}
- **Source basis:** Book ch.6, Basic Development: Manufacturing Resources Specification and Selection; Book ch.6, Ten Core Technologies
- **Prompt priority:** Evaluate machines, material-handling equipment, tooling, jigs/fixtures, and supporting tools to select manufacturing resources.

### Technology Transfer — Advanced Development
- **Core activity:** Develop process plan, SOPs, control documentation, inspection criteria, and data-collection logic for the selected technology.
- **Engineering analysis:** Define process flow, raw materials, tools, fixtures, gauges, SOPs, quality-control process documentation, and MES/ERP data requirements.
- **Engineering synthesis: organisation:** Process engineering, quality assurance, automation/MES team, maintenance, and operations representatives.
- **Engineering synthesis: information:** Detailed process charts, control plan, SOP draft, inspection criteria, MES/ERP data requirements, pilot-validation evidence.
- **Engineering synthesis: resource:** SOP templates, quality tools, metrology resources, MES/ERP interface definitions, predictive-maintenance inputs, digital-twin/simulation support.
- **Engineering evaluation:** Process plan and control documentation validated for pilot testing with a human-approved SOP baseline.
- **Representative tools/methods:** process chart; SOP; control plan; inspection criteria; MES/ERP; predictive maintenance; digital twin
- **Expected deliverables/tollgates:** detailed process chart; required tools/fixtures/gauges list; SOP draft; quality-control documentation; pilot-validation package
- **I5S cues:** {'Social': ['operator input to SOPs'], 'Sustainable': ['process efficiency'], 'Sensing': ['data collection logic'], 'Smart': ['digital twin/predictive maintenance'], 'Safe': ['risk review for selected technology']}
- **Source basis:** Book ch.6, Advanced Development: Process Plan; Book ch.6, Ten Core Technologies
- **Prompt priority:** Develop process plan, SOPs, control documentation, inspection criteria, and data-collection logic for the selected technology.

### Technology Transfer — Launching
- **Core activity:** Set up, test, qualify, and ramp up equipment; conduct pilot production and first-article inspection.
- **Engineering analysis:** Install, qualify, test, and ramp up equipment through installation qualification, pilot production, first-article inspection, and deviation review.
- **Engineering synthesis: organisation:** Commissioning team, operations, maintenance, quality, supplier technical support, and process engineering.
- **Engineering synthesis: information:** Installation qualification report, calibration records, ramp-up checklist, first-article inspection report, deviation log, approval evidence.
- **Engineering synthesis: resource:** Installation tools, calibration equipment, pilot-line resources, inspection systems, MES/SCADA data, digital-twin/simulation support.
- **Engineering evaluation:** Equipment readiness confirmed through installation qualification, first-article inspection, and human-approved production ramp-up decision.
- **Representative tools/methods:** installation qualification; calibration; pilot production; first-article inspection; ramp-up checklist; MES/SCADA
- **Expected deliverables/tollgates:** operational equipment; test result/validation report; ramp-up checklist; first-article inspection report; initial production output
- **I5S cues:** {'Social': ['training during ramp-up'], 'Sustainable': ['energy/process monitoring'], 'Sensing': ['SCADA/MES data'], 'Smart': ['digital twin support'], 'Safe': ['commissioning safety checks']}
- **Source basis:** Book ch.6, Launching: Equipment Set-up and Ramp-up of Production
- **Prompt priority:** Set up, test, qualify, and ramp up equipment; conduct pilot production and first-article inspection.

### Technology Transfer — End-of-Life
- **Core activity:** Retire or adapt equipment, document performance and failures, and update lifecycle-learning records for future transfers.
- **Engineering analysis:** Assess equipment retirement/adaptation options, failure history, maintainability, reuse/recycling potential, and lessons learned.
- **Engineering synthesis: organisation:** Asset management, maintenance, sustainability/logistics, engineering knowledge management, and operations.
- **Engineering synthesis: information:** End-of-life report, failure-history log, lessons-learned archive, digital-twin lifecycle update, reuse/recycling plan.
- **Engineering synthesis: resource:** Maintenance records, MES/ERP or historian data, decommissioning tools, recycling/reuse partners, digital-twin records.
- **Engineering evaluation:** Closure approved after performance history, failure lessons, and reuse or retirement decisions are documented for future technology-transfer learning.
- **Representative tools/methods:** failure-history log; digital twin update; lessons learned; asset recovery; equipment adaptation
- **Expected deliverables/tollgates:** end-of-life report; lessons-learned archive; digital-twin lifecycle update; reuse/recycling plan
- **I5S cues:** {'Social': ['knowledge retention'], 'Sustainable': ['equipment reuse/recycling'], 'Sensing': ['performance/failure records'], 'Smart': ['lifecycle learning'], 'Safe': ['safe decommissioning']}
- **Source basis:** Book ch.6, End of Life: 5S; Book ch.2, End-of-life 10R and 5S
- **Prompt priority:** Retire or adapt equipment, document performance and failures, and update lifecycle-learning records for future transfers.

### Facility Design — Ideation
- **Core activity:** Specify product, process, production-volume, and facility requirements to initiate new or transformed facility design.
- **Engineering analysis:** Define product/process requirements, demand scenarios, facility constraints, site assumptions, and human-factors criteria.
- **Engineering synthesis: organisation:** Manufacturing engineering, operations, product engineering, facilities, safety, ergonomics, sustainability, and supply-chain stakeholders.
- **Engineering synthesis: information:** Facility requirements brief, process-flow assumptions, production-volume forecasts, site constraints, ergonomic criteria, sustainability criteria.
- **Engineering synthesis: resource:** Process data, product documentation, site information, stakeholder workshops, preliminary capacity assumptions, ergonomic assessment data.
- **Engineering evaluation:** Facility requirements brief approved as the ideation gate for manufacturing-system and equipment selection.
- **Representative tools/methods:** facility requirements brief; process flow diagram; demand scenario; stakeholder workshop; ergonomic assessment
- **Expected deliverables/tollgates:** facility requirements brief; initial layout concept; manufacturing process specification; capacity assumption record
- **I5S cues:** {'Social': ['ergonomic criteria'], 'Sustainable': ['sustainability criteria'], 'Sensing': ['future monitoring needs'], 'Smart': ['scenario planning'], 'Safe': ['site/safety constraints']}
- **Source basis:** Book ch.6, Idea | Imagination: Facility Design Product and Manufacturing Process Specification
- **Prompt priority:** Specify product, process, production-volume, and facility requirements to initiate new or transformed facility design.

### Facility Design — Basic Development
- **Core activity:** Select manufacturing systems, equipment, utilities, and facility concepts that satisfy process and capacity needs.
- **Engineering analysis:** Evaluate available technology, machines, material handling equipment, tooling, jigs/fixtures, utility needs, and make-or-buy logic.
- **Engineering synthesis: organisation:** Facilities engineering, manufacturing engineering, procurement, operations, safety, sustainability, and finance/cost review.
- **Engineering synthesis: information:** Equipment/system specification sheet, utility matrix, system alternatives, make-or-buy logic, updated facility layout assumptions.
- **Engineering synthesis: resource:** Supplier data, equipment catalogues, utility-load calculations, layout tools, capacity models, technical-economic benchmarking templates.
- **Engineering evaluation:** Facility concept and equipment/system selection reviewed against process, capacity, safety, sustainability, and utility requirements.
- **Representative tools/methods:** equipment selection; utility matrix; make-or-buy analysis; capacity model; technical-economic benchmarking
- **Expected deliverables/tollgates:** manufacturing system specification; equipment/technology selection report; utility matrix; updated facility layout plan
- **I5S cues:** {'Social': ['operator/facility requirements'], 'Sustainable': ['utilities and energy'], 'Sensing': ['monitoring infrastructure needs'], 'Smart': ['capacity modelling'], 'Safe': ['equipment/facility safety needs']}
- **Source basis:** Book ch.6, Basic Development: Manufacturing System Specification
- **Prompt priority:** Select manufacturing systems, equipment, utilities, and facility concepts that satisfy process and capacity needs.

### Facility Design — Advanced Development
- **Core activity:** Design shop-floor layout, capacity, material flow, manufacturing strategy, storage/buffers, human movement, safety zones, utilities, and digital infrastructure.
- **Engineering analysis:** Develop integrated layout and manufacturing strategy using capacity plans, material-flow logic, station/cell configuration, safety zoning, and simulation evidence.
- **Engineering synthesis: organisation:** Industrial engineering, facilities, operations, maintenance, safety, IT/OT, logistics, quality, and sustainability teams.
- **Engineering synthesis: information:** Detailed layout drawings, capacity analysis, material-flow map, storage/buffer plan, safety-zone plan, utility plan, digital-infrastructure requirements.
- **Engineering synthesis: resource:** Layout simulation, DES or capacity tools, CAD/BIM resources, safety assessment tools, logistics data, utility-planning inputs.
- **Engineering evaluation:** Layout and capacity design validated through flow, safety, utility, and operational-readiness review.
- **Representative tools/methods:** CAD/BIM layout; discrete-event simulation; capacity planning; material-flow analysis; safety zoning; utility planning; FMEA/HAZOP
- **Expected deliverables/tollgates:** detailed layout drawing; capacity simulation report; material-flow diagram; safety-zone plan; utility plan; risk assessment
- **I5S cues:** {'Social': ['human movement paths'], 'Sustainable': ['energy/resource-aware layout'], 'Sensing': ['data-flow requirements'], 'Smart': ['simulation/digital twin'], 'Safe': ['safety zones/risk assessment']}
- **Source basis:** Book ch.6, Advanced Development: Manufacturing System Design; Book ch.6, Ten Core Technologies
- **Prompt priority:** Design shop-floor layout, capacity, material flow, manufacturing strategy, storage/buffers, human movement, safety zones, utilities, and digital infrastructure.

### Facility Design — Launching
- **Core activity:** Build, install, commission, calibrate, validate, ramp up, and evaluate the facility and manufacturing systems.
- **Engineering analysis:** Execute construction/build, equipment installation, commissioning, calibration, site acceptance testing, operator readiness, safety sign-off, and ramp-up evaluation.
- **Engineering synthesis: organisation:** Construction/facilities team, equipment suppliers, commissioning team, operations, maintenance, quality, safety, IT/OT, and training coordinators.
- **Engineering synthesis: information:** Installation records, commissioning checklist, site acceptance test report, ramp-up evaluation checklist, training records, safety sign-off, first-article inspection records.
- **Engineering synthesis: resource:** Construction resources, installation tools, commissioning protocols, calibration equipment, training materials, MES/SCADA infrastructure, inspection tools.
- **Engineering evaluation:** Facility readiness confirmed through commissioning, site acceptance, training, safety sign-off, and human-approved ramp-up decision.
- **Representative tools/methods:** commissioning; calibration; site acceptance testing; operator readiness; safety sign-off; MES/SCADA; first-article inspection
- **Expected deliverables/tollgates:** constructed/operational facility; commissioning records; site acceptance test report; ramp-up evaluation checklist; performance evaluation report
- **I5S cues:** {'Social': ['operator readiness/training'], 'Sustainable': ['commissioning resource data'], 'Sensing': ['calibrated monitoring infrastructure'], 'Smart': ['MES/SCADA readiness'], 'Safe': ['LOTO/safety sign-off']}
- **Source basis:** Book ch.6, Launching: Facility Construction, Set-up, and Ramp-up of Production
- **Prompt priority:** Build, install, commission, calibrate, validate, ramp up, and evaluate the facility and manufacturing systems.

### Facility Design — End-of-Life
- **Core activity:** Audit facility for reuse, reconfiguration, transformation, decommissioning, asset recovery, reverse logistics, and lessons learned.
- **Engineering analysis:** Assess reuse, transformation, decommissioning, asset recovery, reverse logistics, environmental impact, and facility adaptability options.
- **Engineering synthesis: organisation:** Facilities, asset management, sustainability, finance, operations, safety, logistics, and decommissioning/reconfiguration partners.
- **Engineering synthesis: information:** Facility audit, asset inventory, reconfiguration options, decommissioning plan, sustainability assessment, reverse-logistics plan, lessons learned.
- **Engineering synthesis: resource:** Asset records, maintenance data, facility drawings, environmental records, audit checklists, decommissioning/reconfiguration partners.
- **Engineering evaluation:** Reuse, reconfiguration, or decommissioning decision approved with documented asset, safety, and sustainability evidence.
- **Representative tools/methods:** facility audit; asset inventory; reconfiguration study; decommissioning plan; reverse logistics; lessons learned
- **Expected deliverables/tollgates:** reuse/decommissioning audit report; asset inventory; reverse-logistics plan; reconfiguration option report; lessons-learned repository
- **I5S cues:** {'Social': ['community/worker transition'], 'Sustainable': ['asset recovery/reuse'], 'Sensing': ['asset/performance records'], 'Smart': ['reconfiguration protocols'], 'Safe': ['safe decommissioning']}
- **Source basis:** Book ch.6, End of Life: 5S; Book ch.2, End-of-life 10R and 5S
- **Prompt priority:** Audit facility for reuse, reconfiguration, transformation, decommissioning, asset recovery, reverse logistics, and lessons learned.
