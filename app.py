
import streamlit as st
from langchain_community.llms import Ollama
from rag_helper import retrieve_context

st.set_page_config(page_title="HealthPilot AI", page_icon="🩺", layout="wide")

st.markdown("""
<style>
.stApp {
    background-color:#F6F9FC;
}
.big-title {
    font-size:48px;
    font-weight:800;
    color:#1565C0;
    margin-bottom:0px;
}
.subtitle{
    color:#607D8B;
    font-size:17px;
    margin-bottom:18px;
}
.workflow {
    background:#E3F2FD;
    padding:12px;
    border-radius:12px;
    border-left:5px solid #1976D2;
    font-size:16px;
    font-weight:600;
    margin-bottom:15px;
}
.agent-box{
    border-radius:14px;
    padding:10px;
    border:1px solid #E0E0E0;
}
.final-card{
    background:#E8F5E9;
    border-left:6px solid #2E7D32;
    padding:18px;
    border-radius:12px;
    margin-top:15px;
}
.gov-card{
    background:#FFF8E1;
    border-left:6px solid #F9A825;
    padding:16px;
    border-radius:12px;
    margin-top:10px;
}
.stButton>button {
    background-color:#00897B !important;
    color:white !important;
    border-radius:12px;
    height:3.2em;
    width:100%;
    font-size:18px;
    font-weight:700;
    border:none;
    box-shadow:0 3px 10px rgba(0,0,0,0.15);
}
.stButton>button:hover {
    background-color:#00796B !important;
    color:white !important;
}
.sidebar-status{
    font-size:15px;
    line-height:1.9;
}
</style>
""", unsafe_allow_html=True)

llm = Ollama(model="mistral")

# Header
st.markdown('<div class="big-title">🩺 HealthPilot AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI‑Assisted Healthcare Triage • Human‑in‑Loop • RAG Powered • Multi‑Agent Clinical Workflow</div>',
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("⚙️ HealthPilot Control Center")
    st.success("🟢 Local AI Running")
    st.info("Model: Mistral (Ollama)")

    st.markdown("### 🤖 Agent Stack")
    st.markdown(
        """
<div class="sidebar-status">
🧠 Planner Agent &nbsp;&nbsp;🟢<br>
📚 Retriever Agent &nbsp;&nbsp;🟢<br>
🩺 Triage Agent &nbsp;&nbsp;🟢<br>
🏥 Clinical Governance Agent &nbsp;&nbsp;🟢
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption("Educational Healthcare AI Prototype")

# Wider analysis panel
col1, col2 = st.columns([1, 1.35])

with col1:
    st.subheader("📋 Patient Intake Form")

    age = st.number_input("Age", 1, 120, 55)
    symptoms = st.text_area("Symptoms", "fever, cough")
    duration = st.text_input("Duration", "3 days")
    conditions = st.text_input("Existing Conditions", "diabetes")

    analyze = st.button("🧠 Run HealthPilot Analysis")

with col2:
    st.subheader("🏥 HealthPilot Analysis")

    st.markdown(
        """
<div class="workflow">
🧠 Plan → 📚 Retrieve → 🩺 Triage → 🏥 Govern → ✅ Synthesize
</div>
""",
        unsafe_allow_html=True,
    )

    if analyze:

        patient = f"""
Age: {age}
Symptoms: {symptoms}
Duration: {duration}
Conditions: {conditions}
"""

        with st.spinner("HealthPilot multi-agent reasoning in progress..."):

            planner = """
### Reasoning Plan

1. Understand patient symptoms  
2. Retrieve medical guidance  
3. Perform educational triage  
4. Apply clinical governance review  
5. Produce safe final synthesis  
"""

            context = retrieve_context(patient)

            triage = llm.invoke(f"""
You are HealthPilot Triage Agent.

DO NOT diagnose.
DO NOT prescribe.

Patient:
{patient}

Knowledge:
{context}

Provide concise output:

1. Symptom Summary
2. Possible Category
3. Suggested Urgency
4. Safety Guidance
""")

            governance = llm.invoke(f"""
You are Clinical Governance Agent.

Review:

Planner:
{planner}

Retrieved Guidance:
{context}

Triage:
{triage}

Provide concise bullet points:

1. Clinical Summary
2. Risk Considerations
3. Suggested Urgency
4. Governance Review
5. Doctor Review Requirement
""")

            final_summary = llm.invoke(f"""
You are HealthPilot Final Synthesizer.

Based on:
{triage}

and
{governance}

Create concise professional summary with:

Possible Category:
Urgency:
Recommended Action:
Safety Note:

Educational only.
""")

        with st.expander("🧠 Planner Agent", expanded=False):
            st.markdown(planner)

        with st.expander("📚 Retriever Agent", expanded=False):
            st.markdown(context)

        with st.expander("🩺 Triage Agent", expanded=False):
            st.markdown(triage)

        with st.expander("🏥 Clinical Governance Agent", expanded=False):
            st.markdown(governance)

        st.markdown(
            f"""
<div class="final-card">
<h4>✅ HealthPilot Final Guidance</h4>
{final_summary}
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
<div class="gov-card">
⚠️ <b>Clinical Governance Notice</b><br><br>
✓ Educational AI guidance only<br>
✓ Not a clinical diagnosis<br>
✓ Healthcare professional review required
</div>
""",
            unsafe_allow_html=True,
        )
