
import streamlit as st
from langchain_community.llms import Ollama
from rag_helper import retrieve_context

st.set_page_config(page_title="HealthPilot AI", page_icon="🩺", layout="wide")

st.markdown("""
<style>
.stApp {background-color:#F7FAFC;}
.big-title {
    font-size:42px;
    font-weight:700;
    color:#1976D2;
}
.subtitle{
    color:#607D8B;
    font-size:16px;
}
.agent-card{
    border-radius:12px;
    padding:10px;
}
.stButton>button {
    background-color:#009688;
    color:white;
    border-radius:10px;
    height:3em;
    width:100%;
    font-size:18px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🩺 HealthPilot AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI‑Assisted Healthcare Triage • Human‑in‑Loop • RAG Powered</div>', unsafe_allow_html=True)

llm = Ollama(model="mistral")

with st.sidebar:
    st.header("⚙️ HealthPilot Control Center")
    st.success("🟢 Local AI Running")
    st.info("Model: Mistral (Ollama)")
    st.markdown("### 🤖 Agent Stack")
    st.markdown("""
🧠 Planner Agent  
📚 Retriever Agent  
🩺 Triage Agent  
👨‍⚕️ Clinical Governance Agent
""")

col1,col2 = st.columns([1,1.2])

with col1:
    st.subheader("🧾 Patient Information")
    age = st.number_input("Age",1,120,55)
    symptoms = st.text_area("Symptoms","fever, cough")
    duration = st.text_input("Duration","3 days")
    conditions = st.text_input("Existing Conditions","diabetes")
    analyze = st.button("🩺 Analyze Patient")

with col2:
    st.subheader("🏥 HealthPilot Analysis")

    if analyze:
        patient = f"""
Age:{age}
Symptoms:{symptoms}
Duration:{duration}
Conditions:{conditions}
"""

        with st.spinner("HealthPilot reasoning..."):

            planner = f"""
Reasoning Plan

1. Understand symptoms
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

Provide:
1 Symptom Summary
2 Possible Category
3 Suggested Urgency
4 Safety Guidance
""")

            governance = llm.invoke(f"""
You are Clinical Governance Agent.

Review the following:

Planner:
{planner}

Retrieved Guidance:
{context}

Triage:
{triage}

Create:

1 Clinical Summary
2 Risk Considerations
3 Suggested Urgency
4 Governance Review
5 Doctor Review Requirement

Keep concise and professional.
""")

        with st.expander("🧠 Planner Agent", expanded=False):
            st.markdown(planner)

        with st.expander("📚 Retriever Agent", expanded=False):
            st.markdown(context)

        with st.expander("🩺 Triage Agent", expanded=False):
            st.markdown(triage)

        with st.expander("👨‍⚕️ Clinical Governance Agent", expanded=True):
            st.markdown(governance)

        st.warning("Educational AI assistant only. Clinical diagnosis requires healthcare professionals.")
