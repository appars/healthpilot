
import streamlit as st
from langchain_community.llms import Ollama
from rag_helper import retrieve_context

st.set_page_config(page_title="HealthPilot v1", page_icon="🩺", layout="wide")
st.title("🩺 HealthPilot v1")
st.caption("Educational Multi-Agent Healthcare Triage Assistant")

llm = Ollama(model="mistral")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Information")
    age = st.number_input("Age",1,120,55)
    symptoms = st.text_area("Symptoms","fever, cough")
    duration = st.text_input("Duration","3 days")
    conditions = st.text_input("Existing Conditions","diabetes")
    analyze = st.button("Analyze")

with col2:
    st.subheader("HealthPilot Analysis")
    if analyze:
        patient = f"Age:{age}\nSymptoms:{symptoms}\nDuration:{duration}\nConditions:{conditions}"
        with st.spinner("Analyzing..."):
            planner = "Retrieve guidance → Triage → Human review"
            context = retrieve_context(patient)
            triage = llm.invoke(f'''
You are HealthPilot.

DO NOT diagnose or prescribe.

Patient:
{patient}

Knowledge:
{context}

Provide:
1. Symptom summary
2. Possible category
3. Suggested urgency
4. Safety guidance
''')

        with st.expander("🧠 Planner Agent", expanded=True):
            st.write(planner)

        with st.expander("📚 Retriever Agent"):
            st.write(context)

        with st.expander("🩺 Triage Agent", expanded=True):
            st.write(triage)

        with st.expander("👨‍⚕️ Human Review Agent", expanded=True):
            st.warning("Doctor review required. Educational AI assistant only.")
