
import streamlit as st
from langchain_community.llms import Ollama
from rag_helper import retrieve_context
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import tempfile, os

st.set_page_config(page_title="HealthPilot AI", page_icon="🩺", layout="wide")

st.markdown("""
<style>
.stApp {background-color:#F6F9FC;}
.big-title {font-size:48px;font-weight:800;color:#1565C0;}
.subtitle{color:#607D8B;font-size:17px;margin-bottom:18px;}
.workflow {background:#E3F2FD;padding:12px;border-radius:12px;border-left:5px solid #1976D2;font-size:16px;font-weight:600;margin-bottom:15px;}
.final-card{background:#E8F5E9;border-left:6px solid #2E7D32;padding:18px;border-radius:12px;margin-top:15px;}
.gov-card{background:#FFF8E1;border-left:6px solid #F9A825;padding:16px;border-radius:12px;margin-top:10px;}
.stButton>button {
background-color:#00897B !important;color:white !important;border-radius:12px;
height:3.2em;width:100%;font-size:18px;font-weight:700;border:none;
box-shadow:0 3px 10px rgba(0,0,0,0.15);}
.sidebar-status{font-size:15px;line-height:1.9;}
</style>
""", unsafe_allow_html=True)

llm = Ollama(model="mistral")

def create_pdf(patient, final_text):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(tmp.name)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("HealthPilot AI Clinical Summary", styles['Title']))
    story.append(Spacer(1,12))
    story.append(Paragraph("<b>Patient Information</b>", styles['Heading2']))
    story.append(Paragraph(patient.replace("\n","<br/>"), styles['BodyText']))
    story.append(Spacer(1,10))
    story.append(Paragraph("<b>Final Guidance</b>", styles['Heading2']))
    story.append(Paragraph(final_text.replace("\n","<br/>"), styles['BodyText']))
    story.append(Spacer(1,10))
    story.append(Paragraph(
        "Educational AI guidance only. Not a diagnosis. Healthcare professional review required.",
        styles['Italic']))
    doc.build(story)
    return tmp.name

st.markdown('<div class="big-title">🩺 HealthPilot AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI‑Assisted Healthcare Triage • Human‑in‑Loop • RAG Powered • Multi‑Agent Clinical Workflow</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ HealthPilot Control Center")
    st.success("🟢 Local AI Running")
    st.info("Model: Mistral (Ollama)")
    st.markdown("### 🤖 Agent Stack")
    st.markdown("""
<div class="sidebar-status">
🧠 Planner Agent 🟢<br>
🖼️ Vision Agent 🟢<br>
📚 Retriever Agent 🟢<br>
🩺 Triage Agent 🟢<br>
🏥 Clinical Governance Agent 🟢
</div>
""", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Educational Healthcare AI Prototype")

col1, col2 = st.columns([1,1.35])

with col1:
    st.subheader("📋 Patient Intake Form")
    age = st.number_input("Age",1,120,55)
    symptoms = st.text_area("Symptoms","fever, cough")
    duration = st.text_input("Duration","3 days")
    conditions = st.text_input("Existing Conditions","diabetes")
    uploaded_file = st.file_uploader("🖼️ Upload Medical Image (Educational)", type=["png","jpg","jpeg"])
    if uploaded_file:
        st.image(uploaded_file, width=250)
        st.info("Vision Agent: Image received for educational review.")
    if "running" not in st.session_state:
        st.session_state.running = False

    analyze = st.button(
        "🧠 Run HealthPilot Analysis",
        disabled=st.session_state.running,
        type="primary"
    )

with col2:
    st.subheader("🏥 HealthPilot Analysis")
    st.markdown('<div class="workflow">🧠 Plan → 🖼️ Vision → 📚 Retrieve → 🩺 Triage → 🏥 Govern → ✅ Synthesize</div>', unsafe_allow_html=True)

    if analyze:
        st.session_state.running = True
        st.rerun()

    if st.session_state.running:
        patient = f"Age: {age}\nSymptoms: {symptoms}\nDuration: {duration}\nConditions: {conditions}"
        with st.spinner("HealthPilot multi-agent reasoning in progress..."):
            planner = """1. Understand symptoms
2. Retrieve guidance
3. Perform educational triage
4. Apply governance review
5. Produce safe synthesis"""

            vision_note = "No image uploaded."
            if uploaded_file:
                vision_note = "Image uploaded. Educational image review suggested. Human clinical interpretation required."

            retrieval_query = patient + "\nVision: " + vision_note
            context = retrieve_context(retrieval_query)

            triage = llm.invoke(f"""
Patient:
{patient}

Context:
{context}

Give concise bullets:
- Symptom Summary
- Possible Category
- Suggested Urgency
- Safety Guidance
""")

            governance = llm.invoke(f"""
Review this case safely.

Triage:
{triage}

Image:
{vision_note}

Provide concise governance bullets.
""")

            final_summary = llm.invoke(f"""
Create bullet-form final guidance.

Triage:
{triage}

Governance:
{governance}

Format:

Possible Category
Urgency
Recommended Action
Safety Note
""")

        with st.expander("🧠 Planner Agent"):
            st.markdown(planner)
        with st.expander("📚 Retriever Agent"):
            st.markdown(context)
        with st.expander("🩺 Triage Agent"):
            st.markdown(triage)
        with st.expander("🏥 Clinical Governance Agent"):
            st.markdown(governance)
        if uploaded_file:
            with st.expander("🖼️ Vision Agent"):
                st.info(vision_note)

        st.markdown(f"""
<div class="final-card">
<h3>✅ HealthPilot Final Guidance</h3>
{final_summary}
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="gov-card">
⚠️ <b>Clinical Governance Notice</b><br><br>
✓ Educational AI guidance only<br>
✓ Not a clinical diagnosis<br>
✓ Healthcare professional review required
</div>
""", unsafe_allow_html=True)

        pdf_path = create_pdf(patient, final_summary)
        with open(pdf_path, "rb") as f:
            st.download_button(
                "📄 Download Clinical Summary PDF",
                f,
                file_name="HealthPilot_Clinical_Summary.pdf",
                mime="application/pdf"
            )

        st.session_state.running = False
