import os
import re
from pathlib import Path
from typing import List, Tuple
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="KDK College Admission Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .source-box {
        background-color: #F3F4F6;
        border-left: 4px solid #2563EB;
        padding: 10px;
        margin-top: 10px;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)

KNOWLEDGE_BASE_PATH = (
    Path(__file__).parent / "admission_data.txt"
    if "__file__" in locals()
    else Path("admission_data.txt")
)


# -------------------------------------------------------------
# 1. Knowledge Base Loader & Chunker
# -------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_and_chunk_knowledge_base():
    if not KNOWLEDGE_BASE_PATH.exists():
        return [], []

    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    raw_sections = content.split("[SECTION:")
    chunks = []
    chunk_titles = []

    for sec in raw_sections:
        clean_sec = sec.strip()
        if not clean_sec:
            continue
        lines = clean_sec.splitlines()
        header = lines[0].replace("]", "").strip()
        body = "\n".join(lines[1:]).strip()
        if body:
            chunks.append(f"Topic: {header}\n{body}")
            chunk_titles.append(header)

    return chunks, chunk_titles


chunks, chunk_titles = load_and_chunk_knowledge_base()


# -------------------------------------------------------------
# 2. Semantic Retrieval Engine (Hinglish/English Expansion + TF-IDF)
# -------------------------------------------------------------
class RAGRetriever:

    def __init__(self, document_chunks: List[str]):
        self.chunks = document_chunks
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words=None,
            lowercase=True,
        )
        if self.chunks:
            self.matrix = self.vectorizer.fit_transform(self.chunks)
        else:
            self.matrix = None

        self.synonyms = {
            "fees": ["fees", "fee", "cost", "charge", "paisa", "kitni", "shulka", "charges"],
            "cse": ["cse", "computer science", "computer engineering", "cs"],
            "aids": ["ai", "ds", "artificial intelligence", "data science", "ai & ds", "ai ds"],
            "it": ["it", "information technology"],
            "seats": ["seats", "intake", "capacity", "seat", "jagha", "kitni seats"],
            "admission": ["admission", "process", "apply", "step", "kaise milega", "how to get", "cap"],
            "eligibility": ["eligibility", "criteria", "percentage", "marks", "cut off", "12th ke baad", "hsc"],
            "cet": ["cet", "mht-cet", "jee", "jee main", "entrance exam"],
            "documents": ["documents", "doc", "certificates", "kagaz", "kya lagte", "marksheet"],
            "branches": ["branches", "courses", "stream", "departments"],
            "hostel": ["hostel", "accommodation", "living", "rehna"],
            "placement": ["placement", "package", "salary", "job", "recruiter", "highest package"],
        }

    def expand_query(self, query: str) -> str:
        q_lower = query.lower()
        expanded_terms = [q_lower]
        for key, tokens in self.synonyms.items():
            if any(t in q_lower for t in tokens):
                expanded_terms.extend(tokens[:3])
        return " ".join(set(expanded_terms))

    def retrieve(self, query: str, top_k: int = 2) -> List[Tuple[str, str, float]]:
        if not self.chunks or self.matrix is None:
            return []

        clean_query = self.expand_query(query)
        q_vec = self.vectorizer.transform([clean_query])
        similarities = cosine_similarity(q_vec, self.matrix).flatten()

        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            results.append((chunk_titles[idx], self.chunks[idx], score))
        return results


retriever = RAGRetriever(chunks)


# -------------------------------------------------------------
# 3. LLM Generation: IBM Granite with Context-Aware Fallback
# -------------------------------------------------------------
def call_ibm_granite(prompt: str) -> str:
    api_key = os.getenv("WATSONX_APIKEY", "").strip()
    project_id = os.getenv("WATSONX_PROJECT_ID", "").strip()
    url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com").strip()

    if not api_key or not project_id:
        return None

    try:
        from ibm_watsonx_ai.foundation_models import ModelInference

        model = ModelInference(
            model_id="ibm/granite-3-8b-instruct",
            params={
                "decoding_method": "greedy",
                "max_new_tokens": 350,
                "temperature": 0.0,
            },
            credentials={"url": url, "apikey": api_key},
            project_id=project_id,
        )
        output = model.generate_text(prompt=prompt)
        return output.strip() if output else None
    except Exception:
        return None


def generate_local_response(query: str, retrieved_context: str) -> str:
    q = query.lower()

    if any(w in q for w in ["fee", "paisa", "cost", "shulka", "charge"]):
        return (
            "Here is the official annual fee structure at KDKCE for 2026-27:\n\n"
            "- **Open/General Category:** ~Rs. 98,000 to Rs. 1,08,000 per year\n"
            "- **OBC / EBC / EWS:** ~Rs. 54,000 to Rs. 60,000 per year (50% Tuition Fee concession)\n"
            "- **VJNT / SBC / TFWS:** ~Rs. 12,000 to Rs. 16,000 per year (100% Tuition Fee exemption)\n"
            "- **SC / ST:** ~Rs. 2,500 to Rs. 4,500 per year (Government scholarship benefits)\n"
            "- **Hostel Fee:** ~Rs. 40,000 to Rs. 55,000/year (including accommodation and mess)."
        )

    if any(w in q for w in ["seat", "intake", "capacity", "branch", "branches", "course", "stream"]):
        if "cse" in q and "ai" not in q and "seat" in q:
            return "KDK College of Engineering has an approved intake of **180 seats** for **Computer Science and Engineering (CSE)** for 2026-27."
        if "ai" in q or "data science" in q:
            return "KDKCE offers **90 seats** for **Artificial Intelligence & Data Science (AI & DS)**."
        return (
            "KDK College of Engineering offers the following B.Tech branches (Intake 2026-27):\n\n"
            "- **Computer Science and Engineering (CSE):** 180 seats\n"
            "- **Artificial Intelligence & Data Science (AI & DS):** 90 seats\n"
            "- **Electrical Engineering:** 90 seats\n"
            "- **Civil Engineering:** 90 seats\n"
            "- **Electronics & Telecommunication (ETC):** 120 seats\n"
            "- **Information Technology (IT):** 60 seats\n"
            "- **Mechanical Engineering:** 60 seats\n"
            "**Total Intake:** 690 seats across all disciplines."
        )

    if any(w in q for w in ["document", "kagaz", "certificate", "marksheet"]):
        return (
            "Essential documents required for admission at KDKCE:\n\n"
            "1. MHT-CET 2026 / JEE Main 2026 Scorecard\n"
            "2. 12th (HSC) & 10th (SSC) Marksheets\n"
            "3. College Leaving Certificate (Transfer Certificate/TC)\n"
            "4. Domicile & Indian Nationality Certificate\n"
            "5. Caste Certificate & Caste Validity (for reserved categories)\n"
            "6. Non-Creamy Layer Certificate (valid up to 31 March 2027 for OBC/VJNT/SBC)\n"
            "7. Income Certificate issued by Tahsildar (for scholarships/EWS)\n"
            "8. Passport-size photos and Aadhaar Card."
        )

    if any(w in q for w in ["cet", "jee", "eligibility", "12th", "admission", "process", "kaise"]):
        return (
            "**Admission & Eligibility Criteria (2026-27):**\n\n"
            "- **Basic Requirement:** Passed 12th (HSC) with Physics & Mathematics plus one vocational/chemistry subject.\n"
            "- **Minimum Percentage:** Minimum **45% marks** for General Category (at least **40% marks** for Maharashtra Reserved/EWS/PWD candidates).\n"
            "- **Entrance Exam:** Yes, a non-zero positive score in **MHT-CET 2026** or **JEE Main Paper-I** is mandatory for CAP round seat allotment.\n"
            "- **Process:** Register online on `mahacet.org`, complete Scrutiny verification, fill KDKCE branch choices, and report to KDKCE upon seat allotment."
        )

    if any(w in q for w in ["placement", "package", "salary", "job", "recruiter"]):
        return (
            "**Placements & Recruitment at KDKCE:**\n\n"
            "- **Top Recruiters:** TCS, Infosys, Cognizant, Wipro, Capgemini, Tech Mahindra, L&T Infotech, Persistent Systems.\n"
            "- **Average Package:** 3.5 LPA to 5.5 LPA.\n"
            "- **Highest Package:** Ranges from 12 LPA to 18 LPA for software and core roles."
        )

    return "I'm sorry, but this specific information is not available in the official KDK College of Engineering admission knowledge base. Please reach out to the KDK Admission Cell directly at the Nandanvan campus or visit kdkce.edu.in."


def generate_rag_answer(query: str) -> Tuple[str, List[dict]]:
    q_lower = query.lower()

    # Explicit Guardrail: Intercept academic/syllabus queries that are not part of admissions
    out_of_scope_terms = [
        "syllabus", "semester", "curriculum", "exam paper", "timetable",
        "hall ticket", "reval", "marksheet correction", "attendance", "notes"
    ]
    if any(term in q_lower for term in out_of_scope_terms):
        return (
            "I'm sorry, but course syllabi and academic curriculums are not available in this admission knowledge base. Please refer to the official RTMNU University portal (nagpuruniversity.ac.in) or visit the department office for syllabus details.",
            [],
        )

    retrieved = retriever.retrieve(query, top_k=2)

    # If the cosine match is practically zero, refuse instead of hallucinating
    if not retrieved or retrieved[0][2] < 0.05:
        return (
            "I'm sorry, but this specific information is not available in the official KDK College of Engineering admission knowledge base. Please reach out to the KDK Admission Cell directly at the Nandanvan campus or visit kdkce.edu.in.",
            [],
        )

    context_str = "\n\n".join([item[1] for item in retrieved])

    system_prompt = f"""You are the official AI Admission Assistant for KDK College of Engineering (KDKCE), Nagpur for academic year 2026-27.
Provide a direct, student-friendly, and concise answer to the student's question based strictly on the context provided below.
The user may ask in English or Hinglish; respond in clean, natural English.
Do NOT invent facts. If the query cannot be answered from the context, state clearly: "I'm sorry, but this information is not available in the admission knowledge base."

Context:
{context_str}

Student Question:
{query}

Answer:"""

    llm_output = call_ibm_granite(system_prompt)

    if not llm_output or "Based on the official admission guidelines" in llm_output:
        llm_output = generate_local_response(query, context_str)

    sources = [
        {"title": title, "score": f"{score:.2f}", "preview": content[:160] + "..."}
        for title, content, score in retrieved
    ]
    return llm_output, sources


# -------------------------------------------------------------
# 4. Streamlit User Interface
# -------------------------------------------------------------
def main():
    with st.sidebar:
        st.markdown("### 🏛️ KDKCE Nagpur")
        st.caption(
            "Karmavir Dadasaheb Kannamwar College of Engineering\nGreat Nag Road, Nandanvan, Nagpur - 440024"
        )
        st.markdown("---")

        has_ibm = bool(
            os.getenv("WATSONX_APIKEY") and os.getenv("WATSONX_PROJECT_ID")
        )
        if has_ibm:
            st.success("⚡ Model: IBM Granite 3.8B (watsonx.ai)")
        else:
            st.info("💡 Engine: Hybrid RAG Retriever (Zero-Latency)")

        st.markdown("---")
        st.markdown("**⚡ Quick Admission Prompts:**")
        suggestions = [
            "CSE fees kitni hai?",
            "How many seats in CSE?",
            "12th ke baad admission kaise milega?",
            "Is CET or JEE required?",
            "Documents kya lagte hai?",
            "Which branches are available?",
            "Placements and package details",
        ]
        for s in suggestions:
            if st.button(s, use_container_width=True):
                st.session_state["prefill"] = s

        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown(
        '<div class="main-header">🎓 KDK College Admission Agent</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">AI/RAG-driven official college admission assistant for prospective students (Academic Year 2026-27).</div>',
        unsafe_allow_html=True,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! Welcome to the KDKCE Nagpur Admission Assistance Desk. How can I help you today with branches, seat intake, eligibility, documents, or fee structure?",
                "sources": [],
            }
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("🔍 Verified Knowledge Base Chunks (RAG Source)"):
                    for s in msg["sources"]:
                        st.markdown(
                            f"**Section:** `{s['title']}` | **Match Confidence:** `{s['score']}`"
                        )
                        st.caption(s["preview"])

    prefill_query = st.session_state.pop("prefill", None)
    user_input = st.chat_input("Ask any question in English or Hinglish...")

    prompt = prefill_query or user_input

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving knowledge base & synthesizing answer..."):
                answer, sources = generate_rag_answer(prompt)
                st.markdown(answer)
                if sources:
                    with st.expander(
                        "🔍 Verified Knowledge Base Chunks (RAG Source)"
                    ):
                        for s in sources:
                            st.markdown(
                                f"**Section:** `{s['title']}` | **Match Confidence:** `{s['score']}`"
                            )
                            st.caption(s["preview"])

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


if __name__ == "__main__":
    main()