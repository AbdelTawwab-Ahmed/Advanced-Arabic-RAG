import time
import asyncio
import pandas as pd
import streamlit as st
import re

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.evaluation.batch_runner import PIPELINE, run_one
from src.evaluation.evaluator import evaluate_answer


st.set_page_config(page_title="Arabic RAG Evaluator", layout="wide")

# Right-to-left rendering for Arabic text areas/outputs — readability matters for a professional tool
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap');

html, body, [class*="css"], textarea, .stMarkdown, .stDataFrame {
    font-family: 'Tajawal', sans-serif;
    direction: rtl;
    text-align: right;
}

.source-card {
    background-color: #1A1D24;
    border-right: 4px solid #4FD1C5;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
}
.source-header {
    font-weight: 700;
    color: #4FD1C5;
    margin-bottom: 6px;
    display: flex;
    justify-content: space-between;
}
.source-score {
    font-weight: 400;
    color: #999;
    font-size: 0.85em;
}
.source-text {
    font-size: 0.92em;
    color: #C9C9C9;
    line-height: 1.6;
}

.citation-badge {
    background-color: #4FD1C5;
    color: #0E1117;
    font-weight: 700;
    padding: 1px 7px;
    border-radius: 5px;
    font-size: 0.85em;
}

</style>
""", unsafe_allow_html=True)

st.title("Arabic RAG — Evaluation UI")
tab1, tab2 = st.tabs(["Single Question", "Batch Evaluation"])

# ---------------- Citation Helper ----------------

def highlight_citations(text: str) -> str:
    return re.sub(r"\[(\d+)\]", r'<span class="citation-badge">[\1]</span>', text)

# ---------------- Single Question ----------------
with tab1:
    question = st.text_area("Question", height=80, key="single_question")
    ground_truth = st.text_area("Ground truth (optional — enables metrics)", height=100, key="single_gt")

    if st.button("Run", type="primary", key="run_single"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Running pipeline..."):
                result = PIPELINE.invoke({"query": question})

            st.subheader("Route")
            st.write(f"**Technique:** {result['technique']}")
            st.caption(result["reason"])

            st.subheader("Answer")
            st.markdown(highlight_citations(result["answer"]), unsafe_allow_html=True)

            # with st.expander("Retrieved & reranked chunks"):
            #     for c in result["reranked"]:
            #         st.markdown(f"**{c['source_pdf']} — page {c['page_number']}** (rerank score: {c.get('rerank_score', 0):.4f})")
            #         st.text(c["text"][:500])

            st.subheader("📚 Sources")
            st.caption("Numbers below match the [n] citations in the answer above.")
            for i, c in enumerate(result["reranked"], 1):
                with st.container():
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-header">[{i}] {c['source_pdf']} — page {c['page_number']}
                        <span class="source-score">Rerank Score: {c.get('rerank_score', 0):.3f}</span></div>
                        <div class="source-text">{c['text'][:600]}{'...' if len(c['text']) > 600 else ''}</div>
                    </div>
                    """, unsafe_allow_html=True)

            cost = (result["input_tokens"] / 1_000_000 * config.GEMINI_PRICE_PER_1M_INPUT) + \
                   (result["output_tokens"] / 1_000_000 * config.GEMINI_PRICE_PER_1M_OUTPUT)

            c1, c2, c3 = st.columns(3)
            c1.metric("Input tokens", result["input_tokens"])
            c2.metric("Output tokens", result["output_tokens"])
            c3.metric("Cost (USD)", f"${cost:.6f}")

            if ground_truth.strip():
                with st.spinner("Evaluating (Ragas)..."):
                    contexts = [c["text"] for c in result["reranked"]]
                    scores = asyncio.run(evaluate_answer(
                        question=question, answer=result["answer"], contexts=contexts,
                        ground_truth=ground_truth,
                        input_tokens=result["input_tokens"], output_tokens=result["output_tokens"],
                    ))
                st.subheader("Evaluation Metrics")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Context Relevance", f"{scores['context_relevance']:.2f}")
                m2.metric("Faithfulness", f"{scores['faithfulness']:.2f}")
                m3.metric("Answer Relevance", f"{scores['answer_relevance']:.2f}")
                m4.metric("Correctness", f"{scores['correctness']:.2f}")
            else:
                st.info("Add a ground truth above to compute evaluation metrics.")

# ---------------- Batch Evaluation ----------------
with tab2:
    st.write("Add rows below, or upload a CSV with columns: `question`, `ground_truth`.")

    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        df_input = pd.read_csv(uploaded)
    else:
        df_input = pd.DataFrame({"question": [""], "ground_truth": [""]})

    edited_df = st.data_editor(df_input, num_rows="dynamic", use_container_width=True, key="batch_editor")

    if st.button("Run Batch Evaluation", type="primary", key="run_batch"):
        items = [
            {"question": str(r["question"]), "ground_truth": str(r["ground_truth"])}
            for _, r in edited_df.iterrows() if str(r["question"]).strip()
        ]
        if not items:
            st.warning("Add at least one question.")
        else:
            progress = st.progress(0.0, text="Starting...")
            rows = []
            for i, item in enumerate(items):
                progress.progress(i / len(items), text=f"Running {i+1}/{len(items)}: {item['question'][:40]}...")
                rows.append(asyncio.run(run_one(item)))
                if i < len(items) - 1:
                    time.sleep(config.EVAL_BATCH_DELAY_SECONDS)
            progress.progress(1.0, text="Done")
            st.session_state["results_df"] = pd.DataFrame(rows)

    if "results_df" in st.session_state:
        st.subheader("Results")
        st.dataframe(st.session_state["results_df"], use_container_width=True)
        csv_bytes = st.session_state["results_df"].to_csv(index=False).encode("utf-8-sig")
        st.download_button("Download results as CSV", csv_bytes, file_name="rag_eval_results.csv", mime="text/csv")