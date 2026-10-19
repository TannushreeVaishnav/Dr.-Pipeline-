from pathlib import Path
import os

import streamlit as st

from classifier import CATEGORIES, classify_log, load_labeled_logs, train_classifier
from diagnosis import DiagnosisError, generate_diagnosis
from log_cleaner import clean_log, decode_uploaded_log, validate_log_size
from retrieval import load_knowledge_base, retrieve


ROOT = Path(__file__).resolve().parent
KNOWLEDGE_BASE = load_knowledge_base(ROOT / "data" / "knowledge_base.json")


@st.cache_resource
def get_classifier(records: list[dict[str, str]]):
    return train_classifier(records)


st.set_page_config(page_title="PipelineDoctor")
st.title("PipelineDoctor")
st.caption("Find source-backed guidance for ML pipeline failures.")

input_mode = st.radio("Log source", ["Paste or upload", "DVC demo log"], horizontal=True)

if input_mode == "DVC demo log":
    raw_log = (ROOT / "examples" / "dvc_pull_failed.log").read_text(encoding="utf-8")
    st.warning("Synthetic demo input in a realistic log format; this is not a production incident.")
else:
    uploaded = st.file_uploader("Upload a UTF-8 log", type=["log", "txt"])
    pasted = st.text_area(
        "Or paste a log",
        height=220,
        placeholder="Paste GitHub Actions, Docker, DVC, MLflow, or Evidently output…",
    )

    if uploaded is not None:
        try:
            raw_log = decode_uploaded_log(uploaded.name, uploaded.getvalue())
            st.caption("Using the uploaded file; pasted text is ignored while a file is selected.")
        except ValueError as error:
            st.error(str(error))
            raw_log = None
    else:
        raw_log = pasted

if raw_log is not None:
    try:
        validate_log_size(len(raw_log.encode("utf-8")))
    except ValueError as error:
        st.error(str(error))
    else:
        if raw_log.strip():
            cleaned = clean_log(raw_log)
            st.subheader("Cleaned log")
            st.text_area("Prepared log", cleaned, height=300, disabled=True)
            st.caption(
                "Recognizable timestamps and run IDs are redacted. Review the log for secrets before sharing."
            )
            st.subheader("Failure category")
            try:
                classifier = get_classifier(load_labeled_logs(ROOT / "data" / "labeled_logs.json"))
                probabilities = classify_log(cleaned, classifier)
            except (OSError, ValueError):
                st.error("The failure classifier could not load its labeled dataset. Check the dataset configuration.")
            else:
                if probabilities:
                    category = next(iter(probabilities))
                    st.metric(CATEGORIES[category], f"{probabilities[category]:.1%}")
                    st.caption(
                        "Uncalibrated model probability, trained on 20 synthetic logs. "
                        "This is not the probability that a diagnosis is correct."
                    )
                    with st.expander("All category probabilities"):
                        st.table([
                            {"Category": CATEGORIES[name], "Probability": f"{probability:.1%}"}
                            for name, probability in probabilities.items()
                        ])
                else:
                    st.info("No learned vocabulary matched this log, so no failure category was assigned.")
            st.subheader("Matching knowledge")
            st.caption("The current knowledge base covers DVC. Category predictions do not establish source coverage.")
            matches = retrieve(cleaned, KNOWLEDGE_BASE)
            if matches:
                chunks_by_id = {chunk["chunk_id"]: chunk for chunk in KNOWLEDGE_BASE}
                for match in matches:
                    chunk = match["chunk"]
                    with st.container(border=True):
                        st.markdown(f"**{chunk['chunk_id']} · {chunk['tool']} — {chunk['title']}**")
                        st.caption(f"Lexical relevance: {match['score']:.2f}; this is a ranking signal, not diagnosis confidence.")
                        st.markdown(f"**Symptom**  \n{chunk['symptom']}")
                        st.markdown(f"**Documented guidance**  \n{chunk['guidance']}")
                        st.link_button("Open source", chunk["source_url"])
            else:
                st.warning("No relevant knowledge-base entry matched this log; no diagnosis was produced.")

            api_key = os.getenv("OPENAI_API_KEY")
            model = os.getenv("PIPELINEDOCTOR_OPENAI_MODEL", "")
            if not api_key:
                st.info("Set OPENAI_API_KEY and PIPELINEDOCTOR_OPENAI_MODEL to enable generated diagnoses.")
            elif not model:
                st.info("Set PIPELINEDOCTOR_OPENAI_MODEL to a model available to your OpenAI API account.")
            elif matches:
                st.warning(
                    "Generating a diagnosis sends the cleaned log and retrieved evidence to OpenAI "
                    "and may incur API charges. Review the log for secrets first."
                )
                consent = st.checkbox("I reviewed this log and want to send it for diagnosis.")
                if st.button("Generate diagnosis", disabled=not consent):
                    from openai import OpenAI, OpenAIError

                    try:
                        with st.spinner("Generating a source-grounded diagnosis…"):
                            diagnosis = generate_diagnosis(
                                OpenAI(timeout=30, max_retries=1), model, cleaned, matches
                            )
                        if diagnosis.status == "insufficient_evidence":
                            st.warning("The retrieved sources do not support a diagnosis.")
                            st.write("Information needed:")
                            for item in diagnosis.missing_information:
                                st.write(f"- {item}")
                        else:
                            for title, claim in (
                                ("Root cause", diagnosis.root_cause),
                                ("Recommended fix", diagnosis.recommended_fix),
                            ):
                                st.markdown(f"**{title}**")
                                st.write(claim.text)
                                for citation_id in claim.citation_ids:
                                    cited = chunks_by_id[citation_id]
                                    st.link_button(
                                        f"{citation_id} · {cited['title']}", cited["source_url"]
                                    )
                    except DiagnosisError as error:
                        st.error(str(error))
                    except OpenAIError:
                        st.error("OpenAI could not complete the request. Check API access and try again.")
        else:
            st.info("Paste or upload a log to begin.")

st.caption("Logs stay in this app session unless you confirm an OpenAI diagnosis request. Nothing is sent automatically.")
