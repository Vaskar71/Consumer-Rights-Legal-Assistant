import streamlit as st
from backend.file_parser import extract_text_from_file, is_image, is_pdf
from backend.vision import describe_image
from backend.rag import analyze_case

st.set_page_config(
    page_title="Consumer Rights Assistant — Bangladesh",
    page_icon="⚖️",
    layout="centered"
)

st.title("⚖️ Consumer Rights Legal Assistant")
st.caption("Bangladesh — Based on the Consumer Rights Protection Act 2009")
st.info(
    "This tool provides **legal information only** based on the CRPA 2009. "
    "It does not constitute legal advice.",
    icon="ℹ️"
)

st.divider()

# ── INPUT 1: Incident Description ──────────────────────────────────────────────
st.subheader("1. Describe Your Complaint")
description = st.text_area(
    "What happened? Include what you ordered, what went wrong, and how the seller responded.",
    height=180,
    placeholder=(
        "Example: I ordered a Samsung smartphone from Daraz on [date] for BDT 25,000. "
        "It arrived with a cracked screen and missing charger. I contacted the seller "
        "3 times over 2 weeks via Daraz chat but received no response and no refund."
    )
)

# ── INPUT 2: Evidence Files (PDFs + Images) ────────────────────────────────────
st.subheader("2. Upload Evidence")
st.caption("Accepted: PDF, PNG, JPG, JPEG, DOCX, TXT — Images are analyzed by a vision AI, not OCR")
uploaded_files = st.file_uploader(
    "Upload receipts, screenshots, photos of damaged product, chat screenshots, invoices",
    accept_multiple_files=True,
    type=["pdf", "png", "jpg", "jpeg", "docx", "txt", "webp"]
)

# ── INPUT 3: Terms & Conditions ────────────────────────────────────────────────
st.subheader("3. Seller Terms & Conditions (Optional)")
st.caption("Provide the seller's T&C so the app can check if they violated their own policies")

tnc_input_method = st.radio(
    "How would you like to provide the T&C?",
    options=["Paste text", "Upload PDF"],
    horizontal=True
)

tnc_text = ""

if tnc_input_method == "Paste text":
    tnc_text = st.text_area(
        "Paste the seller or platform's Terms & Conditions here",
        height=150,
        placeholder="Paste Daraz, Shajgoj, or any other seller's T&C here..."
    )

elif tnc_input_method == "Upload PDF":
    tnc_file = st.file_uploader(
        "Upload T&C as PDF",
        type=["pdf"],
        key="tnc_uploader"
    )
    if tnc_file is not None:
        with st.spinner("Extracting text from T&C PDF..."):
            tnc_text = extract_text_from_file(tnc_file)
        if tnc_text.strip():
            st.success(f"T&C extracted successfully ({len(tnc_text)} characters)")
        else:
            st.warning("Could not extract text from the T&C PDF. Try pasting the text instead.")

st.divider()

# ── ANALYZE BUTTON ─────────────────────────────────────────────────────────────
if st.button("⚖️ Analyze My Case", type="primary", use_container_width=True):
    if not description.strip():
        st.warning("Please describe your complaint before analyzing.")
        st.stop()

    evidence_text = ""

    if uploaded_files:
        st.write("**Processing evidence files...**")
        progress = st.progress(0)
        total = len(uploaded_files)

        for i, f in enumerate(uploaded_files):
            progress.progress((i + 1) / total)

            if is_image(f.name):
                with st.spinner(f"Analyzing image: {f.name}"):
                    description_from_vision = describe_image(f)
                    evidence_text += f"\n--- Image Evidence: {f.name} ---\n{description_from_vision}\n"

            else:
                with st.spinner(f"Extracting text from: {f.name}"):
                    extracted = extract_text_from_file(f)
                    evidence_text += f"\n--- Document Evidence: {f.name} ---\n{extracted}\n"

        progress.empty()

    with st.spinner("Analyzing your case against the CRPA 2009..."):
        report = analyze_case(description, evidence_text, tnc_text)

    st.divider()
    st.subheader("📋 Legal Information Report")
    st.markdown(report)

    st.download_button(
        label="⬇️ Download Report as TXT",
        data=report,
        file_name="consumer_rights_report.txt",
        mime="text/plain",
        use_container_width=True
    )
