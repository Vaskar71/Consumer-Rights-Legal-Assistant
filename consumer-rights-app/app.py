import streamlit as st
from backend.file_parser import extract_text_from_file, is_image, is_pdf
from backend.vision import describe_image
from backend.rag import analyze_case

# --- Page Config ---
st.set_page_config(
    page_title="ConsumerShield — Bangladesh",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Polish ---
def inject_custom_css():
    try:
        with open("global.css", "r") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

# --- State Initialization ---
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "evidence_text" not in st.session_state:
        st.session_state.evidence_text = ""
    if "tnc_pasted_text" not in st.session_state:
        st.session_state.tnc_pasted_text = ""
    if "tnc_file_text" not in st.session_state:
        st.session_state.tnc_file_text = ""
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "processed_tnc_files" not in st.session_state:
        st.session_state.processed_tnc_files = set()

# --- Sidebar Components ---
def render_sidebar():
    with st.sidebar:
        st.title("🛡️ ConsumerShield")
        st.caption("Bangladesh CRPA 2009 Analysis")
        
        st.divider()
        
        st.subheader("📎 Evidence & Documents")
        st.caption("Upload receipts, chat screenshots, or damaged product photos.")
        uploaded_files = st.file_uploader(
            "Upload Evidence files",
            accept_multiple_files=True,
            type=["pdf", "png", "jpg", "jpeg", "docx", "txt", "webp"],
            label_visibility="collapsed"
        )
        
        # Process files instantly so they are ready for the chat
        if uploaded_files:
            new_files = [f for f in uploaded_files if f"{f.name}_{f.size}" not in st.session_state.processed_files]
            if new_files:
                with st.status("Processing new evidence...", expanded=True) as status:
                    for i, f in enumerate(new_files):
                        st.write(f"Analyzing {f.name}...")
                        try:
                            if is_image(f.name):
                                desc = describe_image(f)
                                st.session_state.evidence_text += f"\n--- Image Evidence: {f.name} ---\n{desc}\n"
                            else:
                                extracted = extract_text_from_file(f)
                                st.session_state.evidence_text += f"\n--- Document Evidence: {f.name} ---\n{extracted}\n"
                            st.session_state.processed_files.add(f"{f.name}_{f.size}")
                        except Exception as e:
                            st.error(f"Failed to process {f.name}: {str(e)}")
                    status.update(label="Evidence processed successfully!", state="complete", expanded=False)
            
            # Show processed count
            if st.session_state.processed_files:
                st.success(f"{len(st.session_state.processed_files)} files loaded and ready.")

        st.divider()
        
        st.subheader("📜 Terms & Conditions")
        st.caption("Optional: Provide seller policies to check for breaches.")
        
        tnc_tab1, tnc_tab2 = st.tabs(["Paste Text", "Upload File(s)"])
        
        with tnc_tab1:
            tnc_input = st.text_area(
                "Paste T&C text",
                height=150,
                placeholder="Paste Daraz, Shajgoj, or seller T&C here...",
                label_visibility="collapsed"
            )
            if tnc_input != st.session_state.tnc_pasted_text:
                st.session_state.tnc_pasted_text = tnc_input

        with tnc_tab2:
            uploaded_tnc = st.file_uploader(
                "Upload T&C documents",
                accept_multiple_files=True,
                type=["pdf", "png", "jpg", "jpeg", "docx", "txt", "webp"],
                key="tnc_uploader",
                label_visibility="collapsed"
            )
            if uploaded_tnc:
                new_tnc_files = [f for f in uploaded_tnc if f"{f.name}_{f.size}" not in st.session_state.processed_tnc_files]
                if new_tnc_files:
                    with st.status("Processing T&C files...", expanded=True) as status:
                        for f in new_tnc_files:
                            st.write(f"Analyzing {f.name}...")
                            try:
                                if is_image(f.name):
                                    desc = describe_image(f)
                                    st.session_state.tnc_file_text += f"\n--- T&C Image: {f.name} ---\n{desc}\n"
                                else:
                                    extracted = extract_text_from_file(f)
                                    st.session_state.tnc_file_text += f"\n--- T&C Document: {f.name} ---\n{extracted}\n"
                                st.session_state.processed_tnc_files.add(f"{f.name}_{f.size}")
                            except Exception as e:
                                st.error(f"Failed to process {f.name}: {str(e)}")
                        status.update(label="T&C files processed successfully!", state="complete", expanded=False)
                if st.session_state.processed_tnc_files:
                    st.success(f"{len(st.session_state.processed_tnc_files)} T&C files loaded.")

        st.divider()
        st.warning("⚠️ **Disclaimer:** This tool provides legal information, not legal advice.")
        
        if st.button("🗑️ Clear Chat & Data", use_container_width=True):
            st.session_state.messages = []
            st.session_state.evidence_text = ""
            st.session_state.tnc_pasted_text = ""
            st.session_state.tnc_file_text = ""
            st.session_state.processed_files = set()
            st.session_state.processed_tnc_files = set()
            st.rerun()

# --- Main Chat Interface ---
def render_chat_interface():
    # Use an empty container so we can dynamically clear it when the first message is sent
    welcome_container = st.empty()

    # Empty State
    if not st.session_state.messages:
        with welcome_container:
            st.markdown("""
                <div class="welcome-card">
                    <div class="welcome-badge">Bangladesh CRPA 2009</div>
                    <div class="welcome-title">ConsumerShield<br/>Your Rights, Explained Simply</div>
                    <div class="welcome-subtitle">
                        Describe what happened with a seller or product. I will look up the relevant laws and explain your rights in plain language — no legal jargon.
                    </div>
                    <div class="feature-grid">
                        <div class="feature-tile">
                            <div class="feature-icon">📄</div>
                            <div class="feature-label">Upload receipts, screenshots & invoices</div>
                        </div>
                        <div class="feature-tile">
                            <div class="feature-icon">⚖️</div>
                            <div class="feature-label">CRPA 2009 analysis in plain language</div>
                        </div>
                        <div class="feature-tile">
                            <div class="feature-icon">📜</div>
                            <div class="feature-label">Check if seller violated their own T&C</div>
                        </div>
                    </div>
                    <div class="example-prompt">
                        <div class="example-label">Try an example like this</div>
                        "I ordered a Samsung phone from Daraz for BDT 25,000. It arrived with a cracked screen and a missing charger. I contacted the seller 3 times over 2 weeks but got no response and no refund."
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # Render History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    if prompt := st.chat_input("Describe your complaint here..."):
        # Immediately clear the welcome screen since we now have a message
        welcome_container.empty()
        
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing case against CRPA 2009..."):
                try:
                    response = analyze_case(
                        description=prompt,
                        evidence_text=st.session_state.evidence_text,
                        tnc_text=st.session_state.tnc_pasted_text + "\n" + st.session_state.tnc_file_text
                    )
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"An error occurred during analysis: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# --- Main App Execution ---
def main():
    inject_custom_css()
    init_session_state()
    render_sidebar()
    render_chat_interface()

if __name__ == "__main__":
    main()
