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
    st.markdown("""
        <style>
            /* Custom styling to make it feel more premium */
            .stChatInputContainer {
                padding-bottom: 2rem;
            }
            /* Change focus border of chat input from red to white */
            .stChatInputContainer:focus-within {
                border-color: white !important;
                box-shadow: 0 0 0 1px white !important;
            }
            /* Aggressively remove extra blank space at the top of the sidebar */
            [data-testid="stSidebarHeader"] {
                padding: 0 !important;
                height: 0 !important;
                min-height: 0 !important;
                overflow: visible !important;
                position: relative;
            }
            /* Fix the sidebar close button being cut off */
            [data-testid="stSidebarHeader"] button {
                position: absolute !important;
                top: 15px !important;
                right: 15px !important;
                z-index: 999 !important;
                transform: scale(0.9);
            }
            [data-testid="stSidebarUserContent"] {
                padding-top: 0 !important;
            }
            section[data-testid="stSidebar"] > div {
                padding-top: 0 !important;
            }
            .st-emotion-cache-1wmy9hl, .st-emotion-cache-6qob1r, .st-emotion-cache-12fmjuu {
                padding-top: 0 !important;
            }
            /* Empty state styling */
            .empty-state {
                text-align: center;
                padding: 4rem 2rem;
                color: #666;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 12px;
                margin-top: 2rem;
            }
            .empty-state h2 {
                font-weight: 600;
                margin-bottom: 1rem;
            }
        </style>
    """, unsafe_allow_html=True)

# --- State Initialization ---
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "evidence_text" not in st.session_state:
        st.session_state.evidence_text = ""
    if "tnc_text" not in st.session_state:
        st.session_state.tnc_text = ""
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

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
            new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
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
                            st.session_state.processed_files.add(f.name)
                        except Exception as e:
                            st.error(f"Failed to process {f.name}: {str(e)}")
                    status.update(label="Evidence processed successfully!", state="complete", expanded=False)
            
            # Show processed count
            if st.session_state.processed_files:
                st.success(f"{len(st.session_state.processed_files)} files loaded and ready.")

        st.divider()
        
        st.subheader("📜 Terms & Conditions")
        st.caption("Optional: Paste seller policies to check for breaches.")
        tnc_input = st.text_area(
            "Paste T&C text",
            height=150,
            placeholder="Paste Daraz, Shajgoj, or seller T&C here...",
            label_visibility="collapsed"
        )
        if tnc_input != st.session_state.tnc_text:
            st.session_state.tnc_text = tnc_input

        st.divider()
        st.warning("⚠️ **Disclaimer:** This tool provides legal information, not legal advice.")
        
        if st.button("🗑️ Clear Chat & Data", use_container_width=True):
            st.session_state.messages = []
            st.session_state.evidence_text = ""
            st.session_state.tnc_text = ""
            st.session_state.processed_files = set()
            st.rerun()

# --- Main Chat Interface ---
def render_chat_interface():
    # Use an empty container so we can dynamically clear it when the first message is sent
    welcome_container = st.empty()

    # Empty State
    if not st.session_state.messages:
        with welcome_container:
            st.markdown("""
                <style>
                    .welcome-card {
                        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                        border-radius: 20px;
                        padding: 3rem;
                        text-align: center;
                        margin-top: 2rem;
                        border: 1px solid rgba(255, 255, 255, 0.08);
                        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                    }
                    .welcome-badge {
                        display: inline-block;
                        background: rgba(99, 179, 237, 0.15);
                        border: 1px solid rgba(99, 179, 237, 0.4);
                        color: #63b3ed;
                        font-size: 0.78rem;
                        font-weight: 600;
                        letter-spacing: 0.1em;
                        text-transform: uppercase;
                        padding: 0.3rem 1rem;
                        border-radius: 999px;
                        margin-bottom: 1.5rem;
                    }
                    .welcome-title {
                        font-size: 2rem;
                        font-weight: 700;
                        color: #f0f4ff;
                        margin-bottom: 0.75rem;
                        line-height: 1.2;
                    }
                    .welcome-subtitle {
                        color: #a0aec0;
                        font-size: 1rem;
                        max-width: 480px;
                        margin: 0 auto 2.5rem auto;
                        line-height: 1.6;
                    }
                    .feature-grid {
                        display: grid;
                        grid-template-columns: repeat(3, 1fr);
                        gap: 1rem;
                        margin: 2rem 0;
                    }
                    .feature-tile {
                        background: rgba(255, 255, 255, 0.05);
                        border: 1px solid rgba(255, 255, 255, 0.08);
                        border-radius: 12px;
                        padding: 1.2rem 1rem;
                        transition: background 0.2s;
                    }
                    .feature-icon {
                        font-size: 1.8rem;
                        margin-bottom: 0.5rem;
                    }
                    .feature-label {
                        color: #e2e8f0;
                        font-size: 0.88rem;
                        font-weight: 500;
                        line-height: 1.4;
                    }
                    .example-prompt {
                        background: rgba(255, 255, 255, 0.07);
                        border-left: 3px solid #63b3ed;
                        border-radius: 8px;
                        padding: 1rem 1.25rem;
                        text-align: left;
                        margin-top: 2rem;
                        color: #cbd5e0;
                        font-size: 0.9rem;
                        font-style: italic;
                        line-height: 1.6;
                    }
                    .example-label {
                        color: #63b3ed;
                        font-size: 0.75rem;
                        font-weight: 700;
                        text-transform: uppercase;
                        letter-spacing: 0.08em;
                        margin-bottom: 0.4rem;
                        font-style: normal;
                    }
                </style>
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
                        tnc_text=st.session_state.tnc_text
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
