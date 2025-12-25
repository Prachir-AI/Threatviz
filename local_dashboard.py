import streamlit as st
import json
import streamlit.components.v1 as components
from report import render_security_report_html
from sanitization import sanitize_cve_id

def start():
    st.set_page_config(
        page_title="ThreatViz CVE Dashboard",
        page_icon=":shield:",
        layout="wide"
    )
    _, center_col, _ = st.columns([1, 2, 1])

    with center_col:
        st.markdown(
            """
            <div style="padding:20px; text-align:center;">
                <h1 style="color:red; font-size:90px; margin:0;">ThreatViz</h1>
                <p style="color:white; font-size:18px;">
                    CVE Threat Analysis & Visualization Dashboard
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        cve_input_raw = st.text_input(
            "",
            placeholder="e.g. CVE-2021-44228"
        )
    if cve_input_raw:
        try:
            cve_input = sanitize_cve_id(cve_input_raw)

            with st.spinner(f"🔍 Analyzing {cve_input}..."):
                from threatviz import parse_args, main
                _, args = parse_args()
                rep = main(args.model, cve_input)
            action_col1, action_col2 = st.columns([1, 6])
            final_report = rep
            if isinstance(final_report, str):
                final_report = json.loads(final_report)

            with action_col1:
                st.download_button(
                    label="📥 Download JSON",
                    data=json.dumps(final_report, indent=2),
                    file_name=f"{cve_input}.json",
                    mime="application/json"
                )

            html_content = render_security_report_html(final_report)

            components.html(
                html_content,
                height=1400,
                scrolling=True
            )

        except ValueError as ve:
            st.warning(str(ve))

        except Exception as e:
            st.error(f"❌ Error rendering report: {e}")
