from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from dotenv import load_dotenv
import argparse
import json
import os
import sys
import uuid
import logging
import warnings
import requests
os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
os.environ['REQUESTS_CA_BUNDLE'] = ''

warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

from colorama import Fore, Style, init
init(autoreset=True)

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from langgraph_checkpoint_aws import AgentCoreMemoryStore, AgentCoreMemorySaver

agentcore_app = BedrockAgentCoreApp()

from helper import (
    retrieve_framework_context,
    sanitize_llm,
    render_security_report_html,
    sanitize_cve_id
)

_ = load_dotenv()


parser = argparse.ArgumentParser()
parser.add_argument("-id", help="CVE ID to analyze")
parser.add_argument(
    "-model",
    choices=["groq", "openai", "claude", "gemini"],
    default="groq",
    help="LLM provider (default: groq)"
)
parser.add_argument("-store", action="store_true", help="Persist CVE report in memory")
parser.add_argument("-deploy", action="store_true", help="Deploy using AWS AgentCore")
parser.add_argument("-html_report", action="store_true")
parser.add_argument("-json_report", action="store_true")
parser.add_argument("-dashboard", action="store_true", help="Launch web dashboard with CVE search")
parser.add_argument("-dashboard_inner", action="store_true", help="Internal flag for launching dashboard")
args = parser.parse_args()

# =========================
# Model Loader (FAIL FAST)
# =========================

def load_llm(provider: str):
    if provider == "groq":
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("❌ GROQ_API_KEY not found in .env")
        return ChatGroq(model="moonshotai/kimi-k2-instruct-0905", temperature=0)

    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("❌ OPENAI_API_KEY not found in .env")
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)

    if provider == "claude":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("❌ ANTHROPIC_API_KEY not found in .env")
        return ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

    if provider == "gemini":
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("❌ GOOGLE_API_KEY not found in .env")
        return ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

    raise ValueError("Unsupported model provider")


llm = load_llm(args.model)

# =========================
# CVE State
# =========================

class CVEState(TypedDict, total=False):
    cve_id: str
    raw_cve_data: dict
    analysis: str
    threat_model: str
    critique_agent: str
    final_report: str

# =========================
# Graph Nodes
# =========================

def fetch_cve(state: CVEState):
    print(f"{Fore.CYAN}[*] Fetching CVE data for {state['cve_id']}...{Style.RESET_ALL}")
    import unicodedata
    import re
    cve = state["cve_id"]
    cve = unicodedata.normalize("NFKC", cve)
    cve = re.sub(r'[–—‑\u2010\u2011\u2012\u2013\u2014]', '-', cve)  # replace all dash variants with '-'
    cve = re.sub(r'[^\w\-]', '', cve)  # remove anything that's not alphanumeric or '-'
    cve = cve.upper()
    state["cve_id"] = cve

    url = f"https://cveawg.mitre.org/api/cve/{cve}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers, timeout=10)
    if response.status_code == 200:
        state["raw_cve_data"] = response.json()
        print(f"{Fore.GREEN}[+] CVE data fetched successfully{Style.RESET_ALL}")
        return state
    else:
        return None

def analyze_cve_with_rag(state: CVEState) -> CVEState:
    print(f"{Fore.CYAN}[*] Analyzing CVE with PASTA/STRIDE framework...{Style.RESET_ALL}")
    SYSTEM_PROMPT = """
You are a security analysis agent.

Rules:
- By reading the json date regarding CVE, you have to write a effective analysys with CVSS score, key findings, mitigation etc.
- You MUST ground all threat modeling decisions in the provided context.
- Do NOT invent new framework steps.
"""
    context = retrieve_framework_context(
        query="PASTA stages attack surface risk analysis STRIDE mapping"
    )
    prompt = f"""

CVE DATA:
{state['raw_cve_data']}

{SYSTEM_PROMPT}
Introduce some PASTA and STRIDE threat model with the refrence context below\n
REFERENCE CONTEXT:
{context}



Task:
Provide a detailed paragraph of the anlysys. Dont add any bullet points or any style content.
"""
    response = llm.invoke(prompt)
    
    state["analysis"] = sanitize_llm(response.content)
    print(f"{Fore.GREEN}[+] CVE analysis completed{Style.RESET_ALL}")
    return state



def generate_threat_model_rag(state: CVEState) -> CVEState:
    print(f"{Fore.CYAN}[*] Generating threat model{Style.RESET_ALL}")
    context = retrieve_framework_context(
        query="""Syntax reference / basics: Mermaid Syntax Reference\nSequence diagrams: Sequence Diagram\nClass diagrams: Class Diagram\nState diagrams: State Diagram\nEntity-relationship diagrams: ER Diagram\nPie charts: Pie Diagram\nC4 / Architecture diagrams: C4 Diagram\nArchitecture diagrams: Architecture"""
    )
    SYSTEM_PROMPT = """
You are a Threat Modeling expert who creates valid Mermaid diagrams.
You know multiple Mermaid diagram types and always use multiple design types and patterns for visualization friendly threat modeling. Always focus on STRIDE and PASTA threat modeling framework while generating the visualization.
Your output should be mermaid synatx only without any extra info"""


    prompt = f"""
    {SYSTEM_PROMPT}

    Mermaid Documentation: 

    Use this docmentation below for effective writing\n.
    {context}

    Use this below report to reconsile the exact info needed for threatmodeling:\n
    {state['analysis']}"""


    response = llm.invoke(prompt)
    mermaid_code = sanitize_llm(response.content)
    state["threat_model"] = mermaid_code
    print(f"{Fore.GREEN}[+] Threat modeling Completed{Style.RESET_ALL}")
    return state

def mermaid_syntax_critique(state: CVEState) -> CVEState:
    print(f"{Fore.CYAN}[*] Critiquing report for Mermaid syntax fixes...{Style.RESET_ALL}")
    SYSTEM_PROMPT = """You are a skilled mermaid syntax analyzer agent, your task is to perform quality check in provided mermaid syntax and output the fixed one. Follow the below rule.
STRICT RULES:
1. DO NOT use the keyword `title` anywhere in the diagram.
2. DO NOT use angle brackets `< >` in labels, node text, or subgraph names.
3. Avoid special characters that can break parsing (use words instead of symbols).
4. Wrap all human-readable labels in double quotes.
5. Ensure all nodes are defined before being styled.
6. Dont use any non-mermaid syntax after .mermaid block for example-
mermaid
%% Diagram: External attacker to database execution flow.
7. Dont use any comments with ""%%"". Only render mermaid syntax not required anything extra

ALLOWED:
- subgraph blocks
- arrows (`-->`, `-- text -->`)
- styles (`style Node fill:#color`)

OUTPUT FORMAT:
- Output ONLY valid Mermaid code
- Do NOT include explanations or markdown
- The diagram must render in Mermaid v10+
"""
    prompt = f"""
{SYSTEM_PROMPT}

See the below code and fix the mermaid syntax.\n
{state['threat_model']}
"""
    response = llm.invoke(prompt)
    state["critique_agent"] = sanitize_llm(response.content)
    print(f"{Fore.GREEN}[+] Report critique completed{Style.RESET_ALL}")
    return state


def generate_report(state: CVEState) -> CVEState:
    SYSTEM_PROMPT = """You are a CVE Threat Intelligence report writer. 
Your task is to analyze raw reports created by juniors and produce a comprehensive, professional report.

Rules:  
1. Your output must be a valid JSON object with the following keys:
{
 "TITLE": "A concise title containing the CVE ID and CVSS score",
 "EXECUTIVE_SUMMARY": "A short introduction of the CVE with CVSS score and other high-level info. Use multiple points separated by a newline, starting with '1.', '2.', etc.",
 "DETAILED_ANALYSIS": "A detailed technical analysis of the CVE. Use multiple points separated by newline characters, starting with '1.', '2.', '3.', etc.",
 "RISK_ASSESSMENT": "Risk assessment including CVSS vector analysis from an attacker’s perspective. Include STRIDE and PASTA threat modeling separately, using newline characters for each point.",
 "THREAT_MODEL": "Mermaid syntax for diagrams representing the threat model. Include sequenceDiagram, graph, stateDiagram, or other supported mermaid types as needed.",
 "MITIGATION": "Exact mitigation steps or recommendations provided in the junior’s report. Use multiple points separated by newline characters starting with '1.', '2.', etc."
}

2. Each field must be either a **string** (with multiple lines separated by newline characters) or a **list of strings**. Do not serialize the JSON object itself inside any field.  
3. Preserve all factual information from the junior's report, but enhance clarity and professionalism.  
4. Do not add any extra information that is not present in the junior's report.  
5. Your output must be a **JSON object only** with no extra text before or after.

Example:

{
 "TITLE": "CVE-2025-55182: Critical Pre-Authentication RCE in React Server Components (CVSS Score: 10)",
 "EXECUTIVE_SUMMARY": "1. CVE-2025-55182 is a critical pre-authentication remote code execution vulnerability affecting React Server Components.\n2. It impacts react-server-dom-parcel, react-server-dom-turbopack, and react-server-dom-webpack versions 19.0.0 to 19.2.0.\n3. The vulnerability has a CVSS score of 10, indicating critical severity.",
 "DETAILED_ANALYSIS": "1. The vulnerability allows an attacker to execute arbitrary code on the server.\n2. Affected packages and versions:\n   1. react-server-dom-parcel: 19.0.0, 19.1.0, 19.1.1, 19.2.0\n   2. react-server-dom-turbopack: 19.0.0, 19.1.0, 19.1.1, 19.2.0\n   3. react-server-dom-webpack: 19.0.0, 19.1.0, 19.1.1, 19.2.0\n3. Exploitation may lead to full system compromise.",
 "RISK_ASSESSMENT": "1. CVSS Score: 10 (Critical)\n2. STRIDE Threats:\n   1. Elevation of Privilege: allows arbitrary code execution.\n   2. Denial of Service: high likelihood of disruption.\n3. PASTA Threat Modeling:\n   1. Business Objectives: Protect React Server Components.\n   2. Technical Scope: Affected packages and versions.",
 "THREAT_MODEL": "graph LR\nparticipant ThreatModeler as 'Threat Modeler'\nparticipant System as 'System'\nThreatModeler-->System: Identify Assets\nSystem-->ThreatModeler: Provide Components",
 "MITIGATION": "1. Upgrade to version 19.2.1 or later.\n2. Implement input validation and secure deserialization.\n3. Apply secure coding practices."
}
"""


    prompt = f"""
{SYSTEM_PROMPT}

Juniors Report:
{state['analysis']}

Mermaid syntax:
{state['critique_agent']}
"""
    response = llm.invoke(prompt)
    state["final_report"] = sanitize_llm(response.content)
    return state


# =========================
# Build Graph
# =========================
graph = StateGraph(CVEState)
graph.add_node("fetcher", fetch_cve)
graph.add_node("analyzer", analyze_cve_with_rag)
graph.add_node("threat_modeler", generate_threat_model_rag)
graph.add_node("critique_agent", mermaid_syntax_critique)
graph.add_node("reporter", generate_report)
graph.set_entry_point("fetcher")
graph.add_edge("fetcher", "analyzer")
graph.add_edge("analyzer", "threat_modeler")
graph.add_edge("threat_modeler", "critique_agent")
graph.add_edge("critique_agent", "reporter")
graph.set_finish_point("reporter")
# =========================
# Initialize Memory (if needed)
# =========================
store = None
checkpointer = None
MEMORY_ID = "threatviz-cve-memory"


NAMESPACE = ("threatviz", "cve-reports")

def load_from_memory(cve_id):
    if not store:
        return None
    item = store.get(NAMESPACE, cve_id)
    return item.value if item else None


def save_to_memory(cve_id, report):
    if store:
        store.put(
            NAMESPACE,
            cve_id,
            {"report": report}
        )

# Setup memory if needed
if args.store:
    try:
        store = AgentCoreMemoryStore(memory_id=MEMORY_ID)
        checkpointer = AgentCoreMemorySaver(memory_id=MEMORY_ID)
        print(Fore.CYAN + "[*] Memory system initialized" + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"❌ Failed to initialize memory: {e}" + Style.RESET_ALL)

# Now compile the graph safely
app = graph.compile(checkpointer=checkpointer)

# =========================
# Execution Logic
# =========================

def run(cveid):
    if not cveid:
        return json.dumps({"error": "No CVE ID provided"})
    
    # 1. Check memory first
    cached = load_from_memory(cveid)
    if cached:
        print(Fore.GREEN + "[+] Loaded report from memory" + Style.RESET_ALL)
        return cached["report"]

    # 2. Execute graph to generate report
    result = app.invoke({"cve_id": cveid})

    # 3. Save report to memory if required
    if args.store and args.deploy:
        save_to_memory(cveid, result["final_report"])
        print(Fore.GREEN + "[+] Report saved to memory" + Style.RESET_ALL)

    return result["final_report"]
#======deploy=======
import shutil
import subprocess
from colorama import Fore, Style
import sys

def deploy_to_bedrock():
    """
    Deploys the application to AWS Bedrock AgentCore.
    Checks for the 'agentcore' CLI and executes deployment with persistent memory.
    """
    print(Fore.CYAN + "[*] Preparing deployment to AWS Bedrock AgentCore..." + Style.RESET_ALL)
    
    # 1. Find AgentCore CLI
    cli_path = shutil.which("agentcore")
    if not cli_path:
        print(Fore.RED + "❌ Error: 'agentcore' CLI not found in system PATH." + Style.RESET_ALL)
        print(Fore.YELLOW + "Please install it using: pip install bedrock-agentcore" + Style.RESET_ALL)
        sys.exit(1)
        
    print(Fore.GREEN + f"[+] Found AgentCore CLI at: {cli_path}" + Style.RESET_ALL)

    # 2. Construct Deployment Command
    # We include the memory ID to ensure persistent storage is linked
    deploy_command = [
        "agentcore", "deploy",
        "--app", os.path.abspath(__file__),
        "--memory-id", MEMORY_ID,
        "--name", "ThreatViz-CVE-Analyzer",
        "--public"  # Optional: makes the endpoint accessible
    ]

    try:
        print(Fore.CYAN + "[*] Executing deployment command..." + Style.RESET_ALL)
        # Using check=True to raise an error if the command fails
        result = subprocess.run(deploy_command, check=True, capture_output=True, text=True)
        
        print(Fore.GREEN + "✅ Deployment Successful!" + Style.RESET_ALL)
        print(Fore.WHITE + result.stdout + Style.RESET_ALL)
        
    except subprocess.CalledProcessError as e:
        print(Fore.RED + "❌ Deployment failed!" + Style.RESET_ALL)
        print(Fore.RED + f"Error Details: {e.stderr}" + Style.RESET_ALL)
        sys.exit(1)

# =========================
# Dashboard with Streamlit
# =========================

if hasattr(args, "dashboard_inner") or "-dashboard_inner" in sys.argv:
    import streamlit as st
    import json
    import streamlit.components.v1 as components

    st.set_page_config(
        page_title="ThreatViz CVE Dashboard",
        page_icon=":shield:",
        layout="wide"
    )

    # =========================
    # HERO SECTION (CENTERED)
    # =========================
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

    # =========================
    # ANALYSIS (FULL WIDTH)
    # =========================
    if cve_input_raw:
        try:
            cve_input = sanitize_cve_id(cve_input_raw)

            with st.spinner(f"🔍 Analyzing {cve_input}..."):
                report_data = run(cve_input)

            report_json = json.loads(report_data)

            # Action bar (full width)
            action_col1, action_col2 = st.columns([1, 6])

            with action_col1:
                st.download_button(
                    label="📥 Download JSON",
                    data=report_data,
                    file_name=f"{cve_input}.json",
                    mime="application/json"
                )

            # Full-width HTML report
            html_content = render_security_report_html(report_json)
            components.html(
                html_content,
                height=1400,
                scrolling=True
            )

        except ValueError as ve:
            st.warning(str(ve))

        except Exception as e:
            st.error(f"❌ Error rendering report: {e}")

#----------------
#agentcore
#------------------
@agentcore_app.entrypoint
def agentcore_handler(payload, context):
    """
    AWS AgentCore entrypoint
    """
    print("Payload:", payload)
    print("Context:", context)

    # Expect CVE ID from payload
    cve_id = payload.get("cve_id") or payload.get("prompt")

    if not cve_id:
        return {"error": "No CVE ID provided"}

    # Invoke LangGraph app
    result = app.invoke(
        {"cve_id": cve_id},
        config={
            "configurable": {
                "thread_id": context.get("session_id", "default-session"),
                "actor_id": context.get("actor_id", "default-user")
            }
        }
    )

    return {
        "cve_id": cve_id,
        "report": result["final_report"]
    }

def save_security_report(result, cve_id, html_report=True, json_report=True):
    """
    Saves security reports in HTML, JSON, or both.
    Automatically handles invalid JSON in result['final_report'].
    """
    report_data = result.get('final_report', "")

    report_dict = None
    if isinstance(report_data, str):
        try:
            report_dict = json.loads(report_data)
        except json.JSONDecodeError:
            report_dict = {
                "TITLE": "Security Report",
                "EXECUTIVE_SUMMARY": [report_data],
                "DETAILED_ANALYSIS": [],
                "RISK_ASSESSMENT": "",
                "THREAT_MODEL": "",
                "MITIGATION": ""
            }
    elif isinstance(report_data, dict):
        report_dict = report_data
    else:
        report_dict = {
            "TITLE": "Security Report",
            "EXECUTIVE_SUMMARY": [str(report_data)],
            "DETAILED_ANALYSIS": [],
            "RISK_ASSESSMENT": "",
            "THREAT_MODEL": "",
            "MITIGATION": ""
        }
    return report_dict
if __name__ == "__main__":
    if args.deploy:
        deploy_to_bedrock()
        sys.exit(0)


    if args.dashboard:
        import subprocess
        script_path = os.path.abspath(__file__)
        print(Fore.CYAN + "[*] Launching Streamlit Dashboard..." + Style.RESET_ALL)
        subprocess.run([sys.executable, "-m", "streamlit", "run", script_path, "--", "-dashboard_inner"])

    elif not args.dashboard and not args.dashboard_inner:
        if args.id:
            print(f"{Fore.YELLOW}{'='*50}")
            print(f"{Fore.YELLOW}  Threatviz CLI Mode: {args.id}")
            print(f"{Fore.YELLOW}{'='*50}{Style.RESET_ALL}\n")
            output = run(args.id)
            if args.html_report:
                html = render_security_report_html(json.loads(output))
                with open(f"{args.id}.html", "w", encoding="utf-8") as f:
                    f.write(html)
                print(Fore.GREEN + f"[+] HTML Report: {args.id}.html" + Style.RESET_ALL)

            if args.json_report:
                with open(f"{args.id}.json", "w", encoding="utf-8") as f:
                    f.write(output)
                print(Fore.GREEN + f"[+] JSON Report: {args.id}.json" + Style.RESET_ALL)

            if not args.html_report and not args.json_report:
                print(json.loads(output))
        else:
            print(Fore.RED + "❌ Error: Please provide a CVE ID (-id) or use the dashboard (-dashboard) flag." + Style.RESET_ALL)
            parser.print_help()