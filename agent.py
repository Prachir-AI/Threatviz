import requests
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from sanitization import sanitize_llm
from rag import retrieve_framework_context
from colorama import Fore, Style, init
import sys

init(autoreset=True)

def agent(llm):
    class CVEState(TypedDict, total=False):
        cve_id: str
        raw_cve_data: dict
        analysis: str
        threat_model: str
        critique_agent: str
        final_report: str


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
            raise ValueError(F'{Fore.RED}[X] The cve record for the cve id does not exist.')
            

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
    app = graph.compile()
    return app