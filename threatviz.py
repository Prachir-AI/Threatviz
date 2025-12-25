import argparse
import json
import os
import sys
import logging
import warnings
os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
os.environ['REQUESTS_CA_BUNDLE'] = ''
from local_dashboard import start
from llm import load_llm
from agent import agent
from report import render_security_report_html

warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

from colorama import Fore, Style, init
init(autoreset=True)



def main(llm, cve):
    if not cve:
        print(Fore.RED + "❌ No CVE ID provided" + Style.RESET_ALL)
        return None
    llm = load_llm(llm)
    app = agent(llm)
    result = app.invoke({"cve_id": cve})
    return result["final_report"]




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-id", help="CVE ID to analyze")
    parser.add_argument("-model", choices=["groq", "openai", "claude", "mistral", "gemini"], default="groq", help="LLM provider (default: groq)")
    parser.add_argument("-html_report", action="store_true")
    parser.add_argument("-json_report", action="store_true")
    parser.add_argument("-dashboard", action="store_true", help="Launch web dashboard with CVE search")
    parser.add_argument("-dashboard_inner", action="store_true", help="Internal flag for launching dashboard")
    args = parser.parse_args()
    return parser, args



if __name__ =='__main__':
    _, args = parse_args()
    if args.dashboard_inner:
        print(f"{Fore.YELLOW}{'='*50}")
        print(f"{Fore.YELLOW}  Threatviz Dashboard Mode Started")
        print(f"{Fore.YELLOW}{'='*50}{Style.RESET_ALL}\n")
        start()
    elif args.dashboard_inner or args.dashboard:
        import subprocess
        script_path = os.path.abspath(__file__)
        print(Fore.CYAN + "[*] Launching Streamlit Dashboard..." + Style.RESET_ALL)
        subprocess.run([sys.executable, "-m", "streamlit", "run", script_path, "--", "-dashboard_inner"])
    elif not args.dashboard and not args.dashboard_inner:
        if args.id:
            print(f"{Fore.YELLOW}{'='*50}")
            print(f"{Fore.YELLOW}  Threatviz CLI Mode: {args.id}")
            print(f"{Fore.YELLOW}{'='*50}{Style.RESET_ALL}\n")
            output = main(args.model, args.id)
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
            args.print_help()