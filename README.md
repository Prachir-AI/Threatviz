# Threatviz - Turn any CVE into AI Powered Threat modeling 
![alt text](asset/logo.png)
* Agentic AI based Threat Modeling Technique
* Supports  STRIDE
* Supports PASTA
#### Just put any CVE id in the search box it will turn that to full end to end threat modeling  diagram.


---
### Threatmodeling with Visualization
![alt text](asset/image.png)
----
![alt text](asset/tm2.png)

----

### AI Tech Stack

- Langgraph for multi-agent orchestration
- Faiss for the knowledge base and  RAG
- Groq, OpenAI, Claude and Gemini for model endpoint
- Huggingface for embedding
- Streamlit for Web Interface

### Deployment
- Local Deployment with light weight GUI.
- Deployed and tested using Amazon Bedrock AgentCore and AgentCore memory services
- AWS CloudWatch for observability

### Usage
```
git clone https://github.com/findthead/Threatviz.git
cd Threatviz
uv sync
source .venv/bin/activate   
uv run threatviz.py -id CVE-2025-55182 -html_report 
```

### Deployment
* Host it locally with below command
```
uv run threatviz.py -dashbord
```
### GUI
![Web Interface](asset/web.png)
----
![Analysis](asset/web2.png)
----
`Note: Working Perfectly with moonshotai/kimi-k2-instruct-0905 from Groq`
---
### Please provide the citation.
```
@software{Threatviz,
  author = {Subhay},
  title = {Agent-Flow: An autonomous multi-agent Threat Modeling framework},
  url = {https://github.com/findthehead/Threatviz},
  version = {0.0.1},
  year = {2026}
}
```

###### This project is built for the purpose of perticipatating at agentic AI coding hackathon sponsored by @codebasics https://github.com/codebasics
---