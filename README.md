# `Threatviz👁️` - Turn any CVE into AI Powered Threat modeling 
![alt text](asset/logo.png)
* Agentic AI based Threat Modeling Technique
* Supports  `STRIDE`
* Supports `PASTA`
### Just Insert any CVE then it will turn that to full end to end threat modeling workflow.


---
### Visualization
![alt text](asset/image.png)
----
![alt text](asset/tm2.png)
----
![alt text](asset/tm3.png)
----
![alt text](asset/tm4.png)
![alt text](asset/tm5.png)

----

### Tech Stack

- `Langgraph` for multi-agent orchestration
- `Faiss` for the knowledge base and  RAG
- `Groq` and `OpenAI` for model endpoint
- `Huggingface` for embeddings
- `Streamlit` for Web Interface
---

### Below AI Models are spported
| Model           | Status | 
| :---------------- | :------: |
| `qwen/qwen3-32b`      |   ✔️   |
| `meta-llama/llama-4-maverick-17b-128e-instruct`          |   ✔️   |
| `openai/gpt-oss-120b`   |  ✔️   | 
| `openai/gpt-oss-20b` |  ✔️   |
| `openai/gpt-4o-mini` |  ✔️   |
| `moonshotai/kimi-k2-instruct-0905`|  ✔️   |

---
### Deployment
- Local Deployment with light weight GUI.
- Deployed and tested using `Amazon Bedrock AgentCore Runtime` and `AgentCore Memory` services
- `AWS CloudWatch` for observability

---
### Usage
- Please check `.env.example` file to setup API key.
- Once setup done, change it from `.env.example` to `.env`
```
git clone https://github.com/findthead/Threatviz.git
cd Threatviz
uv sync
uv run threatviz.py -id CVE-2025-55182 -html_report 
```

### Host it locally with below command
```
uv run threatviz.py -dashbord
```
### GUI for Local Deployment
![Web Interface](asset/web.png)
----
![Analysis](asset/web2.png)

---
### Cloudwatch Observability (optional)
![alt text](asset/cw1.png)
![alt text](asset/cw2.png)
![alt text](asset/cw3.png)

---
### `Please provide the citation.`
```
@software{Threatviz,
  author = {Subhay},
  title = {Threatviz: An autonomous multi-agent Threat Modeling Tool},
  url = {https://github.com/findthehead/Threatviz},
  version = {0.0.3},
  year = {2025}
}
```
---
###### This project is built for the purpose of perticipatating at agentic AI coding and deployment hackathon sponsored by @https://github.com/codebasics
