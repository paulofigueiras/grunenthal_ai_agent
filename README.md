# grunenthal_ai_agent: An AI Agent for Grünenthal
An AI Agent performing RAG over the FDA database of adverse events, Neo4j HealthcareAnalytics database and Grünenthal's 2023 financial report

## Project Structure:
    ├── app.py                          ← Streamlit chatbot UI
    ├── agent.py                        ← LangChain Agent setup
    ├── tools/
    │   ├── neo4j_rag .py               ← Tool: Neo4j graph Cypher query creation through free text RAG and graph query execution 
    │   ├── fda_api.py                  ← Tool: FDA Adverse Events API connection and retrieval
    │   └── financial_report_rag.py     ← Tool: Grünenthal's financial report PDF RAG-based information retrieval
    ├── data/
    │   └── financial_report.pdf        ← Financial report document
    ├── .env                            ← Config and environmental variables (not available in the repo)
    ├── requirements.txt                ← Python library requirements
    └── test_prompts.txt                ← Several prompts used for testing the AI Agent
