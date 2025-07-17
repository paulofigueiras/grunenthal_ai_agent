from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import os

loader = PyPDFLoader("data/grunenthal_report_23_24.pdf")
docs = loader.load()

'''
This function performs RAG-based queries to the financial report PDF for Grünenthal for 2023. 

 Args:
    question (str): The question to be answered from the financial report.
 Returns:
    str: The answer to the question based on the financial report.
'''
def query_financial_report(question: str) -> str:
    
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, GoogleGenerativeAIEmbeddings(model="gemini-embedding-exp-03-07", google_api_key=os.getenv("GOOGLE_API_KEY")))

    llm = init_chat_model(
        "gemini-2.0-flash", 
        model_provider="google_genai",
        temperature=0.2,
        max_tokens=1024
    )
        
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=vectorstore.as_retriever(), return_source_documents=False)

    return qa.invoke({"query": question})