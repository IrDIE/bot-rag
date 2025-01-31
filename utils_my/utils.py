import re, os, bs4
import json
from bs4 import BeautifulSoup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from typing import List

import pickle
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain import hub

from langchain_community.document_loaders import WebBaseLoader
from langchain import hub
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_openai import ChatOpenAI


from langchain_pinecone import PineconeEmbeddings
from pydantic import HttpUrl


def contains_answer_variants(query : str) -> int:
    # Regular expression to detect the pattern "1. ... 2. ... 3. ..." with variable number of answers
    pattern = r"(?:\d+\.\s+.+\n?){2,}"
    
    return True if re.search(pattern, query, re.DOTALL) else False

def test_vpn():
    import requests
    url = 'https://platform.openai.com/'
    myobj = {'somekey': 'somevalue'}
    x = requests.post(url, json = myobj)
    print(x.text)

def magic_prediction(user_query : str ) :
    return run(user_query)

def parce_recurcuve(urls : List[str], save = False):
    results = []

    # Define a custom extractor function to extract text from HTML using BeautifulSoup
    def custom_extractor(html_content):
        soup = BeautifulSoup(html_content, "html.parser")
        return soup.get_text()

    # Instantiate the RecursiveUrlLoader
    for url in urls:
        loader = RecursiveUrlLoader(
            url=url, 
            extractor=custom_extractor, 
            max_depth=5, 
            timeout=10
            )

        # Load the data from the website
        docs = loader.load()
        results.extend(docs)
    return results

    # # Define the file path to store the data
    # output_file = "scraped__data.txt"

    # # Open the file in write mode with UTF-8 encoding
    # with open(output_file, "w", encoding="utf-8") as file:
    #     # Write metadata and content for each document to the file
    #     for doc in docs:
    #         title = doc.metadata.get("title")
    #         source = doc.metadata.get("source")
    #         content = doc.page_content

    #         # Ensure that the title, source, and content are string type
    #         if isinstance(title, str) and isinstance(source, str) and isinstance(content, str):
    #             file.write("Page Title: " + title + "\n")
    #             file.write("Page URL: " + source + "\n")
    #             file.write("Page Content:\n" + content + "\n\n")
    #         else:
    #             print("Skipped a document due to non-string content.")

def parce_recurcuve_depth(urls : List[str], save = False, max_depth = 2):
    results = []

    # Define a custom extractor function to extract text from HTML using BeautifulSoup
    def custom_extractor(html_content):
        soup = BeautifulSoup(html_content, "html.parser")
        return soup.get_text()

    # Instantiate the RecursiveUrlLoader
    for url in urls:
        loader = RecursiveUrlLoader(
            url=url, 
            extractor=custom_extractor, 
            max_depth=max_depth, 
            timeout=10
            )

        # Load the data from the website
        docs = loader.load()
        results.extend(docs)
    return results

def dump_documents(docs, name_f):
    with open(f'{name_f}.pickle', 'wb') as handle:
        pickle.dump(docs, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_documents(paths : list):
    with open('docs.pickle', 'rb') as f:
        docs = pickle.load(f)

def load_documents_langchain_community(urls):
    loader = WebBaseLoader(
    web_paths=urls,
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    return loader.load()

def load_indexed_db(embeddings, path:str = "utils_my/faiss_index_all_summary"):
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def preprocess_user_q(q : str):
    template = "\nВыведи ответ в таком формате:\n1. {правильныя цифра ответа}\n2. {развернутое и подробное объяснение почему этот ответ правильный}\n"
    q = "Вот вопрос и варианты ответов:\n" + q
    q += template
    return q

def preprocess_user_q_no_answers(q : str):
    q = "Вот вопрос:\n" + q
    return q

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def parce_links(source : str):
    pattern = '(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-&?=%.]+'
    res = re.findall(pattern, source)
    return res if res else None

def process_answer_model(answer : str, sources : list):
    parts = answer.split('\n')
    id_answer = int(parts[0].split('1. ')[1])
    reasoning = parts[1].split('2. ')[1]
    links = list(set([doc.metadata['source'] for doc in sources]) )[:3]
    links_ = [HttpUrl(l) for l in links]

    return id_answer, reasoning, links_

def process_answer_model_no_answers(answer : str, sources : list):
    reasoning = answer
    id_answer = None
    links = list(set([doc.metadata['source'] for doc in sources]) )[:3]

    return id_answer, reasoning, links


def run(user_q):
    have_variants = contains_answer_variants(user_q)
    print(f"got \n{user_q}")

    load_dotenv() 
    langchain_tracing_v2 = os.getenv('LANGCHAIN_TRACING_V2')
    langchain_endpoint = os.getenv('LANGCHAIN_ENDPOINT')
    langchain_api_key = os.getenv('LANGCHAIN_API_KEY')

    ## LLM
    openai_api_key = os.getenv('OPENAI_API_KEY')

    ## Pinecone Vector Database
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pinecone_api_host = os.getenv('PINECONE_API_HOST')
    index_name = os.getenv('PINECONE_INDEX_NAME')

    os.environ['LANGCHAIN_TRACING_V2'] = langchain_tracing_v2
    os.environ['LANGCHAIN_ENDPOINT'] = langchain_endpoint
    os.environ['LANGCHAIN_API_KEY'] = langchain_api_key
    os.environ['OPENAI_API_KEY'] = openai_api_key

    #Pinecone keys
    os.environ['PINECONE_API_KEY'] = pinecone_api_key
    os.environ['PINECONE_API_HOST'] = pinecone_api_host
    os.environ['PINECONE_INDEX_NAME'] = index_name

    model_name = "multilingual-e5-large" 
    

    embeddings = PineconeEmbeddings(
        model=model_name,
        pinecone_api_key=pinecone_api_key

    )
    docsearch = load_indexed_db(embeddings)
    # retriever = docsearch.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

    def retrieve(state: State):
        retrieved_docs = docsearch.similarity_search(state["question"])
        return {"context": retrieved_docs}
    
    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}
    

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    test_q = preprocess_user_q(user_q) if have_variants else preprocess_user_q_no_answers(user_q)

    result = graph.invoke({"question": test_q})
    if not have_variants:
        id_answer, reasoning, links_raw_str = process_answer_model_no_answers(result["answer"], result['context'])
    else:
        id_answer, reasoning, links_raw_str = process_answer_model(result["answer"], result['context'])

    return id_answer, reasoning, links_raw_str 


    



    




