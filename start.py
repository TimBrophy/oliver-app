import pandas as pd
import streamlit as st
import os
import openai
from elasticsearch import Elasticsearch
from langchain.embeddings import ElasticsearchEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
import tiktoken
import json
from datetime import datetime, timedelta
from dotenv import load_env
load_dotenv()

#------------------------------------------
#        connect to elasticsearch
#------------------------------------------

BASE_URL = os.environ['openai_api_base']
API_KEY = os.environ['openai_api_key']
DEPLOYMENT_NAME = "timb-fsi-demo"
chat_model = AzureChatOpenAI(
    openai_api_base=BASE_URL,
    openai_api_version=os.environ['openai_api_version'],
    deployment_name=DEPLOYMENT_NAME,
    openai_api_key=API_KEY,
    openai_api_type="azure",
    temperature=0.1
)

es = Elasticsearch(
    cloud_id=os.environ['elastic_cloud_id'],
    basic_auth=(os.environ['elastic_user'], os.environ['elastic_password'])
)

# Instantiate ElasticsearchEmbeddings using credentials
model_id = ".elser_model_1"
embeddings = ElasticsearchEmbeddings.from_es_connection(model_id, es)


def customer_support_search_operation(index, question):
    expansion_query = {
        "bool": {
            "should": [
                {
                    "text_expansion": {
                        "ml.inference.body_content_expanded.predicted_value": {
                            "model_id": model_id,
                            "model_text": question
                        }
                    }
                },
                {
                    "match": {
                        "body_content": question
                    }
                }
            ]
        }
    }
    field_list = ['title', 'body_content', '_score', 'url']
    results = es.search(index=index, query=expansion_query, size=20, fields=field_list)    
    response_data = [{"_score": hit["_score"], **hit["_source"]} for hit in results["hits"]["hits"]]
    documents = []
    # Check if there are hits
    if "hits" in results and "total" in results["hits"]:
        total_hits = results["hits"]["total"]

        # Check if there are any hits with a value greater than 0
        if isinstance(total_hits, dict) and "value" in total_hits and total_hits["value"] > 0:
            for hit in response_data:
                if hit['_score'] > 0:
                    doc_data = {field: hit[field] for field in field_list if field in hit}
                    documents.append(doc_data)
    return documents

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def truncate_text(text, max_tokens):
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return ' '.join(tokens[:max_tokens])

#------------------------------------------------
#        start with the form and control flow
#------------------------------------------------
index = 'search-elastic-docs'

st.title('Conversational search')

with st.form("search-form"):
    question = st.text_input("Go ahead and ask your question:", placeholder="Enter text here...")
    content = st.selectbox("Which website content do you want to use?", 
        ('Elastic Docs', 'VGI', 'TFG', 'Deimos'))    
    
    submitted = st.form_submit_button("Submit")

#-----------------------------------------------------------
#        Search for context and interact with the LLM
#-----------------------------------------------------------

    if submitted:
        if content == 'VGI':
            search_index = 'search-vitality'
        elif content == 'TFG':
            search_index = 'search-tfg'
        elif content == 'Deimos':
            search_index = 'search-homechoice'
        else: 
            search_index = 'search-elastic-docs'
        
        # st.write(st.session_state.assistant_type)
        st.session_state.question = question
        results = customer_support_search_operation(search_index, question)
        string_results = json.dumps(results)
        column_list = ["title", "_score", "url"]
        df_results = pd.DataFrame(results, columns=column_list)
        string_results = truncate_text(string_results, 8000)
        # interact with the LLM
        augmented_prompt = f"""Using only the context below, answer the query.
        Context: {string_results}

        Query: {question}"""

        messages = [
            SystemMessage(content="You are a helpful customer support agent that answers questions based only on the context provided. When you respond, please cite your source."),
            # HumanMessage(content="Hi AI, how are you today?"),
            # AIMessage(content="I am very good. How may I help you?"),
            HumanMessage(content=augmented_prompt)
            ]                 
        st.subheader('Virtual assistant:')
        chat_bot = st.chat_message("ai assistant", avatar="🤖")
        # st.write(num_tokens_from_string(string_results, "cl100k_base"))
        with st.status("Contacting Azure OpenAI...") as status:
            chat_bot.info(chat_model(messages).content)
            status.update(label="AI response complete!", state="complete")

        # handle any context data that we want to represent
        st.dataframe(df_results)