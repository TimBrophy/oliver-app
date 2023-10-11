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
from PIL import Image
from elastic_enterprise_search import AppSearch

os.environ['openai_api_base']=st.secrets['openai_api_base']
os.environ['openai_api_key']=st.secrets['openai_api_key']
os.environ['openai_api_version']=st.secrets['openai_api_version']
os.environ['elastic_cloud_id']=st.secrets['cloud_id']
os.environ['elastic_user']=st.secrets['user']
os.environ['elastic_password']=st.secrets['password']
master_index = 'search-content-master'

# connect to all services

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
app_search = AppSearch(os.environ['app_search_url'],
                       basic_auth=(os.environ['elastic_user'], os.environ['elastic_password']))

# Instantiate ElasticsearchEmbeddings using credentials
model_id = ".elser_model_1"
embeddings = ElasticsearchEmbeddings.from_es_connection(model_id, es)


def get_content_sources():
    documents = []
    field_list = ['name', 'url', 'domain_id']
    results = es.search(index='content-sources', size=100, fields=field_list)
    if results['hits']['total']['value'] > 0:
        for hit in results['hits']['hits']:
            doc_data = {
                'name' : hit['_source']['name'],
                'url' : hit['_source']['url']
            }

            documents.append(doc_data)
    return documents

def search_for_answers(index, question):
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
    resp = app_search.search(
        engine_name=index,
        body=expansion_query
    )

image = Image.open('images/logo.png')

st.image(image, width=200)

st.title('Conversational search')

with st.form("search-form"):
    sources = get_content_sources()
    options = []
    for s in sources:
        content_source = s['name']
        options.append(content_source)
    question = st.text_input("Go ahead and ask your question:", placeholder="Enter text here...")
    content = st.selectbox("Which website content do you want to use?", options)

    submitted = st.form_submit_button("Submit")

# Run the process to interact with the LLM
    if submitted:
        st.session_state.question = question
        results = search_for_answers(master_index, st.session_state.question)