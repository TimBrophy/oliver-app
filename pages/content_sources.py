import streamlit as st
from elasticsearch import Elasticsearch
import os
import uuid
import re
import requests
import json
from requests.auth import HTTPBasicAuth
from urllib.parse import urlsplit

os.environ['elastic_cloud_id'] = st.secrets['cloud_id']
os.environ['elastic_user'] = st.secrets['user']
os.environ['elastic_password'] = st.secrets['password']
os.environ['elasticsearch_url'] = st.secrets['elasticsearch_url']
os.environ['enterprise_search_url'] = st.secrets['enterprise_search_url']
master_index = 'search-content-master'

# ------------------------------------------
#        connect to elasticsearch
# ------------------------------------------

es = Elasticsearch(
    cloud_id=os.environ['elastic_cloud_id'],
    basic_auth=(os.environ['elastic_user'], os.environ['elastic_password'])
)


def crawl_request():
    auth = HTTPBasicAuth(os.environ['elastic_user'], os.environ['password'])
    enterprise_search_url = os.environ['enterprise_search_url']

    response = requests.post(
        f'{enterprise_search_url}/api/ent/v1/internal/indices/{master_index}/crawler2/crawl_requests',
        auth=auth)
    if response.status_code == 200:
        st.write("Crawl initiated")
    elif response.status_code == 400:
        st.write("Theres currently an active crawl happening already.")


def delete_domain(domain_id):
    auth = HTTPBasicAuth(os.environ['elastic_user'], os.environ['password'])
    enterprise_search_url = os.environ['enterprise_search_url']

    response = requests.delete(
        f'{enterprise_search_url}/api/ent/v1/internal/indices/{master_index}/crawler2/domains/{domain_id}',
        auth=auth)
    delete_query = {
        "match": {
            "domain_id": domain_id
        }

    }
    es.delete_by_query(index='content-sources', query=delete_query)
    return


def add_domain(url):
    auth = HTTPBasicAuth(os.environ['elastic_user'], os.environ['password'])
    enterprise_search_url = os.environ['enterprise_search_url']

    data = {
        'name': url
    }
    requests.post(f'{enterprise_search_url}/api/ent/v1/internal/indices/{master_index}/crawler2/domains', json=data,
                  auth=auth)
    response = requests.get(f'{enterprise_search_url}/api/ent/v1/internal/indices/{master_index}/crawler2/domains',
                            auth=auth)
    if response.status_code == 200:
        # Convert the response content to a JSON object
        json_response = response.json()
    for record in json_response["results"]:
        if record["name"] == url:
            domain_id = record["id"]
            return domain_id
    return


def add_entry_point(domain_id, entry_point):
    auth = HTTPBasicAuth(os.environ['elastic_user'], os.environ['password'])
    enterprise_search_url = os.environ['enterprise_search_url']
    data = {
        'value': entry_point
    }
    response = requests.post(
        f'{enterprise_search_url}/api/ent/v1/internal/indices/{master_index}/crawler2/domains/{domain_id}/entry_points',
        json=data,
        auth=auth)
    st.write(response.json())
    return


def validate_url(domain_url):
    url_pattern = r'^https:\/\/[\w.-]+(\.[\w.-]+)+([\w\-.,@?^=%&:/~+#]*[\w\-@?^=%&/~+#])?$'
    if re.match(url_pattern, domain_url):
        return domain_url
    else:
        corrected_url = f"https://{domain_url}"
        corrected_url = corrected_url.rstrip('/')
        return corrected_url


def add_source(doc, doc_id):
    response = es.index(index='content-sources', id=doc_id, document=doc)
    return


# -----------------------------------------------------------
#               PRESENTATION LOGIC
# -----------------------------------------------------------

# Start with a form to add a new domain to the engine/index
crawl = ''
field_list = ['name', 'url', 'domain_id']
results = es.search(index='content-sources', size=100, fields=field_list)
if results['hits']['total']['value'] > 0:
    with st.expander("Current content sources"):
        for hit in results['hits']['hits']:
            name = hit['_source']['name']
            domain_id = hit['_source']['domain_id']
            url = hit['_source']['url']
            get_docs_query = es.search(index=master_index, query={'match': {'domains': url}}, size=1)
            doc_num = json.dumps(get_docs_query['hits']['total']['value'])
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f'{name}: {doc_num} documents')
            with col2:
                st.write(url)
            with col3:
                delete = st.checkbox(label="delete", key=domain_id)
                if delete:
                    delete_domain(domain_id)
    crawl = st.button('Crawl all content')

st.header("Add a new content source")
with st.form("content_source"):
    name = st.text_input('Name', value="")
    url = st.text_input('Full url path', value="")
    url = validate_url(url).lower()

    submitted = st.form_submit_button("Submit")
    if submitted:
        url_parts = urlsplit(url)
        main_url = f"{url_parts.scheme}://{url_parts.netloc}"
        entry_point = url_parts.path
        # add the domain to the existing web crawler
        domain_id = add_domain(main_url)
        # now insert the record into our register of content sources
        doc_id = uuid.uuid4()
        doc = {
            "name": name,
            "url": main_url,
            "entry_point": entry_point,
            "domain_id": domain_id
        }
        add_source(doc, doc_id)
        # if there is a trailing directory add it as an entry point into the crawler config
        if len(entry_point):
            add_entry_point(domain_id, entry_point)

        st._rerun()

if crawl:
    crawl_request()