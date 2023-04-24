from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma, Pinecone
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import openai


loader = UnstructuredPDFLoader("2022-2023.pdf")
data = loader.load()
print(f'You have {len(data)} document(s) in your data')
print(f'There are {len(data[0].page_content)} characters in your document')

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)
print(f'Now you have {len(texts)} documents')

OPENAI_API_KEY = '...'
PINECONE_API_KEY = '...'
PINECONE_API_ENV = '...'
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "atom"
docsearch = Pinecone.from_texts(
    [t.page_content for t in texts], embeddings, index_name=index_name)
query = "What are examples of science classes?"
docs = docsearch.similarity_search(query, include_metadata=True)


llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")
query = "What are available music classes?"
docs = docsearch.similarity_search(query, include_metadata=True)
answer = chain.run(input_documents=docs, question=query)
print(answer)


def transcribe(answer, query):
    messages = [
        "You are an AI college course advisor. Help your students find the best classes based on this information:",
        f"Document content: {answer}",
        f"User question: {query}",
    ]

    prompt = "\n".join(messages)

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=80,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Return the generated response
    return response.choices[0].text.strip()


# Set up Streamlit layout
st.title("Document Question Answering")
query = st.text_input("Enter your question:")
submit_button = st.button("Submit")

# Function to run when user submits a question


# Function to run when user submits a question
def run_query(query):
    docs = docsearch.similarity_search(query, include_metadata=True)
    answer = chain.run(input_documents=docs, question=query)
    return answer


# Main app logic
if submit_button:
    if query:
        answer = run_query(query)
        response = transcribe(answer, query)
        # Display the generated response
        st.write(response)
    else:
        st.write("Please enter a question.")
