import pickle
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load and preprocess the document
loader = UnstructuredPDFLoader("2022-2023.pdf")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

# Save the preprocessed data in a pickle file
with open("preprocessed_data.pkl", "wb") as f:
    pickle.dump(texts, f)
