from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("health_psychology.pdf")
documents = loader.load()

print(documents[0].page_content)