from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA

# Specify the dataset name and the column containing the content
dataset_name = "databricks/databricks-dolly-15k"
page_content_column = "context"  # or any other column you're interested in

# Create a loader instance
loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)

# Load the data
data = loader.load()

# Display the first 15 entries
import ipdb; ipdb.set_trace()
data[:2]


# Create an instance of the RecursiveCharacterTextSplitter class with specific parameters.
# It splits text into chunks of 1000 characters each with a 150-character overlap.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

# 'data' holds the text you want to split, split the text into documents using the text splitter.
docs = text_splitter.split_documents(data)
import ipdb; ipdb.set_trace()
print("B")

# Define the path to the pre-trained model you want to use
modelPath = "BAAI/bge-large-en-v1.5"

# Create a dictionary with model configuration options, specifying to use the CPU for computations
model_kwargs = {'device':'cpu'}

# Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
encode_kwargs = {'normalize_embeddings': False}

# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
)
import ipdb; ipdb.set_trace()

print("c")

text = "This is a sample input"
query_result = embeddings.embed_query(text)
query_result[:3]


from transformers import AutoTokenizer, AutoModel,AutoConfig
import onnx
import onnxruntime as ort
import numpy as np
import torch
from QEfficient.transformers.pytorch_transforms import CustomOpsTransform
# Load the model and tokenizer
model_name = 'BAAI/bge-large-en-v1.5'
config = AutoConfig.from_pretrained(model_name)
config.add_pooling_layer = False  

import ipdb; ipdb.set_trace()
model = AutoModel.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)
# Dummy input for the model
text = "This is a sample input"
inputs = tokenizer(text, return_tensors="pt")

    
with torch.no_grad():
    pt_outputs1 = model(**inputs)    

import ipdb; ipdb.set_trace()

#vector store
db = FAISS.from_documents(docs, embeddings,)
question = "What is cheesemaking?"
searchDocs = db.similarity_search(question)
import ipdb; ipdb.set_trace()
print(searchDocs[0].page_content)


# Create a tokenizer object by loading the pretrained "Intel/dynamic_tinybert" tokenizer.
tokenizer = AutoTokenizer.from_pretrained("Intel/dynamic_tinybert")

# Create a question-answering model object by loading the pretrained "Intel/dynamic_tinybert" model.
model = AutoModelForQuestionAnswering.from_pretrained("Intel/dynamic_tinybert")

# Specify the model name you want to use
model_name = "Intel/dynamic_tinybert"

# Load the tokenizer associated with the specified model
tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=512)

# Define a question-answering pipeline using the model and tokenizer
question_answerer = pipeline(
    "question-answering", 
    model=model_name, 
    tokenizer=tokenizer,
    return_tensors='pt'
)

# Create an instance of the HuggingFacePipeline, which wraps the question-answering pipeline
# with additional model-specific arguments (temperature and max_length)
llm = HuggingFacePipeline(
    pipeline=question_answerer,
    model_kwargs={"temperature": 0.7, "max_length": 512},
)

# Create a retriever object from the 'db' using the 'as_retriever' method.
# This retriever is likely used for retrieving data or documents from the database.
retriever = db.as_retriever()
docs = retriever.get_relevant_documents("What is Cheesemaking?")
print(docs[0].page_content)
# Create a retriever object from the 'db' with a search configuration where it retrieves up to 4 relevant splits/documents.
retriever = db.as_retriever(search_kwargs={"k": 4})

# Create a question-answering instance (qa) using the RetrievalQA class.
# It's configured with a language model (llm), a chain type "refine," the retriever we created, and an option to not return source documents.
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever, return_source_documents=False)


question = "Who is Thomas Jefferson?"
result = qa.run({"query": question})
print(result["result"])

