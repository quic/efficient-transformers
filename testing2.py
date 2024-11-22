
from langchain.document_loaders import HuggingFaceDatasetLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import torch
import onnx
import onnxruntime
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
modelPath = "sentence-transformers/all-MiniLM-l6-v2"

# Create a dictionary with model configuration options, specifying to use the CPU for computations
model_kwargs = {'device':'cpu'}

# Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
encode_kwargs = {'normalize_embeddings': False}

# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)
import ipdb; ipdb.set_trace()

print("c")

text = "This is a test document."
query_result = embeddings.embed_query(text)
query_result[:3]

# Vector store
db = FAISS.from_documents(docs, embeddings)
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
retriever = db.as_retriever()
docs = retriever.get_relevant_documents("What is Cheesemaking?")
print(docs[0].page_content)

# Create a retriever object from the 'db' with a search configuration where it retrieves up to 4 relevant splits/documents.
retriever = db.as_retriever(search_kwargs={"k": 4})

# Create a question-answering instance (qa) using the RetrievalQA class.
# It's configured with a language model (llm) and a retriever.
qa = RetrievalQA(llm=llm, retriever=retriever)

# Export the model to ONNX format
dummy_input = {
    "input_ids": torch.tensor([[0] * 512]),  # Adjust the input size as needed
    "attention_mask": torch.tensor([[0] * 512])
}
torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    "model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["start_logits", "end_logits"],
    dynamic_axes={"input_ids": {0: "batch_size"}, "attention_mask": {0: "batch_size"}}
)

# Verify the ONNX model
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)

# Load the ONNX model with ONNX Runtime
ort_session = onnxruntime.InferenceSession("model.onnx")

# Define a function to run inference with ONNX Runtime
def run_onnx_inference(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    ort_inputs = {k: v.cpu().numpy() for k, v in inputs.items()}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs

# Test the ONNX model
onnx_result = run_onnx_inference("What is cheesemaking?")
print(onnx_result)
