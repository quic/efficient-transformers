from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
import onnxruntime as ort
import numpy as np

# Step 1: Load ONNX-based embedding model
def embed_text(text, ort_session, tokenizer, seq_len=128):
   inputs = tokenizer(text, return_tensors="np", truncation=True, padding="max_length", max_length=seq_len)
   outputs = ort_session.run(None, {
       "input_ids": inputs["input_ids"],
       "attention_mask": inputs["attention_mask"]
   })
   return np.mean(outputs[0], axis=1).squeeze().tolist()

# Initialize ONNX model and tokenizer
model_name = "BAAI/bge-large-en-v1.5"
onnx_model_path = "bge-large-en-v1.5.onnx"
tokenizer = AutoTokenizer.from_pretrained(model_name)
ort_session = ort.InferenceSession(onnx_model_path)

# Step 2: Create FAISS retriever
documents = [
   "LangChain is a framework for developing applications powered by language models.",
   "OpenAI provides state-of-the-art language models that can generate human-like text."
]
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = [chunk for doc in documents for chunk in text_splitter.split_text(doc)]
embeddings = [embed_text(doc, ort_session, tokenizer) for doc in split_docs]
vector_store = FAISS.from_texts(split_docs, embeddings)

# Step 3: Load GPT-2
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Step 4: Query RAG pipeline
query = "What is LangChain used for?"
retrieved_docs = vector_store.similarity_search(query, k=2)
context = " ".join([doc.page_content for doc in retrieved_docs])
prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
inputs = gpt2_tokenizer.encode(prompt, return_tensors="pt")
outputs = gpt2_model.generate(inputs, max_length=100)

# Step 5: Print result
answer = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Answer:", answer)
print("Sources:", [doc.metadata for doc in retrieved_docs])