# Retrival Augmented Generation with LLaMA 2

# This document is used to supplement the article found here: https://www.amrits-blog.com/posts/RAG-With-Llama2

import torch
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from pinecone import Pinecone, PodSpec
from datasets import load_dataset
from torch import cuda, bfloat16
import transformers
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA

# Initialise the Embeddings:


# Select our embedding model to map inputs to a vector space
embed_model_id = 'sentence-transformers/all-MiniLm-l6-v2'
# Ensure we are using a GPU for training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Define Batch size for the embedding model
batch_size = 32
# Load the model onto our GPU
embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': batch_size}
)

# Let's now use the model to embed two sentences

docs = [
    "this is one document",
    "and another document"
]

# Embeddings will be a list where each element contains nested list of 384 values
embeddings = embed_model.embed_documents(docs)
# Extract the number of dimensions per sentence
number_of_dimensions = len(embeddings[0])

print(f"We have {len(embeddings)} embeddings, each with {number_of_dimensions} dimensions.")

# Building the Vector Index

# Load API Key
load_dotenv()
api_key = os.environ.get("PINECONE_API_KEY")

# Instantiate the Pinecone client with the API Key
pinecone = Pinecone(
    api_key=api_key
)

# Create the index
index_name = 'llama-2-rag'

if index_name not in pinecone.list_indexes().names():
    pinecone.create_index(
        name=index_name,
        dimension=number_of_dimensions,
        metric='cosine',
        spec=PodSpec(environment="gcp-starter")
    )

# Check if the index is ready to use
if pinecone.describe_index(index_name).status['ready']:
    print("Ready to go!")

    # Connect to the index
    index = pinecone.Index(index_name)
    print(index.describe_index_stats())


# Load dataset

data = load_dataset(
    'jamescalam/llama-2-arxiv-papers-chunked',
    split='train'
)

data = data.to_pandas()

# Iterate through each batch in the data
for i in range(0, len(data), batch_size):
    # Calculate the final index for each batch avoiding an index error for the final batch
    i_end = min(len(data), i + batch_size)
    # Extract the current batch
    batch = data.iloc[i:i_end]
    # Create a unique ID from doi + chunk_id
    ids = [f"{row['doi']}-{row['chunk-id']}" for _, row in batch.iterrows()]
    # Extract Text data and create embeddings
    texts = [row['chunk'] for _, row in batch.iterrows()]
    embeddings = embed_model.embed_documents(texts)
    # Generate Meta Data
    metadata = [
        {
            'text': row['chunk'],
            'source': row['source'],
            'title': row['title']
        } for _, row in batch.iterrows()
    ]

    # Upload to Pinecone
    index.upsert(vectors=zip(ids, embeddings, metadata))

index.describe_index_stats()


# Load the model

model_id = 'meta-llama/Llama-2-13b-chat-hf'

# Set quantization configuration
quantization_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# Load Hugging Face Token
load_dotenv()
hugging_face_token = os.environ.get('HF_AUTH_TOKEN')
# Set model configuration
model_config = transformers.AutoConfig.from_pretrained(
    pretrained_model_name_or_path=model_id,
    token=hugging_face_token
)

# Load model with quantization and model configurations
model = transformers.AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_id,
    trust_remote_code=True,
    config=model_config,
    # quantization_config=quantization_config,
    device_map='auto',
    token=hugging_face_token
)

# Set model to evaluation mode
model.eval()
print(torch.cuda.is_available())

# Load the Tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    token=hugging_face_token
)


generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    temperature=0.01,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

llm = HuggingFacePipeline(pipeline=generate_text)

# Implement RAG

# Pinecone requires this field for Metadata
text_field = 'text'

vectorstore = Pinecone(
    index,
    embed_model.embed_query,
    text_field
)


rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm, chain_type='stuff',
    retriever=vectorstore.as_retriever()
)

# Use our RAG pipeline
rag_pipeline('what is so special about llama 2?')