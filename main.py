import PyPDF2
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import streamlit as st

# Function to extract text from a PDF file
def pdf_to_text(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

# Function to chunk text into smaller pieces
def chunk_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

# Load and chunk the PDF text
college_info_text = pdf_to_text("Mandatory Disclosure22-23.pdf")
text_chunks = chunk_text(college_info_text)

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = model.encode(text_chunks, convert_to_tensor=True)

# Function to retrieve relevant chunks based on a query
def retrieve_relevant_chunks(query, chunk_embeddings, text_chunks, model, top_k=1):
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = torch.nn.functional.cosine_similarity(query_embedding.unsqueeze(0), chunk_embeddings)
    top_k_indices = torch.topk(similarities.clone().detach(), top_k).indices.tolist()
    
    relevant_chunks = [text_chunks[i] for i in top_k_indices]
    
    # Filter out any irrelevant chunks
    filtered_chunks = [chunk for chunk in relevant_chunks if query.lower() in chunk.lower()]
    
    return filtered_chunks if filtered_chunks else relevant_chunks


# Load GPT-2 model and tokenizer
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# Use GPU if available
generator = pipeline('text-generation', model=gpt2_model, tokenizer=tokenizer, device=-1)

# Function to generate a response using the GPT-2 model
def generate_response(query, relevant_chunks, generator, max_new_tokens=50):
    context = " ".join(relevant_chunks)
    input_text = f"{context}\n\nQuestion: {query}\nAnswer:"
    response = generator(input_text, max_new_tokens=max_new_tokens, num_return_sequences=1)
    return response[0]['generated_text']

# Streamlit UI
# st.title("")import streamlit as st

# Use HTML to center the image and the title
import streamlit as st

st.image("/Users/priyadharshinim/Desktop/hackathon/images.jpeg", width=70)

# Centering the title
st.markdown(
    """
    <h1 style='text-align: center;'>PSG Indigenous Chatbot</h1>
    """,
    unsafe_allow_html=True
)


query = st.text_input("Ask a question:")

if query:
    relevant_chunks = retrieve_relevant_chunks(query, chunk_embeddings, text_chunks, model)
    response = generate_response(query, relevant_chunks, generator)
    st.write(response)
