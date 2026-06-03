
#### File 3: `app.py`
import os
import gradio as gr
import torch

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

vector_store = None
llm = None

def load_model():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=quantization_config, 
        device_map="auto"
    )
    text_generation_pipeline = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=512, temperature=0.1, return_full_text=False,
        pad_token_id=tokenizer.eos_token_id
    )
    return HuggingFacePipeline(pipeline=text_generation_pipeline)

print("Loading model...")
llm = load_model()
print("✅ Model loaded!")

def process_pdf(file_obj):
    global vector_store
    if file_obj is None: return "⚠️ Please upload a PDF file first."
    
    loader = PyPDFLoader(file_obj.name)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return f"✅ Processed: {os.path.basename(file_obj.name)}. Ready to chat!"

def chat_with_pdf(message, history):
    global vector_store, llm
    if vector_store is None or llm is None:
        return history + [[message, "⚠️ Please upload and process a PDF first!"]]
    
    retrieved_docs = vector_store.similarity_search(message, k=3)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    history_text = ""
    if history:
        for user_msg, ai_msg in history:
            history_text += f"User: {user_msg}\nAssistant: {ai_msg}\n"
    
    prompt = f"""You are a helpful enterprise AI assistant. Answer based ONLY on the provided context. 
If the answer is not in the context, say "I don't know based on the provided document."

Context:
{context}

Chat History:
{history_text}

User Question: {message}
Assistant:"""
    
    try:
        response = llm.invoke(prompt)
        answer = response.strip()
    except Exception as e:
        answer = f"⚠️ Error: {str(e)}"
    
    return history + [[message, answer]]

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📄 Open PDF Assistant")
    gr.Markdown("100% Free. No API Keys. Runs locally.")
    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            process_btn = gr.Button("Process Document", variant="primary")
            status_output = gr.Textbox(label="Status", interactive=False)
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat", height=400)
            msg_input = gr.Textbox(label="Ask a question...", placeholder="e.g., What is the policy?")
            clear_btn = gr.ClearButton([msg_input, chatbot])

    process_btn.click(fn=process_pdf, inputs=[pdf_input], outputs=[status_output])
    msg_input.submit(fn=chat_with_pdf, inputs=[msg_input, chatbot], outputs=[chatbot])

if __name__ == "__main__":
    demo.launch(share=True)
    