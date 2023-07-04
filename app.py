from typing import Any
import gradio as gr
from langchain import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader

import fitz
from PIL import Image

import chromadb
import re
import uuid
from dotenv import load_dotenv

from pypdf import PdfReader 

def get_pdf_text(file):
    text = ""
    print(file.name)
    pdf_reader = PdfReader(file.name)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,
        chunk_overlap=0,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def add_text(history, text):
    if not text:
         raise gr.Error('enter text')
    history = history + [(text,'')] 
    return history

class my_app:
    def __init__(self, OPENAI_API_KEY= None ) -> None:
        load_dotenv()
        self.OPENAI_API_KEY = "EMPTY"
        self.chain = None
        self.chat_history = []
        self.N = 0
        self.count = 0
        

    def __call__(self, file) -> Any:
        if self.count==0:
            print('This is here')
            self.build_chain(file)
            self.count+=1

        return self.chain
    
    def chroma_client(self):
        #create a chroma client
        client = chromadb.Client()
        #create a collecyion
        collection = client.get_or_create_collection(name="my-collection")

        return client
    
    
        
    def process_file(self,file):
        # get pdf text
        raw_text = get_pdf_text(file)
 
        # get the text chunks
        text_chunks = get_text_chunks(raw_text)

        # create vector store
        vectorstore = get_vectorstore(text_chunks)
        
        pattern = r"/([^/]+)$"
        match = re.search(pattern, file.name)
        print(match)
        file_name = str.lower(match.group(1))
        return vectorstore, file_name
    
    def build_chain(self, file):
        vectorstore, file_name = self.process_file(file)
        
        self.chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(request_timeout=1200, temperature=0.0, openai_api_key=self.OPENAI_API_KEY, model_name = "gpt-3.5-turbo"), 
                                                            retriever=vectorstore.as_retriever(search_kwargs={"k": 1}),
                                                            return_source_documents=True,)
        return self.chain
    

def get_response(history, query, file):
        
        
        if not file:
            raise gr.Error(message='Upload a PDF')
           
        chain = app(file)

        result = chain({"question": query, "chat_history": app.chat_history},return_only_outputs=True)
        app.chat_history += [(query, result["answer"])]

        for char in result['answer']:
           history[-1][-1] += char
           yield history,''


def render_first(file):
        doc = fitz.open(file.name)
        page = doc[0]
        #Render the page as a PNG image with a resolution of 300 DPI
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        return image,[]

app = my_app()
with gr.Blocks() as demo:
    state = gr.State(uuid.uuid4().hex)
    with gr.Column():
        with gr.Row():           
            chatbot = gr.Chatbot(value=[], elem_id='chatbot').style(height=650)
            show_img = gr.Image(label='Upload PDF', tool='select' ).style(height=680)
    with gr.Row():
        with gr.Column(scale=0.60):
            txt = gr.Textbox(
                        show_label=False,
                        placeholder="Enter text and press enter",
                    ).style(container=False)
        with gr.Column(scale=0.20):
            submit_btn = gr.Button('submit')
        with gr.Column(scale=0.20):
            btn = gr.UploadButton("📁 upload a PDF", file_types=[".pdf"]).style()
        
  
    btn.upload(fn=render_first, inputs=[btn], outputs=[show_img,chatbot],)
    
    submit_btn.click(fn=add_text, inputs=[chatbot,txt], outputs=[chatbot, ], queue=False).success(fn=get_response,inputs = [chatbot, txt, btn],
                                    outputs = [chatbot,txt])

    
demo.queue()
demo.launch()  
