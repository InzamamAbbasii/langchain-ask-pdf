import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import subprocess


def ensure_ollama_model(model_name):
    """Check if Ollama model is available, otherwise pull it."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if model_name not in result.stdout:
            st.warning(f"Model '{model_name}' not found locally. Pulling now...")
            subprocess.run(["ollama", "pull", model_name])
    except Exception:
        st.error("⚠️ Ollama is not running. Please start Ollama first (run `ollama serve`).")
        st.stop()


def main():
    st.set_page_config(page_title="Ask your PDF (Free)", layout="wide")
    st.header("📄 Chat with PDF - Free & Local (Ollama)")

    # Ensure models exist
    ensure_ollama_model("nomic-embed-text")
    ensure_ollama_model("mistral")

    # Upload PDF
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())

        # Split text
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Create embeddings locally using Ollama
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        knowledge_base = Chroma.from_texts(chunks, embeddings, persist_directory="chroma_db")

        # Setup retriever
        retriever = knowledge_base.as_retriever(search_kwargs={"k": 3})

        # Local LLM (Ollama)
        llm = Ollama(model="mistral")

        # Use RetrievalQA instead of load_qa_chain
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True
        )

        # Ask a question
        user_question = st.text_input("Ask something about your PDF:")
        if user_question:
            result = qa({"query": user_question})

            st.write("### Answer:")
            st.write(result["result"])

            # Optional: show sources
            with st.expander("See sources"):
                for doc in result["source_documents"]:
                    st.write(doc.page_content)


if __name__ == "__main__":
    main()

# from dotenv import load_dotenv
# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# # from langchain_community.embeddings import HuggingFaceEmbeddings
# # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import OpenAI
# from langchain.callbacks import get_openai_callback


# def main():
#     load_dotenv()
#     st.set_page_config(page_title="Ask your PDF")
#     st.header("Ask your PDF 💬")
    
#     # upload file
#     pdf = st.file_uploader("Upload your PDF", type="pdf")
    
#     # extract the text
#     if pdf is not None:
#       pdf_reader = PdfReader(pdf)
#       text = ""
#       for page in pdf_reader.pages:
#         text += page.extract_text()
        
#       # split into chunks
#       text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#       )
#       chunks = text_splitter.split_text(text)
      
#       # create embeddings
#       embeddings = OpenAIEmbeddings()
#       knowledge_base = FAISS.from_texts(chunks, embeddings)
      
#       # show user input
#       user_question = st.text_input("Ask a question about your PDF:")
#       if user_question:
#         docs = knowledge_base.similarity_search(user_question)
        
#         llm = OpenAI()
#         chain = load_qa_chain(llm, chain_type="stuff")
#         with get_openai_callback() as cb:
#           response = chain.run(input_documents=docs, question=user_question)
#           print(cb)
           
#         st.write(response)
    

# if __name__ == '__main__':
#     main()
