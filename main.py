#from dotenv import load_dotenv
#load_dotenv()

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os

# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


# 제목
st.title("ChatPDF로 물어보자🤓")
st.write("---")

# 파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 업로드하세요, OPENAI가 답변 해드려요", type=['pdf'])
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

if uploaded_file is not None:
    # PDF 파일 처리
    pages = pdf_to_document(uploaded_file)
    
    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_documents(pages)

    # 임베딩 모델
    embeddings_model = OpenAIEmbeddings()

    # Chroma에 로드
    db = Chroma.from_documents(texts, embeddings_model)

    # 질문 입력
    st.header("PDF에게 질문해보세요!")
    question = st.text_input('질문을 입력하세요')

    if st.button('질문하기'):
        with st.spinner('작동 중 입니다...'):
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
            result = qa_chain({"query": question})
            st.write(result["result"])
