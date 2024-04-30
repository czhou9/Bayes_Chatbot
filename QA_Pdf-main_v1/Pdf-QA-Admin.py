import streamlit as st
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from openai import InvalidRequestError
from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import re

from openai.error import AuthenticationError

def ask_question(question, qa, chat_history):
    try:
        result = qa({"question": question, "chat_history": chat_history})
        
        chat_history.append((question, result['answer']))
        st.write(f"-> **Question**: {question}")
        st.write(f"**Answer**: {result['answer']}")
        
        st.button()
    except InvalidRequestError:
        st.write("Try another chain type as the token size is larger for this chain type")


def display_chat_history(chat_history):
    
    for i, (question, answer) in enumerate(chat_history):
        st.info(f"Question {i + 1}: {question}")
        st.success(f"Answer {i + 1}: {answer}")
        st.write("----------")


def load_embeddings(api_key):
    embedding = OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.load_local("embeeding/PdfEmbeedingdb1", embedding)
    return embedding, db


def process_uploaded_files(uploaded_files,api_key):
    embedding, db = load_embeddings(api_key)

    loader = PyPDFDirectoryLoader("embeeding/useruploaded")

    docs = loader.load_and_split(text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        ))
    metadata = []  # Empty list to store metadata
    pages=[]
    for i in range(len(docs)):
        # print(i)
        # print(docs[i].metadata)
        metadata.append(docs[i].metadata)
        pages.append(docs[i].page_content)

    db =  FAISS.from_texts(pages, embedding,metadatas=metadata)

    db1 = FAISS.load_local("embeeding/PdfEmbeedingdb1", embedding)
    db.merge_from(db1)
    db.save_local("embeeding/PdfEmbeedingdb1")

    return db




# Define function to start a new chat
def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])        
    st.session_state["history"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state.clear()
    # st.session_state.memory.buffer.clear()
    # st.session_state.memory


def main():

    st.title("Ask Your Blockchain Expert ")

    with st.sidebar.expander("API Key Input"):
        with st.form(key="api_form"):
            api_key = st.text_input("Enter your OpenAI API key:", type="password")
            submit_button = st.form_submit_button(label="Submit")

            if submit_button and api_key:
                # Perform actions using the API key
                st.success("API key submitted:")

    def download_chat_history():
        chat_history_str = "\n".join([f"Question {i+1}: {q}\nAnswer {i+1}: {a}\n----------" for i, (q, a) in enumerate(st.session_state.history)])

        # Save the chat history to a text file
        with open("chat_history.txt", "w") as file:
            file.write(chat_history_str)

        st.success("Chat history downloaded successfully!")

    if api_key:
            
        try    :
        
            embedding, db = load_embeddings(api_key)
            

            with st.sidebar.expander("File Uploader") :

                uploaded_files = st.file_uploader(
                    "Choose PDF files", type=["pdf"], accept_multiple_files=True
                )

            if uploaded_files:
                    
                # Create a folder to save the uploaded files
                folder_name = "useruploaded"
                os.makedirs(folder_name, exist_ok=True)

                # Save the uploaded files to the folder
                for file in uploaded_files:
                    file_path = os.path.join(folder_name, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                db = process_uploaded_files(uploaded_files,api_key)

            retriever = db.as_retriever()
            model = OpenAI(temperature=0, openai_api_key=api_key)

            if 'history' not in st.session_state:
                st.session_state['history'] = []

            if 'generated' not in st.session_state:
                st.session_state['generated'] = ["Hello ! Ask me anything about Blockchain 🤗"]

            if 'past' not in st.session_state:
                st.session_state['past'] = ["Hey ! 👋"]

            if 'memory' not in st.session_state:
                st.session_state.memory = ConversationBufferMemory(
                    memory_key="chat_history", return_messages=True
                    )


            question_generator = LLMChain(llm=model, prompt=CONDENSE_QUESTION_PROMPT)
        
            doc_chain = load_qa_with_sources_chain(llm=model, chain_type="stuff")

            chain = ConversationalRetrievalChain(
                retriever=retriever,
                memory=st.session_state.memory,
                
                question_generator=question_generator,
                combine_docs_chain=doc_chain,
            )

                #container for the chat history
            response_container = st.container()
            #container for the user's text input
            container = st.container()
            

            def conversational_chat(query):
                try:
            
                    with get_openai_callback() as cb:
                        result = chain({"question": query, "chat_history": st.session_state['history']})
                        reference_Doc = db.similarity_search(query)
                        answer = re.sub(r'SOURCES:.*', '', result["answer"])
                        

                    
                        st.session_state['history'].append((query, answer))
                    
                        return answer,reference_Doc
                    
                
                except InvalidRequestError:
                    st.write("Try another chain type as the token size is larger for this chain type")
            

            # Allow to download as well
            download_str = []
            query=""
            with container:
                
                with st.form(key='my_form', clear_on_submit=True):
                    
                    user_input = st.text_input("Query:", placeholder="Type Your Query (:", key='input')
                    submit_button = st.form_submit_button(label='Send',type='primary')


                if submit_button and user_input:
                    output,reference_Doc = conversational_chat(user_input)
                    output, reference_Doc = conversational_chat(user_input)
                    print(output)
                    if output.strip() == "I don't know.":
                        output = "I'm still learning and don't have the information you're looking for."
                    print(output)
                    
                    st.session_state['past'].append(user_input)
                    st.session_state['generated'].append(output)
                    
                    
                    if st.session_state['generated']:
                        with response_container:
                            for i in range(len(st.session_state['generated'])):
                                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="personas")
                                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")
                                download_str.append(st.session_state["past"][i])
                                download_str.append(st.session_state["generated"][i])
                    
                    with st.expander("Show Reference Documents:"):
                        # reference_Doc = db.similarity_search(user_input)
                            # docs = [...]  # Your list of documents
                        refercemetadata = []  # Empty list to store metadata
                        refercepages=[]
                        for i in range(len(reference_Doc)):

                            refercemetadata.append(reference_Doc[i].metadata)
                            refercepages.append(reference_Doc[i].page_content)

                        st.write(refercemetadata)
                
                
                # download_str = '\n'.join(download_str)
                # if download_str:
                #         st.sidebar.download_button('Download Conversion',download_str)

                # with st.sidebar.expander("**View Chat History**"):
                #     display_chat_history(st.session_state.history)


                if st.session_state.history:   
                    if st.button("Clear-all",help="Clear all chat"):
                        st.session_state.history=[]

                # st.session_state.me
                st.button("New Chat", on_click = new_chat, type='primary')
                # if st.button("Download Chat History", help="Download chat history as a text file"):
                #     download_chat_history()

        except AuthenticationError as e :
            link = "[Click here](https://platform.openai.com/account/api-keys)"
            st.error(f"Ensure the API key used is correct, clear your browser cache, or generate a new one {link}")   


    else :
        st.sidebar.warning("Please Enter Api Key")    




if __name__ == '__main__':
    main()
