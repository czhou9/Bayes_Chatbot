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
import psutil
from langchain.document_loaders import PyPDFDirectoryLoader
import os
from openai.error import AuthenticationError


def get_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_usage_mb = memory_info.rss / (1024 * 1024)  # Memory usage in megabytes
    return memory_usage_mb


def ask_question(question, qa, chat_history):
    try:
        result = qa({"question": question, "chat_history": chat_history})
        chat_history.append((question, result['answer']))
        st.write(f"-> **Question**: {question}")
        st.write(f"**Answer**: {result['answer']}")
    except InvalidRequestError:
        st.write("Try another chain type as the token size is larger for this chain type")


def display_chat_history(chat_history):
    
    for i, (question, answer) in enumerate(chat_history):
        st.info(f"Question {i + 1}: {question}")
        st.success(f"Answer {i + 1}: {answer}")
        st.write("----------")


def load_embeddings(api_key):
    embedding = OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.load_local("embeeding/PdfEmbeedingdb", embedding)
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



def main():

    st.title("Ask Your Blockchain Expert ")

    with st.sidebar.expander("API Key Input"):
        with st.form(key="api_form"):
            api_key = st.text_input("Enter your OpenAI API key:", type="password")
            submit_button = st.form_submit_button(label="Submit")

            if submit_button and api_key:
                # Perform actions using the API key
                st.success("API key submitted:")



    if api_key:

        try :

            embedding, db = load_embeddings(api_key)
            

            with st.sidebar.expander("File Uploader") :

                uploaded_files = st.file_uploader(
                    "Choose PDF files", type=["pdf"], accept_multiple_files=True
                )


            if uploaded_files:
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
            # st.session_state.me

            

            chain = ConversationalRetrievalChain.from_llm(
                llm=model, memory=st.session_state.memory, retriever=retriever, chain_type="stuff"
            )

                #container for the chat history
            response_container = st.container()
            #container for the user's text input
            container = st.container()

            def conversational_chat(query):
                try:
            
                    with get_openai_callback() as cb:
                        result = chain({"question": query, "chat_history": st.session_state['history']})
                        st.session_state['history'].append((query, result["answer"]))
                        # expander = st.sidebar.expander("Token Details")

                        # if expander.button("Show token"):
                        #     expander.write(f"Total Tokens: {cb.total_tokens}")
                        #     expander.write(f"Prompt Tokens: {cb.prompt_tokens}")
                        #     expander.write(f"Completion Tokens: {cb.completion_tokens}")
                        #     expander.write(f"Total Cost (USD): ${cb.total_cost}")
                                    
                        return result["answer"]
                    
                
                except InvalidRequestError:
                    st.write("Try another chain type as the token size is larger for this chain type")
            

            # Allow to download as well
            download_str = []
            with container:
                
                with st.form(key='my_form', clear_on_submit=True):
                    
                    user_input = st.text_input("Query:", placeholder="Type Your Query (:", key='input')
                    submit_button = st.form_submit_button(label='Send',type='primary')
                    
                if submit_button and user_input:
                    output = conversational_chat(user_input)

                    print(output)
                    if output.strip() == "I don't know.":
                        output = "I'm still in the process of learning, and I may not have the exact information you're seeking. However, I'll do my best to assist you with any questions or concerns you have. Please let me know how I can be of help to you."
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
                    
                    
        
                
                
                # download_str = '\n'.join(download_str)
                # if download_str:
                #         st.sidebar.download_button('Download Conversion',download_str)

                # with st.sidebar.expander("**View Chat History**"):
                #     display_chat_history(st.session_state.history)

                if st.session_state.history:   
                    if st.button("Clear-all",help="Clear all chat"):
                        st.session_state.history=[]

                st.button("New Chat", on_click = new_chat, type='primary')
                
                # Allow the user to clear all stored conversation sessions
            # if st.button("Download Chat History"):
            #     # Create a string representation of the chat history
            #     chat_history_str = "\n".join([f"Question {i+1}: {q}\nAnswer {i+1}: {a}\n----------" for i, (q, a) in enumerate(st.session_state.memory)])

            #     # Save the chat history to a text file
            #     with open("QA_chat_history.txt", "w") as file:
            #         file.write(chat_history_str)

            #     st.success("Chat history downloaded successfully!")
            memory_usage = get_memory_usage()
        # st.write(f"Memory usage: {memory_usage} MB")
        except AuthenticationError as e :
            link = "[Click here](https://platform.openai.com/account/api-keys)"
            st.error(f"Ensure the API key used is correct, clear your browser cache, or generate a new one {link}")



    else :
        st.sidebar.warning("Please Enter Api Key")    




if __name__ == '__main__':
    main()
