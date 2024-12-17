import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import (
    StuffDocumentsChain,
    LLMChain,
    ConversationalRetrievalChain
)

# App UI
st.title("OdinSchool Conversational Chatbot")
st.sidebar.header("Configuration related to OpenAI")
api_key = st.sidebar.text_input("OpenAI Key Required", type="password")

st.header("Chat with OdinSchool's Website Content")
if "conversation" not in st.session_state:
    st.session_state["conversation"] = []

user_input = st.text_input("Enter your question", key="user_input")

if api_key:
    try:
        # Load the URLs and process data
        URLs = ["https://www.odinschool.com"]
        loaders = UnstructuredURLLoader(urls=URLs)
        data = loaders.load()

        # Split the data into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        chunks = text_splitter.split_documents(data)

        # Create embeddings using sentence-transformers model
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Store the chunks in a vector database
        vectordatabase = FAISS.from_documents(chunks, embedding_model)

        # Initialize the LLM
        llm = OpenAI(api_key=api_key)

        # Question Generator Template
        question_generator_template = (
            "Combine the chat history and follow-up question into "
            "a standalone question. Chat History: {chat_history}\n"
            "Follow-up question: {question}"
        )
        question_generator_prompt = PromptTemplate.from_template(question_generator_template)
        question_generator_chain = LLMChain(llm=llm, prompt=question_generator_prompt)

        # Combine Documents Chain Template
        combine_docs_template = (
            "Use the provided documents and the standalone question to generate a helpful answer. "
            "Documents: {documents}\nQuestion: {question}\nAnswer:"
        )
        combine_docs_prompt = PromptTemplate.from_template(combine_docs_template)
        combine_docs_chain = StuffDocumentsChain(
            llm_chain=LLMChain(llm=llm, prompt=combine_docs_prompt),
            document_variable_name="documents"  # Explicitly specify this variable
        )

        # Retriever
        retriever = vectordatabase.as_retriever()

        # Conversational Retrieval Chain
        chain = ConversationalRetrievalChain(
            combine_docs_chain=combine_docs_chain,
            retriever=retriever,
            question_generator=question_generator_chain
        )

        # Handle user input
        if user_input:
            with st.spinner("Thinking..."):
                result = chain({"question": user_input, "chat_history": st.session_state["conversation"]})
                answer = result["answer"]
                st.session_state["conversation"].append((user_input, answer))
                
                # Display conversation history
                st.text_area(
                    "Conversation History",
                    value="\n".join([f"User: {q}\nBot: {a}" for q, a in st.session_state["conversation"]]),
                    height=400
                )
                st.subheader("Generated Answer:")
                st.write(answer)
    except Exception as e:
        st.error(f"We ran into an error: {str(e)}")
else:
    st.warning("Please enter your OpenAI API key in the sidebar.")
