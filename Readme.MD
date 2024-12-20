#**Odin School RAG-based Chatbot**
#Description
This project implements a Retrieval-Augmented Generation (RAG) based chatbot for Odin School. It utilizes Langchain to integrate language models with a PostgreSQL database for storing queries and answers. The chatbot allows users to ask questions about Odin School, retrieves relevant information from a pre-loaded dataset, and provides answers using a language model (OpenAI). The results are logged in a PostgreSQL database for further analysis.

#Prerequisites
Before running the project, ensure you have the following installed:

Python (3.x)
PostgreSQL
Langchain
Streamlit

Setup
Clone the repository:

git clone <https://github.com/ramona1999/rag_14dec>
cd Odin-School-RAG-Chatbot
Install the required Python packages:

pip install -r requirements.txt

#**Set up the PostgreSQL database**:

#Create a new PostgreSQL database.
Update the DATABASE_URL in the .env file with your PostgreSQL details:

DATABASE_URL=postgresql://<username>:<password>@localhost:5432/Odin_Rag_DB

#**Create a .env file:**

Create a .env file in the root directory of the project with the following content:

DATABASE_URL=postgresql://<username>:<password>@localhost:5432/Odin_Rag_DB

#Running the Chatbot
Start the application:

Streamlit run app1.py

#Access the Chatbot:

Open your web browser and navigate to the address provided by Streamlit (usually http://localhost:8501).

#Use the Chatbot:

Enter your questions about Odin School in the input field.
The chatbot will retrieve relevant data from the database and provide answers using the integrated language model.

#**Features**git 
Retrieval-Augmented Generation (RAG): Combines the power of pre-trained language models with a vector database for accurate and context-aware answers.
Database Logging: Records all queries and corresponding answers to the PostgreSQL database for analysis and future use.
Streamlit Integration: Provides an intuitive user interface for interacting with the chatbot.

#**Troubleshooting**
If you encounter any issues, ensure the following:

The PostgreSQL server is running.
The .env file is correctly set up with the database details.
The required packages are installed.
There are no conflicts with other packages installed in the environment.

Contact
For any issues or questions, feel free to contact the project maintainer:

Name: Ramona D Lima
Email: [ramonadlima@gmail.com]