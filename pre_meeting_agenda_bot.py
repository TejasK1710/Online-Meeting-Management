from dotenv import load_dotenv
load_dotenv(".env")
import os

# ChatGroq speeds up the LLM process by providing fast AI inference
from langchain_groq import ChatGroq

# Pinecone vector database for storing and querying embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from langchain.schema.runnable import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader

# Google Generative AI embeddings are used to embed discussion points
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time

# Configure the google API key to access GoogleGenerativeAIEmbeddings
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API')
os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')

# Initialize the embedding model. "embedding-001" is chosen, which generates vectors with 768 dimensions
embedding_model = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

class Pre_meeting_agenda_generation_bot():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    # Loads the txt files having discussion points which were uploaded in the Home_Page
    loader = DirectoryLoader("user_before_meet_data/", glob="**/*.txt",loader_cls=TextLoader)
    docs = loader.load()
    texts = text_splitter.split_documents(docs)

    # Used LLaMA LLM model to generate the meeting agenda.
    # ChatGroq is used as it accelerates the LLM model inference
    llm_model = ChatGroq(model="llama3-8b-8192")

    # The index name must match the one declared in the database_creation file
    index_name="meeting-database"

    # Embed the discussion points and store them in the Pinecone vector database
    vectore_store = PineconeVectorStore.from_documents(
        documents=texts,
        index_name=index_name,
        embedding=embedding_model,
        namespace="wondervector5000"
    )
    print("data uploaded")

    time.sleep(3)

    # Here the value of k roughly determines the maximum number of text files that are used in agenda generation provided that the proceesed document chunk size is below 1000 for each one
    # K value should always be greater than the number of participents in the meeting (for this application)
    retriever = vectore_store.as_retriever(search_kwargs={"k": 20})

    template_for_meeting_data = """
    Based on the {context} provided organise the discussion points present in the context by grouping similar topics or the topics that belong to same domain.
    Then based on this, generate the meeting agenda or the meeting flow which guides how the meeting should happen.Decsribe in detail with respect to the context in the attractive format.
    It should provide the complete meeting agenda that can help to streamline the meeting flow. All discussion points should be processed.
    The topic that are mentioned more times or those which are very important should be highlighted in Major Issue section.
    If along with the discussion points, the person name is mentioned who is assigned the work then mention his name in the Assigned tasks to people section or if not then just mention the task.
    In the Introduction section, provide summary of all the discussion points.
    In the metadata, the source text file name is the name of the person who mentioned the points.
    In Proposed Points section mention the name of the person who gave the points alongs with the short names for the queries in an organized way. Like Cost Optimization in cloud Services, Customer Support issue.
    Only use the data provided in the context and only provide the output and no other things.

    For Example: Output
    Today Meeting Plan
    Introduction:''
    Queries need to discuss:''
    Major Issue:''
    Assigned tasks to people:''
    Departments Involved:''
    Proposed Points:''
    and any other importat topics like approximate time and all.

    Question:{input}
    Output:''
    """

    prompt_for_meeting_data = PromptTemplate(
        template=template_for_meeting_data,
        input_variables=["context", "input"]
    )

    # RAG chain for generating the meeting agenda based on the uploaded discussion points
    rag_chain = (
        {"context": retriever, "input": RunnablePassthrough()}
        | prompt_for_meeting_data
        | llm_model
        | StrOutputParser()
    )

    template_for_queries="""
    Based on the {context} provided organise the discussion points present in the context by linking similar topics.
    Mention all the issues in a form of bullet points that are mentioned in the context.
    In output give title for each and very short information about it.
    Only give output in the answer and don't mention other things.

    For Example: Output
    Issues:

    Question:{input}
    Output:''
    """

    prompt_for_queries_pre_meeting = PromptTemplate(
    template=template_for_queries,
    input_variables=["context", "input"]
    )

    # RAG chain for extracting and summarizing the unique issues from the uploaded discussion points
    rag_chain_for_queries_pre_meeting = (
    {"context": retriever, "input": RunnablePassthrough()}
    | prompt_for_queries_pre_meeting
    | llm_model
    | StrOutputParser()
    )


