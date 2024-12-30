from dotenv import load_dotenv
load_dotenv(".env")
import os
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API')
pc = Pinecone()

# Define the name of the index to be created in the Pinecone database
# Remember the name as same will be used in the pre_meeting_agenda_bot for saving the discussion points in the database
index_name = "meeting-database"

# Create the index if it doesn't already exist
# The database is set up to store vector dimension of 768. 
# This dimension is declared because of GoogleGenerativeAIEmbeddings which generates the embedding of 768. Its defined in pre_meeting_agenda_bot
if index_name not in pc.list_indexes():
  try:
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"Created index {index_name}")
  except:
    print(f"{index_name} Already exist")

print(pc.list_indexes()[0])