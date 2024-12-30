import streamlit as st
import os
import nltk
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader

# Initialize session state to store decision points and download 'punkt' only once, avoiding repetitive downloads irrespective of Home_page clicks.
if 'docs' not in  st.session_state:
    nltk.download('punkt')
    st.session_state.docs = ''

# Initialize the variables to be used in Pre_meeting_tab here so as to prevent from error if someone directly go to Pre Meeting tab without uploading files.
if "pre_meeting_run" not in st.session_state:
    st.session_state.pre_meeting_run=0
    st.session_state.output_agenda='First upload atleast two files containing discussion points in the Home Page section.'
    st.session_state.pre_meeting_issues='Discussion points not uploaded'
 
st.title("Online Meeting Management with AI")

st.markdown("<h4>Upload the txt file having discussion points that you want to discuss.</h4>", unsafe_allow_html=True)

# Define the folder where uploaded files will be stored.
UPLOAD_FOLDER = 'user_before_meet_data/'

# Create the folder if doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Allow to accept multiple txt file having discussion points
uploaded_files = st.file_uploader(
    "(Start with 'Discussion Points:' on the first line, followed by your discussion points in either bullet points or a paragraph. Save the file as .txt using your name as the filename (e.g., firstname_lastname.txt).)",
    type="txt", accept_multiple_files=True
)

# Executes when someone uploades the files. 
if uploaded_files:
    # Whenever the files are uploaded pre_meeting_run set to 1 to start the new session in Pre_meeting_tab
    st.session_state.pre_meeting_run=1
    for file in uploaded_files:
        file_name = file.name
        
        # Save the uploaded file in the provided folder
        save_path = os.path.join(UPLOAD_FOLDER, file_name)
        with open(save_path, 'wb') as f:
            f.write(file.getbuffer())
    
    # Load all the files present in the mentioned folder and store in docs variable to access in Pre_meeting_tab
    loader = DirectoryLoader("user_before_meet_data/", glob="**/*.txt",loader_cls=TextLoader)
    docs = loader.load()
    st.session_state.docs=docs
    st.write(f"Total files uploaded are: {len(st.session_state.docs)}")

    st.info("Now select 'Pre Meeting Tab' from sidebar for getting Meeting Agenda.")


