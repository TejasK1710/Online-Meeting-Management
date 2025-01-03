import streamlit as st 
import os
from post_meeting_bot import Post_meeting_summary_generation_bot

# Module to delete all the files from the local directory 
import shutil

# Library for extracting transcript from video recordings
import assemblyai as aai

# Set up the API key for accessing the Assembly AI transcription service
aai.settings.api_key = os.getenv('ASSIMBLYAI_API_KEY')
transcriber = aai.Transcriber()

st.title("Post Meeting Tab")

if "pre_meeting_run" not in st.session_state:
    st.session_state.pre_meeting_run=0
    st.session_state.output_agenda='First upload atleast two files containing discussion points in the Home Page section.'

# This (Post_Meeting_Tab) tab can be access without uploading the discussion points files 
# Prevent from crashing the application if someone directly use this tab.  
if "post_meeting_output" not in st.session_state:
    st.session_state.post_meeting_output='Upload the meeting Video'
    st.session_state.remaining_issues='Initialized'

# Define the folder where uploaded Meeting video will be saved
upload_folder = "Uploaded_meeting_video"

# Create the folder if doesn't exist
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

# Function to delete all files in the folder
# When new video is uploaded this makes sure that the recent one is used and prevents from unnecessary excess memory consumption
def delete_all_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)  
        except Exception:
            st.error(f"Error deleting file")

delete_all_files_in_folder(upload_folder)

# Ensure the folder still present after deleting all the previous files
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

# Allow to upload a single meeting video at a time
uploaded_file = st.file_uploader("Upload the Meetings recording", type=["mp4", "avi", "mkv"])

if uploaded_file is not None:
    meeting_video_path = os.path.join(upload_folder, uploaded_file.name)
    
    # Save the video file on local disk to extract transcript from the video rather than accessing streamlit storage
    with open(meeting_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.caption("Meeting Video Preview")

    # Display a preview of the uploaded video 
    with st.container():  
        col1, col2 = st.columns([2, 3])  
        with col1:
            st.video(meeting_video_path) 

        # Just to resize the preview window of the video
        st.write("""
        <style>
            .stVideo > video {
                width: 200px;  /* Adjust width as desired */
                height: 150px; /* Adjust height as desired */
            }
        </style>
        """, unsafe_allow_html=True)
    
    # Extract and process the text (transcript) from the video file with the help of Assembly AI API
    # Usually takes more time to process 
    # with st.spinner('Extracting and processing the text....'):
    #     post_meeting_transcript = transcriber.transcribe(meeting_video_path)

    # For testing purpose I have made a varibale that contains the transcript of the meeting. To use it uncomment the below and comment the above two lines 
    post_meeting_transcript="""Hello everyone. So shall we start todays meeting. Ok we will start after 5 min. Wait for five minutes. Let everyone join. Is everyone ready now. Ok. So myself Nachiket Jape. Today we will be discussing about the company progress. Or whatever things you have done in last few weeks. So anyone want to start. Hello sir myself Saurabh Jaiswal. We are getting lot of problems with the EC2 machine. The cloud infrastructure need to be fixed. Its not able to handle lot of traffic. Every time it crosses 10k users. It crashes. It will be good If you tell rohit to look into it and fix the issue. Ok so Rohit are you there. Yes Sir. So rohit check the cloud infrastructure. Sir I think the users are so much that the virtual machine is not able to handle it. I think if we upgrade the instance then it will be solved. Ok do it then. Hello sir. Myself Basant. I think we should open our new centre in Pune as there’s lot infrastructure and also we can manage the traffic routing there. Himanshu lives there. So I think it will be good if he can do more research for the exact location. I think Ayush is good in this work. So ayush you go and check the location. See if Pune fits for us. Ok will look into it. Sir the customer are getting lots of queries. I think we should make a team for it. It will be very helpful. It can benefit our customer relationship and will also help to keep us in the market. Ok. Dhruv look into this customer support. Do it as soon as possible. Anyone want to say anything more. Ok then meet you all in the next meeting. Thank you. Bye. Bye sir. Bye sir."""
    
    post_meeting_input='Generate the report of the meeting.'
    
    # LLM model to summarize the extracted text from the meeting video
    # Load LLM model from the Post_meeting_summary_generation_bot class
    st.session_state.post_meeting_output=Post_meeting_summary_generation_bot.rag_chain_for_post_meeting.invoke({
        "post_meeting_transcript": post_meeting_transcript,
        "input": post_meeting_input
    })

    st.header("Meeting Summary")
    st.markdown(st.session_state.post_meeting_output,unsafe_allow_html=True)
    print(st.session_state.post_meeting_output)

    st.divider()
    st.divider()

    # Below code only gets executed when you have uploaded the discussion points in the Home Tab
    # This code helps to find which issues were not addressed in the meeting based on the uploaded discussion points
    # It also finds those which were not fully resolved in the meeting
    # It uses pre_meeting_issues variable which is declared in Pre_Meeting_Tab
    if st.session_state.pre_meeting_issues== 'Discussion points not uploaded':
        st.write("Upload discussion points in the Home Tab to compare the remaining issues.")
    else:
        post_meeting_query_input='Find all the issue that were not discussed in the meeting.'
        st.session_state.remaining_issues = Post_meeting_summary_generation_bot.rag_chain_for_queries_post_meeting.invoke({
            "post_meeting_transcript": post_meeting_transcript,
            "pre_meeting_issues": st.session_state.pre_meeting_issues,
            "input": post_meeting_query_input
        })

        st.markdown(st.session_state.remaining_issues,unsafe_allow_html=True)
        print(st.session_state.remaining_issues)
    
        
