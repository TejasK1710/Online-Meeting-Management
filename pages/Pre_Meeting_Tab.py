import streamlit as st

st.title("Pre Meeting Tab")

# Initialize variables to prevent system from crash when other tabs are directly accessed without visiting this tab
# The 'pre_meeting_run' variable is set to 0 to avoid loading the LLM model unless files are uploaded in the Home Tab.
if "pre_meeting_run" not in st.session_state:
    st.session_state.pre_meeting_run=0
    st.session_state.output_agenda='First upload atleast two files containing discussion points in the Home Page section.'
    st.session_state.pre_meeting_issues='Discussion points not uploaded.'

# If someone uploades the files then only this block executes. Maintain the session state
if(st.session_state.pre_meeting_run==1):

    # Loads the LLM model only when files are uploaded  
    from pre_meeting_agenda_bot import Pre_meeting_agenda_generation_bot

    # Reset the session and prevents from loading the LLM model if the Pre_meeting_tab is clicked again without uploading new files in Home_Page 
    st.session_state.pre_meeting_run=0

    input_question_for_agenda = "Discuss the points in the context of this meeting"

    # Invoke the LLM model from the 'Pre_meeting_agenda_generation_bot' class to generate the meeting agenda.
    st.session_state.output_agenda=Pre_meeting_agenda_generation_bot.rag_chain.invoke(input_question_for_agenda)

    st.info("Visit the 'Post-Meeting' tab to compare given queries with the meeting outcome.")


    pre_meeting_query_input='Mention all the issues provided in the discussion points.'

    # The LLM model identifies all the unique issues mentioned in the uploaded files.
    # This variable is later used in the Post-Meeting_Tab to check which issues were not addressed during the meeting.
    st.session_state.pre_meeting_issues=Pre_meeting_agenda_generation_bot.rag_chain_for_queries_pre_meeting.invoke(pre_meeting_query_input)

st.write(st.session_state.output_agenda)






