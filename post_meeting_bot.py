import assemblyai as aai
import os
from dotenv import load_dotenv
load_dotenv(".env")

from langchain.schema.runnable import RunnablePassthrough, RunnableMap
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ChatGroq speeds up the LLM process by providing fast AI inference
from langchain_groq import ChatGroq

# Used LLaMA LLM model to generate the meeting agenda.
# ChatGroq is used as it accelerates the LLM model inference
llm_model = ChatGroq(model="llama3-8b-8192")

class Post_meeting_summary_generation_bot():
    template_for_post_meeting="""
    Based on the {post_meeting_transcript} provided, summarise the the meeting for someone who has not attended it.
    Mention the key decisions taken in the post_meeting_transcript or in the meeting. 
    Also find the key decisions made during the meeting or post_meeting_transcript. Give title and explain briefly.
    Also find what task was assigned to whome. Give name of the task and name of people who are assigned to it.
    Output the result in bullet points. Only output the result and no other things.

    For Example: Output
    ## **What was discussed in the meeting:**
    ## **Key decisions made during the meeting:** 
    ## **Assigned tasks:**

    Question:{input}
    Output:''
    """

    prompt_for_post_meeting = PromptTemplate(
    template=template_for_post_meeting,
    input_variables=["post_meeting_transcript", "input"]
    )

    # RAG chain for generates the meeting summary using the transcript extracted from the video
    rag_chain_for_post_meeting = (
    RunnableMap({
        "post_meeting_transcript": RunnablePassthrough(),
        "input": RunnablePassthrough()
    })
    | prompt_for_post_meeting
    | llm_model
    | StrOutputParser()
    )

    template_for_post_queries="""
    You are expert in comparing and finding out what topics were remaining from the discussion after the meeting. Or the topics which were discussed but not fully resolved or addressed during the meeting.
    Based on the {post_meeting_transcript} provided organise the similar points present in the post_meeting_transcript by linking similar topics.
    This is the transcript data of the meeting. Figure out all the issues that are mentioned or disucssed in the post_meeting_transcript or the meeting.
    Compare this with the data present in {pre_meeting_issues}. This contains the data that was feed before the meeting started. This issues were to be discussed in the meeting.
    Based on these two contexts, list all the issues there were not discussed or not fully resolved in the meeting. It means the issue that are present in pre_meeting_issues but were not discussed in the meeting or post_meeting_transcript.

    Only give output in the answer and don't mention other things.

    For Example: Output
    ## Issues Remaining or not addressed:''

    Question:{input}
    Output:''
    """

    prompt_for_queries_post_meeting = PromptTemplate(
    template=template_for_post_queries,
    input_variables=["post_meeting_transcript","pre_meeting_issues","input"]
    )

    # RAG chain to identify the issues that were not resolved or skipped during the meeting
    # Uses output of the LLM model present in the pre_meeting_agenda_bot that generated the issues from the discussion points
    # Those issues are used to compare with the current issues to find the not addressed issue in the meeting 
    rag_chain_for_queries_post_meeting = (
        RunnableMap({
        "post_meeting_transcript": RunnablePassthrough(),
        "pre_meeting_issues": RunnablePassthrough(),
        "input": RunnablePassthrough()
    })
    | prompt_for_queries_post_meeting
    | llm_model
    | StrOutputParser()
    )



