#%% md
# ## Nuru HIV Informational Chatbot
#%%
# Import libraries
import os
import json
import logging
from datetime import datetime
from lingua import Language, LanguageDetectorBuilder 
import gradio as gr
from openai import OpenAI as OpenAIOG
from llama_index.llms.openai import OpenAI
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
import utils.helpers as helpers

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API Key (Ensure this is set in the environment)
# load_dotenv("config.env")
os.environ.get("OPENAI_API_KEY")

# Initialize OpenAI clients
llm = OpenAI(temperature=0.0, model="gpt-4o")
client = OpenAIOG()

# Load index for retrieval
storage_context = StorageContext.from_defaults(persist_dir="arv_metadata")
index = load_index_from_storage(storage_context)
# retriever = index.as_retriever(similarity_top_k=10,
#                                 # Similarity threshold for filtering
#                                 similarity_threshold=0.5)

#%%
# Define Gradio function
def nishauri(user_params: str, conversation_history: list[str]):

    """Process user query, detect language, handle greetings, acknowledgments, and retrieve relevant information."""
    # context = " ".join([item["user"] + " " + item["chatbot"] for item in conversation_history])    
    # formatted_history = convert_conversation_format(conversation_history)
    # summary = summarize_conversation(formatted_history)
    user_params = json.loads(user_params)
    
    # Extract user information
    consent = user_params.get("CONSENT")
    person_info = user_params.get("PERSON_INFO", {})
    gender = person_info.get("GENDER", "")
    age = person_info.get("AGE", "")
    vl_result = person_info.get("VIRAL_LOAD", "")
    vl_date = helpers.convert_to_date(person_info.get("VIRAL_LOAD_DATETIME", ""), datetime)
    next_appt_date = helpers.convert_to_date(person_info.get("APPOINTMENT_DATETIME", ""), datetime)
    regimen = person_info.get("REGIMEN", "")
    question = user_params.get("QUESTION", "")

    info_pieces = [
        "Here is information about the person asking the question."
        f"The person is {gender}." if gender else "",
        f"The person is age {age}." if age else "",
        f"The person's next clinical check-in is scheduled for {next_appt_date}." if next_appt_date else "",
        f"The person is on the following regimen for HIV: {regimen}." if regimen else "",
        f"The person's most recent viral load result was {vl_result}." if vl_result else "",
        f"The person's most recent viral load was taken on {vl_date}." if vl_date else "",
    ]
    full_text = " ".join(filter(None, info_pieces))

    # detect language of user
    lang_question = helpers.detect_language(question, Language, LanguageDetectorBuilder, client, logging)
    print(lang_question)

    # If user is making a greeting or acknowledgement, address that accordingly
    intent = helpers.detect_intention(question, client = client)
    if intent == "greeting":
        prompt = f"""
        The user greeted you as follows: {question}.
        Respond by asking if they have any questions about HIV.
        Respond in {"Swahili" if lang_question == "sw" else "English"}.
        """
    elif intent == "acknowledgment":
        prompt = f"""
        The user acknowledged a response you gave to a prior question as follows {question}.
        Respond by saying you are ready to help if they have any more questions.
        Respond in {"Swahili" if lang_question == "sw" else "English"}.
        """         
    else:
        prompt = None

    if prompt:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        reply_to_user = completion.choices[0].message.content
        conversation_history.append({"user": question, "chatbot": reply_to_user})
        return reply_to_user, conversation_history

    # If the user is asking a question, proceed with the RAG pipeline
    # Translate if needed
    if lang_question == "sw":
        question = GoogleTranslator(source='sw', target='en').translate(question)
    
    # Retrieve relevant sources
    # sources = retriever.retrieve(question)
    # Summarize the conversation history
    history_summary = " ".join(
        [f"User: {turn['user']} Assistant: {turn['chatbot']}" for turn in conversation_history]
    )
    query_with_context = f"Current question: {question}\n\nSummary of prior context: {history_summary}"

    # Initialize the LLMRerank postprocessor
    reranker = LLMRerank(top_n=3)

    # Attach the reranker to the retriever
    retriever_with_rerank = index.as_retriever(
        similarity_top_k=10,
        similarity_threshold=0.6,
        postprocessors=[reranker]
    )

    # Retrieve and re-rank sources
    sources = retriever_with_rerank.retrieve(query_with_context)

    # Combine the top-ranked sources
    retrieved_text = "\n\n".join([f"Source {i+1}: {source.text}" for i, source in enumerate(sources)])


    # Combine into new user question - conversation history, new question, retrieved sources
    question_final = (
        f"The user asked the following question: \"{question}\"\n\n"
        f"Use only the content below to answer the question:\n\n{retrieved_text}\n\n"
        "Guidelines:\n"
        "- Only answer the question that was asked.\n"
        "- Do not change the subject or include unrelated information.\n"
        "- If the question is not about HIV, say that you can only answer HIV-related questions.\n"
    )

    # Set LLM instructions. If user consented, add user parameters, otherwise proceed without
    system_prompt = (
        "You are a helpful assistant who only answers questions about HIV.\n"
        "- Only answers questions about HIV (Human Immunodeficiency Virus).\n"
        "- Recognize that users may type 'HIV' with any capitalization (e.g., HIV, hiv, Hiv, etc.) or make minor typos (e.g., hvi, hiv/aids).\n"
        "- If a question is ambiguous or might be indirectly related to HIV (e.g., symptoms, illness, or general health concerns), assume it could be relevant to HIV and respond accordingly.\n"
        "- If a question is about using the Nishauri app, such as finding viral load results, regimen details, or the next appointment, provide clear instructions on how to navigate the app to find this information.\n"
        "- Do not answer questions about other topics (e.g., malaria or tuberculosis).\n"
        "- If a question is unrelated to HIV, politely respond that you can only answer HIV-related questions.\n\n"
    
        "The person asking the question is living with HIV.\n"
        "- Do not suggest they get tested for HIV or take post-exposure prophylaxis (PEP).\n"
        "- You may mention that their partners might benefit from testing or PEP, if relevant.\n"
        "- Do not mention in your response that the person is living with HIV.\n"
        "- Only suggest things relevant to someone who already has HIV.\n\n"
        "- Keep the answer under 50 words.\n"
        "- The user may user lowercase or slang for HIV or related terms.\n"
        "- Use simple, easy-to-understand language. Avoid medical jargon.\n"
    
        "Use the following authoritative information about viral loads:\n"
        "- A high or non-suppressed viral load is above 200 copies/ml.\n"
        "- A viral load above 1000 copies/ml suggests treatment failure.\n"
        "- A suppressed viral load is one below 200 copies/ml.\n\n"
    )
 
    if consent == "YES":
        system_prompt = f"{system_prompt} {full_text}."

    # Start with context
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history
    for turn in conversation_history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["chatbot"]})
    
    # Finally, add the current question
    messages.append({"role": "user", "content": question_final})

    # Generate response
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    # Collect response
    reply_to_user = completion.choices[0].message.content

    # add question and reply to conversation history
    conversation_history.append({"user": question, "chatbot": reply_to_user})  

    # If initial question was in swahili, translate response to swahili
    if lang_question=="sw":
        reply_to_user = GoogleTranslator(source='auto', target='sw').translate(reply_to_user) 

    # return system_prompt, conversation_history 
    return reply_to_user, conversation_history 

#%%
demo = gr.Interface(
    title = "Nuru Chatbot Demo",
    description="Enter a question and see the processed outputs in collapsible boxes",
    fn=nishauri,
    inputs=["text", gr.State(value=[])],
    outputs=[
        gr.Textbox(label = "Nuru Response", type = "text"),
        gr.State()
            ],
)

# for testing purposes to run as script
# if __name__ == "__main__":
#     demo.launch()

demo.launch()