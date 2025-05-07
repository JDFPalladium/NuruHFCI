#%% md
# ## Nuru HIV Informational Chatbot
#%%
# Import libraries
import os
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
retriever = index.as_retriever(similarity_top_k=10,
                                # Similarity threshold for filtering
                                similarity_threshold=0.5,
                                # Use LLM reranking to filter results
                                reranker=LLMRerank(top_n=3))

#%%
# Define Gradio function
def nishauri(question, conversation_history: list[str]):

    """Process user query, detect language, handle greetings, acknowledgments, and retrieve relevant information."""
    # context = " ".join([item["user"] + " " + item["chatbot"] for item in conversation_history])    
    # formatted_history = convert_conversation_format(conversation_history)
    # summary = summarize_conversation(formatted_history)

    # detect language of user
    lang_question = helpers.detect_language(question, Language, LanguageDetectorBuilder, client)
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
    sources = retriever.retrieve(question)
    retrieved_text = "\n\n".join([f"Source {i+1}: {source.text}" for i, source in enumerate(sources[:3])])

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
        "- Use your best judgment to understand when a user intends to refer to HIV. Politely correct any significant misunderstandings, but otherwise proceed to answer normally.\n"
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

demo.launch()