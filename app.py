#%% md
# ## Nuru HIV Informational Chatbot
#%%
# Import libraries
import os
import logging
import re
from langdetect import detect
from lingua import Language, LanguageDetectorBuilder
import gradio as gr
from openai import OpenAI as OpenAIOG
from llama_index.llms.openai import OpenAI
from llama_index.core import StorageContext, load_index_from_storage
from deep_translator import GoogleTranslator
from dotenv import load_dotenv

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
retriever = index.as_retriever(similarity_top_k=5)

# Define keyword lists
acknowledgment_keywords_sw = ["sawa", "ndiyo", "naam", "hakika", "asante", "nimeelewa", "nimekupata", "ni kweli", "kwa hakika", "nimesikia", "ahsante"]
acknowledgment_keywords_en = ["thanks", "thank you", "thx", "ok", "okay", "great", "got it", "appreciate", "good", "makes sense"]
follow_up_keywords = ["but", "also", "and", "what", "how", "why", "when", "is", "?", "lakini", "pia", "na", "nini", "vipi", "kwanini", "wakati"]
greeting_keywords_sw = ["sasa", "niaje", "habari", "mambo", "jambo", "shikamoo", "marahaba", "hujambo", "hamjambo", "salama", "vipi"]
greeting_keywords_en = ["hi", "hello", "hey", "how's it", "what's up", "yo", "howdy"]
#%%
# Define helper functions

def contains_exact_word_or_phrase(text, keywords):
    """Check if the given text contains any exact keyword from the list."""
    text = text.lower()
    return any(re.search(r'\b' + re.escape(keyword) + r'\b', text) for keyword in keywords)

def contains_greeting_sw(text):
    return contains_exact_word_or_phrase(text, greeting_keywords_sw)

def contains_greeting_en(text):
    return contains_exact_word_or_phrase(text, greeting_keywords_en)

def contains_acknowledgment_sw(text):
    return contains_exact_word_or_phrase(text, acknowledgment_keywords_sw)

def contains_acknowledgment_en(text):
    return contains_exact_word_or_phrase(text, acknowledgment_keywords_en)

def contains_follow_up(text):
    return contains_exact_word_or_phrase(text, follow_up_keywords)

def detect_language(text):
    """Detect language of a given text using Lingua for short texts and langdetect for longer ones."""
    if len(text.split()) < 5:
        languages = [Language.ENGLISH, Language.SWAHILI]
        detector = LanguageDetectorBuilder.from_languages(*languages).build()
        detected_language = detector.detect_language_of(text)
        return "sw" if detected_language == Language.SWAHILI else "en"
    try:
        return detect(text)
    except Exception as e:
        logging.warning(f"Language detection error: {e}")
        return "unknown"
#%%
# Define Gradio function
def nishauri(question, conversation_history: list[str]):

    """Process user query, detect language, handle greetings, acknowledgments, and retrieve relevant information."""
    context = " ".join([item["user"] + " " + item["chatbot"] for item in conversation_history])    
    
    # Process greetings and acknowledgments
    for lang, contains_greeting, contains_acknowledgment in [("en", contains_greeting_en, contains_acknowledgment_en), ("sw", contains_greeting_sw, contains_acknowledgment_sw)]:
        if contains_greeting(question) and not contains_follow_up(question):
            prompt = f"The user said: {question}. Respond accordingly in {lang}."
        elif contains_acknowledgment(question) and not contains_follow_up(question):
            prompt = f"The user acknowledged: {question}. Respond accordingly in {lang}."
        else:
            continue
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        reply_to_user = completion.choices[0].message.content
        conversation_history.append({"user": question, "chatbot": reply_to_user})
        source1return = ""
        source2return = ""
        source3return = ""
        source4return = ""
        source5return = ""
        return reply_to_user, source1return, source2return, source3return, source4return, source5return, conversation_history

    # Detect language and translate if needed
    lang_question = detect_language(question)
    if lang_question == "sw":
        question = GoogleTranslator(source='sw', target='en').translate(question)
    
    # Retrieve relevant sources
    sources = retriever.retrieve(question)
    retrieved_text = "\n\n".join([f"Source {i+1}: {source.text}" for i, source in enumerate(sources[:5])])

    source1return = ("File Name: " +
                     sources[0].metadata["file_name"] +
                     "\nPage Number: " +
                     sources[0].metadata["page_label"] +
                     "\n Source Text: " +
                     sources[0].text)
    
    source2return = ("File Name: " +
                     sources[1].metadata["file_name"] +
                     "\nPage Number: " +
                     sources[1].metadata["page_label"] +
                     "\n Source Text: " +
                     sources[1].text)
        
    source3return = ("File Name: " +
                     sources[2].metadata["file_name"] +
                     "\nPage Number: " +
                     sources[2].metadata["page_label"] +
                     "\n Source Text: " +
                     sources[2].text)

    source4return = ("File Name: " +
                     sources[3].metadata["file_name"] +
                     "\nPage Number: " +
                     sources[3].metadata["page_label"] +
                     "\n Source Text: " +
                     sources[3].text)

    source5return = ("File Name: " +
                     sources[4].metadata["file_name"] +
                     "\nPage Number: " +
                     sources[4].metadata["page_label"] +
                     "\n Source Text: " +
                     sources[4].text)

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
        "- Only answers questions about HIV (Human Immunodeficiency Virus). Recognize that users may type 'HIV' with any capitalization (e.g., HIV, hiv, Hiv, etc.) or make minor typos (e.g., hvi, hiv/aids). Use your best judgment to understand when a user intends to refer to HIV. Politely correct any significant misunderstandings, but otherwise proceed to answer normally.\n"
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
    return reply_to_user, source1return, source2return, source3return, source4return, source5return, conversation_history 

#%%
demo = gr.Interface(
    title = "Nuru Chatbot Demo",
    description="Enter a question and see the processed outputs in collapsible boxes.",
    fn=nishauri,
    inputs=["text", gr.State(value=[])],
    outputs=[
        gr.Textbox(label = "Nuru Response", type = "text"),
        gr.Textbox(label = "Source 1", max_lines = 10, autoscroll = False, type = "text"),
        gr.Textbox(label = "Source 2", max_lines = 10, autoscroll = False, type = "text"),
        gr.Textbox(label = "Source 3", max_lines = 10, autoscroll = False, type = "text"),
        gr.Textbox(label = "Source 4", max_lines = 10, autoscroll = False, type = "text"),
        gr.Textbox(label = "Source 5", max_lines = 10, autoscroll = False, type = "text"),
        gr.State()
            ],
)

demo.launch()