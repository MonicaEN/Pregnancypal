import requests
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
# pip install -U langchain-huggingface (for removing depreciation warning)
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
import smtplib
import imaplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
import datetime  # For timestamp tracking
import email.utils  # For robust email date parsing
import asyncio  # For non-blocking delays

# --- NEW IMPORTS FOR ML MODEL ---
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os
# ---------------------------------

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Provide a helpful response based on the information provided. 
If the answer isn't clear, simply state that you don't know rather than attempting to guess.

Background Information: {context}
User's Query: {question}

Respond only with a useful and accurate answer below.
Helpful answer:
"""

# Email credentials and configuration
DOCTOR_EMAIL = "haritha.appaswamy@gmail.com"  # Replace with the doctor's email
SENDER_EMAIL = "dharpadma2004@gmail.com"  # Replace with your email
SENDER_PASSWORD = "dzyd qery dhud ovro"  # Replace with your generated App Password
IMAP_SERVER = "imap.gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Global variable to store the query timestamp
query_timestamp = None

# --- REPLACED KEYWORD-BASED CLASSIFIER WITH ML MODEL ---
def classify_question(question_text):
    """
    Loads the fine-tuned DistilBERT model and classifies a single question.
    Returns 'CRITICAL' or 'NON_CRITICAL'.
    """
    model_path = "./pregnancy_pal_classifier"
    
    # Check if the trained model directory exists
    if not os.path.exists(model_path):
        print(f"CRITICAL ERROR: Model not found at {model_path}")
        print("Please make sure the 'pregnancy_pal_classifier' folder is in the same directory as this script.")
        # Default to critical as a safety measure if the model is missing
        return "CRITICAL"

    try:
        # Load the saved model and tokenizer
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)

        # Prepare the input
        inputs = tokenizer(question_text, return_tensors="pt", truncation=True, padding=True, max_length=128)

        # Make a prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = torch.argmax(logits, dim=1).item()

        # Return the human-readable label
        return model.config.id2label[predicted_class_id]
    except Exception as e:
        print(f"An error occurred during classification: {e}")
        # Default to critical on any error for safety
        return "CRITICAL"
# -----------------------------------------------------------


# Function to notify the doctor via email
def notify_doctor(question):
    subject = "Critical Query Received from Pregnancy Pal"
    body = f"A critical query has been received from a user:\n\n'{question}'\n\nPlease reply to this email to respond."

    # Create the email message
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = DOCTOR_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Connect to the SMTP server and send the email
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, DOCTOR_EMAIL, msg.as_string())
        server.quit()
        print("Email alert sent to doctor successfully.")
    except Exception as e:
        print(f"Failed to send email. Error: {e}")

# Function to fetch the doctor's reply from the email
def fetch_doctor_reply(query_time):
    try:
        # Connect to the IMAP server
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(SENDER_EMAIL, SENDER_PASSWORD)
        mail.select("inbox")

        # Search for emails from the doctor
        status, messages = mail.search(None, f'FROM "{DOCTOR_EMAIL}"')
        if status != "OK" or not messages[0]:
            return None

        # Process emails from latest to oldest
        for email_id in reversed(messages[0].split()):
            status, data = mail.fetch(email_id, "(RFC822)")
            if status != "OK" or not data:
                continue

            # Parse the raw email
            raw_email = data[0][1]
            msg = email.message_from_bytes(raw_email)

            # Parse email timestamp
            email_date = msg["Date"]
            email_timestamp = email.utils.parsedate_to_datetime(email_date)

            # Check if the email was sent after the query timestamp
            if email_timestamp > query_time:
                # Extract and return the email body
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body = part.get_payload(decode=True).decode(errors="ignore").strip()
                            return body
                else:
                    body = msg.get_payload(decode=True).decode(errors="ignore").strip()
                    return body
        return None
    except Exception as e:
        print(f"Failed to fetch doctor's reply. Error: {e}")
        return None

# Wait for the doctor's reply and update the chatbot
async def wait_for_doctor_reply(query_time):
    await cl.Message(content="Waiting for the doctor's reply...").send()
    for attempt in range(10):  # Poll for 10 attempts
        print(f"Polling attempt {attempt + 1}...")
        reply = fetch_doctor_reply(query_time)
        if reply:
            try:
                # Notify the user with the doctor's reply
                await cl.Message(content=f"**Doctor's Reply:**\n{reply.strip()}").send()
                return  # Exit after successfully sending the reply
            except Exception as e:
                print(f"Failed to send the doctor's reply in chatbot: {e}")
        else:
            # Send a progress update after each unsuccessful attempt
            if attempt < 9: # Don't send on the last attempt
                await cl.Message(content=f"Polling attempt {attempt + 1}: No reply yet. Retrying in 30 seconds...").send()
        await asyncio.sleep(30)  # Use await asyncio.sleep for non-blocking delay

    # Notify the user after 10 failed attempts
    await cl.Message(content="No reply from the doctor yet. Please check back later.").send()

# Prompt configuration
def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt}
                                           )
    return qa_chain

def load_llm():
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Pregnancy Pal. What is your query?"
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    global query_timestamp
    
    # Use the ML model to classify the question
    query_type = classify_question(message.content)
    print(f"Query classified as: {query_type}")

    if query_type == "NON_CRITICAL":
        chain = cl.user_session.get("chain") 
        cb = cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
        )
        cb.answer_reached = True
        res = await chain.ainvoke(message.content, callbacks=[cb])
        answer = res["result"]
        sources = res["source_documents"]

        formatted_answer = f"Answer:\n{answer.strip()}\n\n"
        if sources:
            formatted_answer += "Sources:\n"
            for idx, doc in enumerate(sources, 1):
                source_info = f"Source {idx}: {doc.metadata['source']}, Page: {doc.metadata['page']}"
                formatted_answer += f"{source_info}\n"
        else:
            formatted_answer += "\n*No sources found*"

        await cl.Message(content=formatted_answer).send()
    else: # This handles "CRITICAL"
        query_timestamp = datetime.datetime.now(datetime.timezone.utc)  # Record the query timestamp
        await cl.Message(content="This seems to be a critical question. A doctor will get back to you shortly.").send()
        notify_doctor(message.content)  # Notify the doctor

        # Wait for a reply from the doctor
        await wait_for_doctor_reply(query_timestamp)
