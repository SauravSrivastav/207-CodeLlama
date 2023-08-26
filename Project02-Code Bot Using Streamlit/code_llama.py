# Imagine this is like the start of a fun adventure with Code Llama!
# Code Llama is a friendly AI that helps with coding problems.
# We'll go through the adventure step-by-step, like chapters in a book.

# Step 1: Importing Friends

# These lines bring in Code Llama's friends (tools) to help with the adventure.

from langchain.llms import CTransformers  # Smart coding friend
from langchain.chains import LLMChain      # Another friend on the team
from langchain import PromptTemplate      # A note to be polite
import os                                 # Think of it as a map
import io                                 # For sending and receiving messages
import streamlit as st                    # A magical talking tool
import time                               # To keep track of time

# Step 2: Writing a Friendly Note

# Code Llama wants to be friendly, so it writes a note to explain what it does.
# It's like writing a nice message to a friend.

# This is Code Llama's friendly note:
custom_prompt_template = """
You are an AI Coding Assistant, and your task is to solve coding problems and return code snippets based on the given user's query. Below is the user's query.
Query: {query}

You just return the helpful code.
Helpful Answer:
"""

# Code Llama creates the note and gets ready to use it when talking to people.
def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template,
    input_variables=['query'])
    return prompt

# Step 3: Summoning the Coding Dragon

# Code Llama needs something super-smart, like a magical coding dragon.
# This dragon knows a lot about coding.

# This is how Code Llama summons the dragon:
def load_model():
    llm = CTransformers(
        model = "codellama-7b-instruct.ggmlv3.Q4_0.bin",  # The dragon's name
        model_type="llama",            # What kind of dragon it is
        max_new_tokens = 1096,         # How much it can write
        temperature = 0.2,            # How creative it is
        repetition_penalty = 1.13      # Avoiding too much repetition
    )

    return llm

# Step 4: Assembling the Team

# Now, let's put everything together, like assembling a team of superheroes!

# Code Llama gets its dragon ready and gives it the friendly note:
def chain_pipeline():
    llm = load_model()                # Getting our dragon ready
    qa_prompt = set_custom_prompt()   # Giving our team the friendly note
    qa_chain = LLMChain(             # Assembling the team
        prompt=qa_prompt,
        llm=llm
    )
    return qa_chain

# The team is ready now, and Code Llama can start helping with coding questions!
llmchain = chain_pipeline()

def bot(query):
    llm_response = llmchain.run({"query": query})
    return llm_response

# Step 5: Setting Up a Chat Room

# Code Llama sets up a friendly chat room, like a place to talk with friends:
st.title('Code Llama Demo 🦙')
chat_history = []
message = st.text_input("Enter your message 📝")

# Step 6: Responding to Questions

if st.button("Send 🚀"):
    bot_message = bot(message)        # Asking our team for answers
    chat_history.append((message, bot_message))
    for msg in chat_history:
        st.write(f"User: {msg[0]}")
        st.write(f"Bot: {msg[1]}")
