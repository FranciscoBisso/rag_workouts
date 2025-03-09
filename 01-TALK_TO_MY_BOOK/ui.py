"""STREAMLIT'S UI"""

import streamlit as st
from typing import List
from llm_handler import get_response_from_llm

# PAGE'S CONFIG
st.set_page_config(
    page_title="Colegal",
    page_icon=":scroll:",
    layout="centered",
)

# PAGE'S TITLE
st.markdown(
    """# :blue[Co]legal
##### :gray[_Tu asistente virtual en derecho procesal civil y comercial argentino._]"""
)


# INITIALIZE CHAT'S HISTORY
if "messages" not in st.session_state:
    st.session_state.messages = []


# DISPLAY CHAT MESSAGES FROM CHAT'S HISTORY
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar="🧑‍⚖️"):
            st.markdown(message["content"])


# USER INPUT
if user_input := st.chat_input("¿En qué puedo ayudarte?"):

    # DISPLAY USER MESSAGE IN CHAT MESSAGE CONTAINER
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    # ADD USER MESSAGE TO CHAT HISTORY
    st.session_state.messages.append({"role": "user", "content": user_input})

    # PLACEHOLDER FOR ASSISTANT'S RESPONSE
    with st.chat_message("assistant", avatar="🧑‍⚖️"):
        response_placeholder = st.empty()
        full_response: List = []

        # SHOW SPINNER WHILE STREAMING RESPONSE
        with st.spinner(":gray[_Generando respuesta..._]"):
            for chunk in get_response_from_llm(user_input):
                # ACCUMULATE STREAMED CHUNKS
                full_response.append(str(chunk.content))
                response_placeholder.markdown("".join(full_response))

        # ADD ASSISTANT'S RESPONSE TO CHAT HISTORY
        st.session_state.messages.append(
            {"role": "assistant", "content": "".join(full_response)}
        )
