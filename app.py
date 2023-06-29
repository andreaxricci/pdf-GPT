import base64

import streamlit as st
from streamlit_chat import message
from openai.error import OpenAIError

from utils import text_split, parse_pdf, get_embeddings, get_sources, get_answer, get_condensed_question

st.set_page_config(page_title="pdf-GPT", page_icon="📖", layout="wide")
st.header("pdf-GPT")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def clear_submit():
    st.session_state["submit"] = False


def displayPDF(upl_file):
    # Read file as bytes:
    bytes_data = upl_file.getvalue()

    # Convert to utf-8
    base64_pdf = base64.b64encode(bytes_data).decode("utf-8")

    # Embed PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'

    # Display file
    st.markdown(pdf_display, unsafe_allow_html=True)


with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key", key="file_qa_api_key", type="password"
    )

    model_name = st.radio(
        "Select the model",
        ('gpt-3.5-turbo', 'text-davinci-003', 'gpt-4'))

    uploaded_file = st.file_uploader(
        "Upload file",
        type=["pdf"],
        help="Only PDF files are supported",
        on_change=clear_submit,
    )


col1, col2 = st.columns(spec=[2, 1], gap="small")


if uploaded_file:
    with col1:
        displayPDF(uploaded_file)

    with col2:
        pdf_document = parse_pdf(uploaded_file)
        text = text_split(pdf_document)
        try:
            vs = get_embeddings(text,openai_api_key)
            st.session_state["api_key_configured"] = True
        except OpenAIError as e:
            st.error(e._message)

        question = st.text_input(
            "Ask something about the file",
            placeholder="Can you give me a short summary?",
            disabled=not uploaded_file,
            on_change=clear_submit,
        )

        if uploaded_file and question and not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")

        if uploaded_file and question and openai_api_key:

            try:
                chat_history_tuples = [(st.session_state['past'][i], st.session_state['generated'][i]) for i in range(len(st.session_state['generated']))]
                condensed_question = get_condensed_question(question, chat_history_tuples, model_name, openai_api_key)
    
                sources = get_sources(vs, condensed_question)
                answer = get_answer(sources, condensed_question, openai_api_key)

                st.session_state.generated.append(answer["output_text"]) 
                st.session_state.past.append(question)

                #for source in sources:
                #    st.markdown(source.page_content)
                #    st.markdown(source.metadata["source"])
                #    st.markdown("---")

            except OpenAIError as e:
                st.error(e._message)

            if st.session_state['generated']:
                for i in range(len(st.session_state['generated'])-1, -1, -1):
                    message(st.session_state["generated"][i], avatar_style="initials", seed="B", key=str(i))
                    message(st.session_state['past'][i], is_user=True, avatar_style="initials", seed="A", key=str(i) + '_user')

