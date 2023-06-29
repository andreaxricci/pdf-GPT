import base64

import streamlit as st
from openai.error import OpenAIError

from utils import text_split, parse_pdf, get_embeddings, get_sources, get_answer


st.set_page_config(page_title="pdf-GPT", page_icon="📖", layout="wide")
st.header("pdf-GPT")


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
            with st.spinner("Indexing document... This may take a while⏳"):
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
                sources = get_sources(vs, question)
                answer = get_answer(sources, question, openai_api_key)
                #sources = get_sources(answer, sources)
                st.markdown("#### Answer")
                st.markdown(answer["output_text"].split("SOURCES: ")[0])

                #st.markdown("#### Sources")
                #for source in sources:
                #    st.markdown(source.page_content)
                #    st.markdown(source.metadata["source"])
                #    st.markdown("---")

            except OpenAIError as e:
                st.error(e._message)
