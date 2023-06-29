from typing import Any, Dict, List

from io import BytesIO
import re

from pypdf import PdfReader

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.conversational_retrieval.base import _get_chat_history

from langchain.vectorstores import VectorStore
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI

from prompts import STUFF_PROMPT

def parse_pdf(file: BytesIO) -> List[str]:
    ''' PDF parser '''
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)

        output.append(text)

    return output


def text_split(text: str) -> List[Document]:
    """Converts a string to a list of Documents (chunks of fixed size)
    and returns this along with the following metadata:
        - page number
        - chunk number
        - source (concatenation of page & chunk number)
    """

    if isinstance(text, str):
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    doc_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048,
        chunk_overlap=20,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len,
    )

    for doc in page_docs:
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)

    return doc_chunks


def get_embeddings(docs: List[Document], openai_api_key: str) -> VectorStore:
    vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings(openai_api_key=openai_api_key), ids=[doc.metadata["source"] for doc in docs])

    return vectorstore


def get_sources(vectorstore: VectorStore, query: str) -> List[Document]:
    # Search for similar chunks
    docs = vectorstore.similarity_search(query, k=5)

    return docs


def get_condensed_question(user_input: str, chat_history_tuples, model_name: str, openai_api_key: str):
    llm=ChatOpenAI(model_name=model_name, openai_api_key=openai_api_key)
    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)

    condensed_question = question_generator.predict(question=user_input, chat_history=_get_chat_history(chat_history_tuples))

    return condensed_question


def get_answer(docs: List[Document], query: str, openai_api_key: str) -> Dict[str, Any]:
    """Gets an answer to a question from a list of Documents."""

    # Get the answer

    chain = load_qa_with_sources_chain(
        OpenAI(
            temperature=0, openai_api_key=openai_api_key
        ),  # type: ignore
        chain_type="stuff",
        prompt=STUFF_PROMPT,
    )

    answer = chain(
        {"input_documents": docs, "question": query}, return_only_outputs=True
    )

    return answer

