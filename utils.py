from typing import List

from io import BytesIO
import re

from pypdf import PdfReader

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def parse_pdf(file: BytesIO) -> List[str]:
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

def text_split(text: str ) -> List[Document]:
    """Converts a string or list of strings to a list of Documents (chunks of fixed size)
    with metadata."""

    if isinstance(text, str):
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    doc_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2048,
    chunk_overlap  = 20,
    #separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    length_function = len,
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
    print(doc_chunks)
    return doc_chunks
