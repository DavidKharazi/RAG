# разделение по слову Идентификатор.

def split_docs_to_chunks_by_keyword(documents: dict, file_types: List[str], keyword="—BLOCK—"):
    all_chunks = []

    def split_by_keyword(text, keyword):
        # Разделяем текст по ключевому слову и сохраняем ключевое слово в начале каждого чанка
        parts = re.split(f"({keyword})", text)
        chunks = [parts[i] + parts[i + 1] for i in range(1, len(parts) - 1, 2)]
        if parts[0]:
            chunks.insert(0, parts[0])
        if len(parts) % 2 == 0:
            chunks.append(parts[-1])
        return chunks

    if 'txt' in file_types and documents['txt'] is not None:
        for doc in documents['txt']:
            chunks = split_by_keyword(doc.page_content, keyword)
            for chunk in chunks:
                all_chunks.append(Document(source=doc.source, page_content=chunk, metadata=doc.metadata))

    if 'json' in file_types and documents['json'] is not None:
        for idx, doc in enumerate(documents['json']):
            text = json.dumps(doc, ensure_ascii=False)
            chunks = split_by_keyword(text, keyword)
            for chunk in chunks:
                all_chunks.append(Document(source=documents['json_metadata'][idx]['source'], page_content=chunk))

    if 'docx' in file_types and documents['docx'] is not None:
        for doc in documents['docx']:
            chunks = split_by_keyword(doc.page_content, keyword)
            for chunk in chunks:
                all_chunks.append(Document(source=doc.source, page_content=chunk, metadata=doc.metadata))

    return all_chunks