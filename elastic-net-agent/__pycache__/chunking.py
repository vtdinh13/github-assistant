def sliding_window(seq, size, step):
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")

    n = len(seq)
    result = []
    for i in range(0, n, step):
        chunk = seq[i:i+size]
        result.append({'start': i, 'chunk': chunk})
        if i + size >= n:
            break

    return result


def create_chunks(repo_doc:list):
    doc_chunks = []

    for doc in repo_doc:
        doc_copy = doc.copy()
        doc_content = doc_copy.pop('content')
        chunks = sliding_window(doc_content, 2000, 1000)
        for chunk in chunks:
            chunk.update(doc_copy)
        doc_chunks.extend(chunks)
    return doc_chunks