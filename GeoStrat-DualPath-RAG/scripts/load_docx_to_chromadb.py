"""
将福建地质志.docx存入ChromaDB（直接使用chromadb，绕过langchain包装器）
"""
import os, sys, re, time, uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import docx
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


def extract_paragraphs(docx_path):
    doc = docx.Document(docx_path)
    paragraphs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text: continue
        if re.match(r'^\s*\d{1,4}\s*$', text): continue
        paragraphs.append({'text': text, 'length': len(text)})
    return paragraphs


def merge_and_chunk(paragraphs, min_chars=200, max_chars=1200):
    result = []
    buffer = ""
    for p in paragraphs:
        t = p['text']
        if len(t) >= max_chars:
            if buffer:
                result.append(buffer.strip())
                buffer = ""
            sentences = re.split(r'(?<=[。！？])', t)
            chunk = ""
            for s in sentences:
                if len(chunk) + len(s) > max_chars and chunk:
                    result.append(chunk.strip())
                    chunk = s
                else:
                    chunk += s
            if chunk.strip():
                result.append(chunk.strip())
        elif len(t) < min_chars:
            buffer = (buffer + "\n" + t).strip() if buffer else t
            if len(buffer) >= min_chars:
                result.append(buffer.strip())
                buffer = ""
        else:
            if buffer:
                result.append(buffer.strip())
                buffer = ""
            result.append(t)
    if buffer.strip():
        result.append(buffer.strip())
    return result


def main():
    docx_path = r'E:\地质文本空间化平台\福建地质志.docx'

    print('Loading docx...')
    paragraphs = extract_paragraphs(docx_path)
    print(f'Extracted {len(paragraphs)} paragraphs')

    print('Merging and chunking...')
    chunks = merge_and_chunk(paragraphs)
    valid = [c for c in chunks if 150 <= len(c) <= 2000]
    print(f'Valid chunks: {len(valid)}')

    print('Loading embedding model...')
    model = SentenceTransformer('shibing624/text2vec-base-chinese')

    print('Connecting to ChromaDB...')
    client = chromadb.PersistentClient(
        path='./chroma_db',
        settings=Settings(anonymized_telemetry=False)
    )

    # 使用langchain默认集合名，与rag_core.py保持一致
    try:
        client.delete_collection('langchain')
        print('Deleted existing collection')
    except:
        pass

    col = client.create_collection('langchain')
    print(f'Collection created')

    # 分批embed+add
    batch_size = 40
    for i in range(0, len(valid), batch_size):
        batch = valid[i:i + batch_size]
        ids = [str(uuid.uuid4()) for _ in batch]
        try:
            embeddings = model.encode(batch).tolist()
            col.add(ids=ids, documents=batch, embeddings=embeddings)
        except Exception as e:
            print(f'  Batch {i} encode error: {e}')
            # Fallback: let chromadb auto-embed
            col.add(ids=ids, documents=batch)

        pct = min(i + batch_size, len(valid))
        if pct % 200 == 0 or pct == len(valid):
            print(f'  [{pct}/{len(valid)}] stored')
        time.sleep(0.3)

    print(f'\n=== Done ===')
    print(f'Total chunks: {len(valid)}')
    print(f'Collection name: langchain')
    print(f'Embedding model: shibing624/text2vec-base-chinese')


if __name__ == '__main__':
    main()
