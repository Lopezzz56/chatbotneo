import fitz 
import uuid

def process_pdf(uploaded_file):
    doc_id = f"pdf-{uuid.uuid4().hex[:8]}"
    content = uploaded_file.read()
    pdf = fitz.open(stream=content, filetype="pdf")
    
    chunks = []
    for page_no in range(len(pdf)):
        text = pdf[page_no].get_text()
        if not text.strip(): continue
        
        
        size, stride = 800, 200
        for i in range(0, len(text), size - stride):
            chunk = text[i:i + size]
            chunks.append({
                "text": chunk,
                "page": page_no + 1,
                "char_start": i,
                "char_end": i + len(chunk)
            })
    return doc_id, chunks