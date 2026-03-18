import zipfile
import xml.etree.ElementTree as ET

def extract_text_from_docx(docx_path):
    with zipfile.ZipFile(docx_path) as z:
        xml_content = z.read('word/document.xml')
        tree = ET.fromstring(xml_content)
        
        # We want to dump all text to see where the equations might be.
        # Or look for m:oMath elements specifically.
        # Let's just dump all text nodes and see it.
        namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
                      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}
        
        texts = []
        for elem in tree.iter():
            if elem.text:
                texts.append(elem.text)
        print(f"--- {docx_path} ---")
        full_text = "".join(texts)
        if "W" in full_text:
            print("Contains W")
        
        # Let's specifically look for math elements if any
        math_elements = tree.findall('.//m:oMath', namespaces)
        if math_elements:
            print(f"Found {len(math_elements)} math elements.")
            for m in math_elements:
                m_texts = [e.text for e in m.iter() if e.text]
                print("Math:", "".join(m_texts))
        
        # Let's just print a window around 'Hybrid Adapter' or 'EGMP'
        for i, t in enumerate(texts):
            if 'expressed mathematically using LaTeX' in t:
                start = max(0, i-5)
                end = min(len(texts), i+20)
                print("Context:", "".join(texts[start:end]))
            if 'manifold' in t and 'projected into a low' in t:
                start = max(0, i-5)
                end = min(len(texts), i+20)
                print("Context:", "".join(texts[start:end]))
                
extract_text_from_docx('LLM Optimization MRED on MacBook.docx')
extract_text_from_docx('Neural Gravity_ M3 LLM Breakthrough.docx')
