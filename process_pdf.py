import PyPDF2
import re

def re_compose_pdf(text:str) -> str:
    line_count = text.count('\n') + 1
    lines = text.split('\n')
    text = '\n'.join(lines[5:line_count-15])
    return re.sub(r'\.{10,}\d+', '', text)
    
filename = "dataset/doc/cwe_latest.pdf"
pdf_file = open(filename, 'rb')

reader = PyPDF2.PdfReader(pdf_file)
text = ""

for i in range(2, 18):
    
    page_num = i
    page = reader.pages[page_num]
    text += page.extract_text()

text = re_compose_pdf(text)

print('--------------------------------------------------')
print(text)

with open("dataset/doc/cwe_latest.md", "w") as f:
    f.write(text)

pdf_file.close()