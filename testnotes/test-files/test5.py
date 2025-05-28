import os

doc_path = r"testnotes\testdata\encryption.pdf"
abs_path = os.path.abspath(doc_path)
print("Looking for PDF at:", abs_path)
print("File exists:", os.path.exists(abs_path))
