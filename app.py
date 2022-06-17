from OCR1 import *
from topic_modeling_app import *
import streamlit as st

text = ocr_pdf_extractor()
print("TEXTT",text)
st.write(text)
if text:
    if st.button("Run Topic modelling"):
        modeling(text)





# if __name__ == "__main__":
#     path = './text.txt'
#     input_text = read_text(path)
#     print(input_text)



    
