import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Function to translate text
def translate_text(text, source_lang='en', target_lang='fr'):
    try:
        # Prepare the model and tokenizer based on the source and target language
        model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

        # Tokenize the input text and perform translation
        tokenized_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated_tokens = model.generate(**tokenized_text)
        translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

        return translated_text[0]
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.title("AI-Powered Translation Service")

# Input text for translation
text = st.text_area("Input Text", "Translation software is any computer application that converts text from one language to another")

# Dropdown menus for selecting source and target languages
source_language = st.selectbox("Source Language", ['en', 'fr', 'de', 'es'], index=0)
target_language = st.selectbox("Target Language", ['en', 'fr', 'de', 'es'], index=1)

# Translate button
if st.button("Translate"):
    # Check if source and target languages are the same
    if source_language == target_language:
        st.warning("Source and target languages cannot be the same. Please select different languages.")
    else:
        # Perform translation
        translated_text = translate_text(text, source_language, target_language)
        st.text_area("Translated Text", translated_text, height=150)
