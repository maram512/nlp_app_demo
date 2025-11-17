import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import re

st.title("🔤 NLP App: Correction + Next Sentence Prediction")

user_input = st.text_input("Enter a sentence:")

# ------------------- Lazy Loading Models -------------------
@st.cache_resource
def load_correction_model():
    model = AutoModelForSeq2SeqLM.from_pretrained("maram5/correction_model")
    tokenizer = AutoTokenizer.from_pretrained("maram5/correction_model")
    return model, tokenizer

@st.cache_resource
def load_next_word_model():

    """
    Load GPT-2 fine-tuned model from correct folder inside HuggingFace repo
    """
    

    model = GPT2LMHeadModel.from_pretrained("maram5/next_word_model",
    subfolder="gpt2-finetuned-final")
    tokenizer = GPT2Tokenizer.from_pretrained("maram5/next_word_model",
    subfolder="gpt2-finetuned-final")

    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# ------------------- Buttons -------------------

if st.button("Correct Sentence"):
    with st.spinner("Loading correction model..."):
        corr_model, corr_tokenizer = load_correction_model()
    with st.spinner("Generating corrected sentence..."):
        inputs = corr_tokenizer(user_input, return_tensors="pt")
        outputs = corr_model.generate(**inputs)
        corrected = corr_tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.success(f"Corrected: {corrected}")

if st.button("Predict Next Sentence"):
    with st.spinner("Loading next sentence model..."):
        next_model, next_tokenizer = load_next_word_model()
    with st.spinner("Generating next sentence..."):
        prompt = "Continue the sentence: " + user_input
        inputs = next_tokenizer(prompt, return_tensors="pt")
        outputs = next_model.generate(
            inputs.input_ids,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            pad_token_id=next_tokenizer.eos_token_id
        )
        full_text = next_tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_part = full_text.replace(prompt, "").strip()
        first_sentence = re.split(r"[.!?]", generated_part)[0]
    st.info(f"Next sentence: {first_sentence.strip()}")

if st.button("Correct + Predict"):
    # Correction
    with st.spinner("Loading correction model..."):
        corr_model, corr_tokenizer = load_correction_model()
    with st.spinner("Generating corrected sentence..."):
        inputs = corr_tokenizer(user_input, return_tensors="pt")
        outputs = corr_model.generate(**inputs)
        corrected = corr_tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.success(f"Corrected: {corrected}")

    # Next Sentence Prediction
    with st.spinner("Loading next sentence model..."):
        next_model, next_tokenizer = load_next_word_model()
    with st.spinner("Generating next sentence..."):
        prompt = "Continue the sentence: " + corrected
        inputs = next_tokenizer(prompt, return_tensors="pt")
        outputs = next_model.generate(
            inputs.input_ids,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            pad_token_id=next_tokenizer.eos_token_id
        )
        full_text = next_tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_part = full_text.replace(prompt, "").strip()
        first_sentence = re.split(r"[.!?]", generated_part)[0]
    st.info(f"Next sentence: {first_sentence.strip()}")




