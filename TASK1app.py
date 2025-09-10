import streamlit as st
from transformers import pipeline

st.title("AG News Topic Classifier (BERT)")
classifier = pipeline("text-classification",
                      model="models/agnews-bert",
                      tokenizer="models/agnews-bert",
                      return_all_scores=True)

text = st.text_input("Enter a news headline:")
if st.button("Predict"):
    if not text:
        st.warning("Please type something first!")
    else:
        preds = classifier(text)[0]
        preds_sorted = sorted(preds, key=lambda x: x['score'], reverse=True)
        for p in preds_sorted:
            st.write(f"{p['label']}: {p['score']:.3f}")
