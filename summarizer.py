import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st

st.markdown("""
    <style>
        body {
            background-color: yellow;
        }

        .css-1d391kg {
            background-color: yellow;
        }

        .stApp {
            background-color: yellow;
        }
            
        .highlight-text {
            background-color: beige;  /* Yellow background color */
            padding: 10px;
            border-radius: 15px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Text Summarizer")

text = st.text_input("You:", placeholder="Type your message here...", key="user_input_key")

def summarization(txt):
    try:
        sentences = nltk.sent_tokenize(text)

        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(sentences)

        cos_sim = cosine_similarity(X)

        sentence_scores = np.sum(cos_sim, axis=1)

        sorted_score_ind = sentence_scores.argsort()[-3:]
        top_sentences = [sentences[i] for i in sorted_score_ind]
        top_sentences=top_sentences[::-1]

        return ' '.join(top_sentences)
    except:
        print("No text can be processed")

summary = summarization(text)

if text:
    st.write("\n")
    st.markdown("<font size='6'>**SUMMARY**</font>", unsafe_allow_html=True)
    st.write("\n"*5)
    st.markdown(f'<p class="highlight-text"><font size="4"><b>{summary}</b></font></p>', unsafe_allow_html=True)

# if text:
#     st.text_area("Summary :", value=summary, height=200)

