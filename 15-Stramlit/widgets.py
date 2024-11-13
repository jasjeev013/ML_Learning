import streamlit as st
import pandas as pd

st.title("Hello Streamlit")

name = st.text_input("Enter your name")
age = st.slider("Enter your age", 0, 100, 25)
if name:
    st.write(f"Hello {name}")