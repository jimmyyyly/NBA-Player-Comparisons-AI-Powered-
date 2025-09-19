import streamlit as st
import pandas as pd

st.title("Simple Button Test")

# Initialize session state
if 'clicked' not in st.session_state:
    st.session_state.clicked = False
if 'data' not in st.session_state:
    st.session_state.data = None

st.write(f"Button clicked: {st.session_state.clicked}")
st.write(f"Data: {st.session_state.data}")

if st.button("Click Me"):
    st.write("Button was clicked!")
    st.session_state.clicked = True
    st.session_state.data = "Hello World"
    st.rerun()

if st.session_state.data:
    st.success(f"Data loaded: {st.session_state.data}")
    
    if st.button("Reset"):
        st.session_state.clicked = False
        st.session_state.data = None
        st.rerun()
