import streamlit as st
from generate import load_model, generate_text

st.set_page_config(
    page_title="MuseLSTM | Creative Text Generator",
    page_icon="✨",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg, #10111d 0%, #19172b 100%);
    color: #f5f2ff;
}
h1 { color: #c8adff; }
div.stButton > button {
    background: #8064d8;
    color: white;
    border: 0;
    border-radius: 10px;
    min-height: 2.8rem;
}
div.stButton > button:hover {
    background: #987ce9;
    color: white;
    border: 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    return load_model()

st.title("✨ MuseLSTM")
st.subheader("An AI-powered creative text generator")
st.write(
    "Give the model a starting phrase, choose its creativity level, "
    "and let an LSTM continue the text."
)
st.divider()

try:
    model, char_to_idx, idx_to_char = get_model()
    st.success("Trained LSTM model loaded.")

    seed = st.text_area(
        "Starting text",
        value="The stars",
        help="Use characters that appear in dataset/corpus.txt."
    )

    left, right = st.columns(2)
    with left:
        temperature = st.slider(
            "Temperature", 0.1, 1.5, 0.8, 0.1,
            help="Lower values are more predictable; higher values are more random."
        )
    with right:
        max_length = st.slider(
            "Maximum new characters", 50, 1000, 300, 50
        )

    if temperature < 0.5:
        st.caption("Creativity: focused and predictable")
    elif temperature < 1.0:
        st.caption("Creativity: balanced")
    else:
        st.caption("Creativity: experimental")

    if st.button("✨ Generate text", use_container_width=True):
        with st.spinner("MuseLSTM is writing..."):
            try:
                result = generate_text(
                    model, char_to_idx, idx_to_char,
                    seed, temperature, max_length
                )
                st.subheader("Your generated text")
                st.text_area("Generated output", value=result, height=260)
                st.download_button(
                    "Download generated text",
                    data=result,
                    file_name="muse_generated.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            except ValueError as error:
                st.warning(str(error))

except FileNotFoundError:
    st.error("Trained model files were not found. Run `python train.py` first.")
except Exception as error:
    st.error(f"Could not load the model: {error}")

st.divider()
st.caption("MuseLSTM · Built with Python, PyTorch and Streamlit")
