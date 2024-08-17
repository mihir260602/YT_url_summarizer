import streamlit as st
import nltk
import validators
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Streamlit App Configuration
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")

# Custom CSS
st.markdown("""
    <style>
    body {
        background-color: black;
        color: beige;
        font-family: Arial, sans-serif;
    }
    .stButton > button {
        background-color: black;
        color: beige;
        border: 1px solid beige;
        padding: 10px 20px;
        border-radius: 5px;
        transition: background-color 0.3s, color 0.3s;
    }
    .stButton > button:hover {
        background-color: darkred;
        color: beige;
    }
    .stTextInput > div > div > input {
        color: black;
        background-color: beige;
        border: 1px solid beige;
        padding: 10px;
        border-radius: 5px;
    }
    .stTextInput > div > label {
        color: beige;
    }
    .stMarkdown, .stTitle, .stHeader, .stSubheader {
        color: beige;
    }
    .stApp {
        background-color: black;
    }
    .stResult {
        color: black;
        background-color: beige;
        border-radius: 5px;
        padding: 10px;
        border: 1px solid beige;
        margin-top: 10px;
        width: 150%;
        box-sizing: border-box;
    }
    .stResult > div {
        background-color: beige;
        color: black;
    }
    .stText, .stTextInput, .stMarkdown {
        color: beige;
    }
    .stTextInput > div > div > input {
        color: black;
        background-color: beige;
        border: 1px solid beige;
    }
    .stMarkdown {
        color: beige;
    }
    .url-history {
        display: flex;
        flex-direction: column;
        margin-top: 20px;
    }
    .url-entry {
        display: flex;
        justify-content: space-between;
        padding: 10px;
        border-bottom: 1px solid beige;
    }
    .url-summary {
        flex: 2;
        background-color: beige;
        color: black;
        border-radius: 5px;
        padding: 10px;
        margin-right: 10px;
    }
    .url-link {
        flex: 1;
        color: beige;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for API key, URL history, and summaries
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""
if "url_history" not in st.session_state:
    st.session_state.url_history = []

# Page to enter Groq API Key
if not st.session_state.groq_api_key:
    st.title("Enter Groq API Key")
    st.session_state.groq_api_key = st.text_input("Groq API Key", value="", type="password")

    if st.button("Submit"):
        if st.session_state.groq_api_key.strip():
            st.experimental_rerun()
        else:
            st.error("Please provide the Groq API key to proceed.")
else:
    # Main functionality page after API key is provided
    st.title("🦜 LangChain: Summarize Text From YT or Website")
    st.subheader('Summarize URL')

    # Input field for URL
    generic_url = st.text_input("URL", label_visibility="collapsed")

    # Summarize and Clear buttons
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("Summarize the Content from YT or Website"):
            # Validate all the inputs
            if not generic_url.strip():
                st.error("Please provide the URL to get started")
            elif not validators.url(generic_url):
                st.error("Please enter a valid URL. It can be a YT video URL or website URL")
            else:
                try:
                    st.write(f"Processing URL: {generic_url}")
                    with st.spinner("Waiting..."):
                        # Add the current URL to history
                        st.session_state.url_history.append({"url": generic_url, "summary": ""})

                        # Convert shortened YouTube URL if needed
                        def convert_youtube_short_url(url):
                            if "youtu.be" in url:
                                video_id = url.split('/')[-1].split('?')[0]
                                return f"https://www.youtube.com/watch?v={video_id}"
                            return url

                        generic_url = convert_youtube_short_url(generic_url)

                        # Loading the website or YT video data
                        if "youtube.com" in generic_url:
                            st.write("Detected YouTube URL. Attempting to load video...")
                            loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                        else:
                            st.write("Detected website URL. Attempting to load...")
                            loader = UnstructuredURLLoader(
                                urls=[generic_url],
                                ssl_verify=False,
                                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                            )
                        docs = loader.load()

                        # Check what was loaded
                        if not docs or not docs[0].page_content.strip():
                            st.error("Unable to retrieve content from the provided URL.")
                        else:
                            st.write(f"Retrieved {len(docs)} documents.")
                            # Chain for Summarization
                            llm = ChatGroq(model="Gemma-7b-It", groq_api_key=st.session_state.groq_api_key)
                            prompt_template = """
                            Provide a summary of the following content in 300 words:
                            Content: {text}
                            """
                            prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
                            chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                            output_summary = chain.run(docs)

                            # Update session state with the summary
                            st.session_state.url_history[-1]["summary"] = output_summary
                            st.markdown(f"<div class='stResult'>{output_summary}</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.exception(f"Exception: {e}")

    with col2:
        if st.button("Clear"):
            st.session_state.url_history = []
            st.experimental_rerun()

    # Display URL History with summaries and URLs side by side
    if st.session_state.url_history:
        st.write("### URL History")
        url_history_container = st.container()
        with url_history_container:
            st.markdown('<div class="url-history">', unsafe_allow_html=True)
            for item in st.session_state.url_history:
                st.markdown(f"""
                    <div class="url-entry">
                        <div class="url-summary">{item['summary']}</div>
                        <div class="url-link"><a href="{item['url']}" target="_blank">{item['url']}</a></div>
                    </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
