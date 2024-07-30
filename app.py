import streamlit as st
from huggingface_hub import login
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone as langchain_pinecone
import transformers
from torch import bfloat16

def reset_chat(): # Resets the chat history being displayed
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask a question related to chemistry"}
    ]

def generateResponse(model, question): # Calls the model with a prompt to get response
    response = model(question)
    print(f"Prompt: {question}")
    print(f"Response: {response['result']}")
    print("\n")
    print(f"Source Documents: {response['source_documents']}")
    return response['result']


st.header("Ask Questions About the JCESR Papers!")
device = 'cpu'
hf_k = "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"
login(token=hf_k)

# Initialize Pinecone

if "messages" not in st.session_state.keys():  # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask a question about the JCESR papers"}
    ]

@st.cache_resource(show_spinner=False) # Load the model and configure the pipeline for answer generation
def load_data():
    with st.spinner(text="Configuring Model, please wait"):
        index_name = "jcsr-test"
        pc = Pinecone(api_key="7284d8ea-51a2-47d0-91f0-eb00ceafd17d", environment="us-west1-gcp")
        if index_name not in pc.list_indexes().names():
            pc.create_index(
            name = index_name,
            dimension = 384,
            metric = 'cosine',
            spec = ServerlessSpec(cloud = 'aws', region = 'us-east-1')
            )
        index = pc.Index(index_name)
        #tokenizer = AutoTokenizer.from_pretrained(model_name)
        #model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

        embed_model = HuggingFaceEmbeddings(
            model_name=embed_model_id,
            model_kwargs={'device': device},
            encode_kwargs={'device': device, 'batch_size': 32}
        )

        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        # set quantization configuration to load large model with less GPU memory
        # this requires the `bitsandbytes` library

        # begin initializing HF items, need auth token for these
        hf_auth = "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"
        model_config = transformers.AutoConfig.from_pretrained(
            model_id,
            use_auth_token=hf_auth
        )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=model_config,
            use_auth_token=hf_auth,
        )

        model.eval()
        print(f"Model loaded on {device}")


        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id,
            use_auth_token=hf_auth
        )

        generate_text = transformers.pipeline(
            model=model, tokenizer=tokenizer,
            return_full_text=False,  # langchain expects the full text
            task='text-generation',
            # we pass model parameters here too  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            max_new_tokens=512,  # max number of tokens to generate in the output
            repetition_penalty=1.1  # without this output begins repeating
        )

        llm = HuggingFacePipeline(pipeline=generate_text)

        text_field = 'text'  # field in metadata that contains text content

        vectorstore = langchain_pinecone(
            index, embed_model.embed_query, text_field
        )

        rag_pipeline = RetrievalQA.from_chain_type(
            llm=llm, chain_type='stuff',
            retriever=vectorstore.as_retriever(),
            return_source_documents=True # Returns the top source documents as part of the answer dictionary, if false it will not
        )

    return rag_pipeline

pipeline = load_data()

st.sidebar.button("Reset Chat", on_click=reset_chat, use_container_width=True)

if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

if "messages" in st.session_state.keys() and st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generateResponse(pipeline, prompt)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)
            st.write(response) 
