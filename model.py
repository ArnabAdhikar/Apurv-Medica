from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
# only use retrival chain for this project
from langchain.chains import RetrievalQA
import chainlit as cl
import pickle

DB_FAISS_PATH = 'vectorstore/db_faiss'

# Assuming you trust this pickle file:
with open('vectorstore/db_faiss/index.pkl', 'rb') as file:
    data = pickle.load(file)  # Deserializes the file content



# prompt template
# context=> from LLM
# questions=> End users
custom_prompt_template = """Use the following pieces of information to answer the user's question and or add up a long new information also that are relevent to the context.
If the answer is in the context but you don't have the information, provide the user with most relevent answer detailing your logic.
If the question is out of the context, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# setup the prompt
def set_custom_prompt():
    # Prompt template for QA retrieval for each vectorstore
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt


# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt}
                                           )
    return qa_chain


# Loading LLM model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=1050,
        temperature=0.5
    )
    return llm

# QA Model Function
def qa_bot():
    # use our embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cuda'})
    # load our database here
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    # combine the QA chain
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa


# output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response


# chainlit code(documentation)
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hey, Welcome to Apurv Medica. Let's get the question for today?"
    await msg.update()

    cl.user_session.set("chain", chain)

# get the result
@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"
    await cl.Message(content=answer).send()


# Remove cl.run() call as Chainlit automatically starts server
# chainlit run model.py --port 8001