from langchain.vectorstores import Qdrant
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import qdrant_client
import os
import time
from langchain_community.llms import LlamaCpp
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
import torch
from qdrant_client.http.models import Distance, VectorParams
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, TensorDataset
from qdrant_client import QdrantClient

QUESTION_FILE = "question.txt"
ANSWER_FILE = "answer.txt"

def clear_file(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"{filepath} deleted.")

clear_file(QUESTION_FILE)
clear_file(ANSWER_FILE)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 64).to(x.device)
        c0 = torch.zeros(2, x.size(0), 64).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Model parameters
input_size = 50  # Same as max_features in TF-IDF
hidden_size = 64
num_layers = 2
output_size = 2

df = pd.read_csv('greetings.csv')
vectorizer = TfidfVectorizer(max_features=50)
X_tfidf = vectorizer.fit_transform(df['Question']).toarray()

model_loaded = LSTMClassifier(input_size, hidden_size, num_layers, output_size)
model_loaded.load_state_dict(torch.load("lstm_text_classifier.pth"))
model_loaded.eval()  # Set model to evaluation mode

def predict(text, model, vectorizer):
    model.eval()  # Set to evaluation mode
    # Convert text to TF-IDF vector
    text_vector = vectorizer.transform([text]).toarray()
    text_tensor = torch.tensor(np.expand_dims(text_vector, axis=1), dtype=torch.float32)

    # Get model prediction
    with torch.no_grad():
        output = model(text_tensor)
        predicted_label = torch.argmax(output, dim=1).item()

    return "Greeting" if predicted_label == 1 else "Not Greeting"

embedding_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
embedding_model = HuggingFaceEmbeddings(
    model_name=embedding_model_id,
    encode_kwargs={'batch_size': 16}
)

client = QdrantClient("localhost", port=6333)
# collection_name = 'pulkitcollection'
# client.recreate_collection(
#  collection_name=collection_name,
#  vectors_config=VectorParams(size=384, distance=Distance.COSINE),
# )
vectorstore = Qdrant(client=client,collection_name='pulkitcollection',embeddings=embedding_model)

llm = LlamaCpp(model_path = "phi-2.Q4_K_M.gguf",n_ctx=2048,stop =  ["<|endoftext|>", "</s>"], echo = False, temperature = 0)
llm2 = LlamaCpp(model_path = "phi-2.Q4_K_M.gguf",n_ctx=2048,stop =  ["<|endoftext|>", "</s>"], echo = False, temperature = 0.2)

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
retriever = vectorstore.as_retriever()
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)
qa_system_prompt = """You are an assistant for question-answering tasks.\
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
def fetchprompt(question):
    initial_prompt = """You are a friendly AI assistant that replies **only with a brief greeting** to user greetings.  
            Respond exactly as in the examples, with no extra text.  

            Examples:  

            User: Hi!  
            AI: Hello!  

            User: Hey, how are you?  
            AI: I'm good, thanks!  

            User: Good Morning!  
            AI: Good Morning!  

            User: Glad to see you!  
            AI: Same here!  

            Now, respond strictly to this:  
            User: {input}  
            AI:"""  

    final_prompt = initial_prompt.format(input= question)
    return final_prompt

def getFinalAnswer(ai_msg_1):
    import re

    text = ai_msg_1['answer']
    # Start after the first ': '
    colon_index = text.find(': ')
    if colon_index == -1:
        return text.strip()
    remaining_text = text[colon_index + 2:].lstrip()

    # Stop at 'Human:', 'User:', or 'Assistant:' if they appear
    stop_tokens = ['Human:', 'User:', 'Assistant:']
    earliest_stop = len(remaining_text)
    for token in stop_tokens:
        idx = remaining_text.find(token)
        if idx != -1 and idx < earliest_stop:
            earliest_stop = idx

    trimmed_text = remaining_text[:earliest_stop]

    return trimmed_text.strip()

chat_history = []
questions = []
responses = []
with open("chat_responses.txt", "w") as file:
    pass  # Do nothing, just create the file

def process_question_interactive(question):
    print(f"\nNew question received: {question}")

    print("Query:", question)
    if(question.lower() == "exit"):
        return None
    questions.append(question)
    predicted_label = predict(question, model_loaded, vectorizer)
    
    if(predicted_label == "Greeting"):
        print("This was a greeting")
        final_prompt = fetchprompt(question)
        print("Final prompt fetched successfully!")
        response = llm2(final_prompt).strip().split("\n")[0] 
        print(response)
        responses.append(response)
        # Append response to a text file
        with open("chat_responses.txt", "a", encoding="utf-8") as f:
            f.write(f"Q: {question}\nA: {response}\n\n")
        return response
    
    ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
    response = getFinalAnswer(ai_msg_1)
    print(ai_msg_1['answer'])
    responses.append(response)
    chat_history.extend([HumanMessage(content=question), response])
    # Append response to a text file
    with open("chat_responses.txt", "a", encoding="utf-8") as f:
        f.write(f"Q: {question}\nA: {response}\n\n")
    
    return response


def watch_file():
    last_question = ""
    while True:
        if os.path.exists(QUESTION_FILE):
            with open(QUESTION_FILE, "r") as f:
                question = f.read().strip()
                
                if question != "" and question != last_question:
                    print(f"New question detected: {question}")
                    
                    # Clear answer.txt before processing new question
                    clear_file(ANSWER_FILE)
                    print("answer.txt cleared")

                    # Get manual answer from user
                    answer = process_question_interactive(question)
                    if(answer == None):
                        with open(ANSWER_FILE, "w") as af:
                            af.write("Thank You! Have a nice day!")
                        clear_file(QUESTION_FILE)
                        return
                    # Write answer to answer.txt
                    with open(ANSWER_FILE, "w") as af:
                        af.write(answer)
                    print(f"Answer written to {ANSWER_FILE}")

                    # Update last_question and clear question.txt
                    last_question = question
                    clear_file(QUESTION_FILE)
                    print(f"{QUESTION_FILE} cleared")

        time.sleep(1)  # check every second

print("Backend started and watching for questions...")
watch_file()
