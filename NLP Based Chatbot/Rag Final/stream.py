import streamlit as st
import time
import os

ANSWER_FILE = "answer.txt"
QUESTION_FILE = "question.txt"

def clear_file(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"{filepath} deleted.")

def read_answer_file(answer_file):
    if os.path.exists(answer_file):
        with open(answer_file, "r") as f:
            return f.read().strip()
    return ""

def clear_answer_file(answer_file):
    if os.path.exists(answer_file):
        with open(answer_file, "w") as f:
            f.write("")

# Initialize chat history in session_state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def main():
    st.title("💬 IIIT Delhi Chatbot")

    st.write("Built by Pulkit Nargotra and Harshvardhan Singh")

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["question"])
        with st.chat_message("assistant"):
            st.markdown(chat["answer"])

    question = st.chat_input("Enter your question or type 'EXIT' to end the chat:")
    
    if question:
        # Clear previous answer file
        clear_answer_file(ANSWER_FILE)

        # Write new question to file
        with open(QUESTION_FILE, "w") as f:
            f.write(question)

        with st.chat_message("user"):
            st.markdown(question)

        # Placeholder for answer
        with st.chat_message("assistant"):
            answer_placeholder = st.empty()

        # Poll for answer
        prev_answer = ""
        timeout = time.time() + 300  # 5-minute timeout

        while time.time() < timeout:
            answer = read_answer_file(ANSWER_FILE)
            if answer and answer != prev_answer:
                answer_placeholder.markdown(answer)
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer
                })
                if(answer == 'Thank You! Have a nice day!'):
                    return
                break
            time.sleep(1)
        else:
            answer_placeholder.error("❌ Timed out waiting for answer.")

if __name__ == "__main__":
    main()
