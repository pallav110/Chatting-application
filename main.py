import json
from difflib import get_close_matches
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging

# Set up logging
# logging.basicConfig(level=logging.DEBUG)
# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
print(nltk.data.find('tokenizers/punkt'))

# Load and save knowledge base
def load_knowledge_base(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return {"questions": []}

def save_knowledge_base(file_path: str, data: dict):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)

# Load and save personality data
def load_personalities(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return {"personalities": []}

def find_best_match(user_question: str, questions: list[str]) -> str:
    matches = get_close_matches(user_question, questions, n=1, cutoff=0.6)
    return matches[0] if matches else None

def get_answer_for_question(question: str, knowledge_base: dict) -> str:
    for q in knowledge_base.get("questions", []):
        if q["question"] == question:
            return q["answer"]
    return None

class GPT2ChatAPI:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()  # Set model to evaluation mode

    def generate_response(self, prompt: str, max_length: int = 100) -> str:
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')

        outputs = self.model.generate(
            inputs,
            max_length=max_length + len(inputs[0]),
            do_sample=True,
            temperature=0.3,  # Lower temperature for more focused responses
            top_k=50,
            top_p=0.85,
            pad_token_id=self.tokenizer.eos_token_id
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        response = response.split(".")[0] + "."
        return response

def select_personality(data: dict) -> dict:
    print("Select a personality to chat with:")
    for index, personality in enumerate(data.get('personalities', [])):
        print(f"{index + 1}. {personality.get('name', 'Unknown')}")

    while True:
        try:
            choice = int(input("Enter the number corresponding to the personality: "))
            if 1 <= choice <= len(data['personalities']):
                selected_personality = data['personalities'][choice - 1]
                print(f"\nYou are now chatting with {selected_personality['name']}!")
                return selected_personality
            else:
                print("Invalid choice. Please select a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def construct_prompt(personality: dict, user_input: str) -> str:
    return (f"Imagine you are {personality['name']}, who is {personality['description']}. "
            f"You are having a conversation with a user. The user just asked: '{user_input}'\n"
            f"{personality['name']}: ")

def generate_ai_response(personality: dict, user_input: str) -> str:
    prompt = construct_prompt(personality, user_input)
    response_text = chatbot.generate_response(prompt)
    return response_text

def chat_bot():
    knowledge_base = load_knowledge_base('data/knowledge_base.json')
    personalities = load_personalities('data/personalities.json')

    if not personalities.get('personalities'):
        print("Error: No personalities found in the personality data.")
        return

    selected_personality = select_personality(personalities)

    while True:
        user_input = input('You: ')

        if user_input.lower() == 'exit':
            print("Ending the chat. Have a great day!")
            break

        best_match = find_best_match(user_input, [q["question"] for q in knowledge_base.get("questions", [])])

        if best_match:
            answer = get_answer_for_question(best_match, knowledge_base)
            print(f'Bot: {answer}')
        else:
            response = generate_ai_response(selected_personality, user_input)
            print(f'Bot: {response}')

            # Remove the teaching part
            # teach_response = input('Bot: I don’t know the answer. Can you teach me? (yes/no): ').lower()
            # if teach_response == 'yes':
            #     new_answer = input('Type the answer or "skip" to skip: ')
            #     if new_answer.lower() != 'skip':
            #         knowledge_base["questions"].append({"question": user_input, "answer": new_answer})
            #         save_knowledge_base('data/knowledge_base.json', knowledge_base)
            #         print('Bot: Thank you! I learned a new response.')

if __name__ == '__main__':
    chatbot = GPT2ChatAPI()
    chat_bot()
