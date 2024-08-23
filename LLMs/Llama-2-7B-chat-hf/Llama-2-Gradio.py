from transformers import AutoTokenizer, pipeline
import transformers
import torch
import gradio as gr

model = "Tharun2331/Llama-2-7B-chat-hf-custom"

tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

llama_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device=0,
)

SYSTEM_PROMPT = """<s>[INST] <<SYS>>
You are a helpful bot. Your answers are clear and concise.
<</SYS>>

"""

def format_message(message: str, history: list, memory_limit: int = 3) -> str:
    """
    Formats the message and history for the Llama model.

    Parameters:
        message (str): Current message to send.
        history (list): Past conversation history.
        memory_limit (int): Limit on how many past interactions to consider.

    Returns:
        str: Formatted message string
    """
    if len(history) > memory_limit:
        history = history[-memory_limit:]

    if len(history) == 0:
        return SYSTEM_PROMPT + f"{message} [/INST]"

    formatted_message = SYSTEM_PROMPT + f"{history[0][0]} [/INST] {history[0][1]} </s>"

    for user_msg, model_answer in history[1:]:
        formatted_message += f"<s>[INST] {user_msg} [/INST] {model_answer} </s>"

    formatted_message += f"<s>[INST] {message} [/INST]"

    return formatted_message

def get_llama_response(message: str, history: list) -> str:
    """
    Generates a conversational response from the Llama model.

    Parameters:
        message (str): User's input message.
        history (list): Past conversation history.

    Returns:
        str: Generated response from the Llama model.
    """
    query = format_message(message, history)
    response = ""

    sequences = llama_pipeline(
        query,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=1024,
    )

    generated_text = sequences[0]['generated_text']
    response = generated_text[len(query):]

    print("Chatbot:", response.strip())
    return response.strip()

llama_gr = gr.ChatInterface(get_llama_response)
llama_gr.launch()