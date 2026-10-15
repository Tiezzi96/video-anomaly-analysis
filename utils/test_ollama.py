import ollama

def ask_model(model_name, prompt):
    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0.1
        }
    )

    return response["message"]["content"]

prompt = """
Contesto:
Ingresso di un ospedale.

Descrizione:
Una persona entra portando una grande bandiera.

Valuta se rappresenta un'anomalia contestuale.
"""

print("=== Qwen ===")
print(ask_model("qwen3.5:9b", prompt))

print("\n=== Llama ===")
print(ask_model("llama3.1", prompt))