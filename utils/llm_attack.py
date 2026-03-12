from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="gpt2",
    max_length=50
)

def generate_llm_attacks(caption, num_attacks=3):

    prompt = f"Generate incorrect image captions different from: {caption}"

    outputs = generator(prompt, num_return_sequences=num_attacks)

    attacks = []

    for o in outputs:
        text = o["generated_text"]
        attacks.append(text)

    return attacks