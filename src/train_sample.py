from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    model_name = "distilgpt2"
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    input_text = "Hello, llm-lab!"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model(**inputs)
    print("Forward pass completed. Logits shape:", outputs.logits.shape)


if __name__ == "__main__":
    main()
