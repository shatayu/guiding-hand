from chonkie import SemanticChunker
from transformers import AutoModelForCausalLM, AutoTokenizer

# Here's some text to chunk
text = """
Machine learning is a subset of artificial intelligence focused on building algorithms that learn from data. 
Neural networks, inspired by the brain, consist of layers of interconnected nodes. 
Deep learning extends this idea with many layers, enabling models to learn complex patterns.

The history of the Roman Empire spans centuries. 
It started as a republic before transitioning into imperial rule with Augustus. 
Major events include the Pax Romana, the spread of Christianity, and the empire's eventual decline.

Cooking pasta requires boiling water with salt. 
Add pasta when the water reaches a rolling boil. 
Fresh basil and garlic make excellent seasonings for Italian dishes. 
Always taste your sauce before serving to guests.

The theory of relativity, proposed by Albert Einstein, revolutionized physics. 
It introduced concepts like spacetime curvature and the relationship between mass and energy (E=mc^2).
This theory underpins modern cosmology and GPS technology.
"""

def chunk_text(text, model, tokenizer):
    # Initialize the chunker
    # Basic initialization with default parameters
    chunker = SemanticChunker(
        embedding_model="minishlab/potion-base-32M",  # Default model
        threshold=0.999999,                               # Similarity threshold (0-1)
        chunk_size=2048,                             # Maximum tokens per chunk
        similarity_window=3,                         # Window for similarity calculation
        skip_window=0                                # Skip-and-merge window (0=disabled)
    )

    # Chunk some text
    chunks = chunker(text)

    # Access chunks
    for chunk in chunks:
        print(f"Chunk: {chunk.text}")
        print(f"Tokens: {chunk.token_count}")
        
        # Create prompt string inside the loop
        prompt = f"""
        You are an AI assistant tasked with analyzing text chunks.

        Full text:
        {text}

        Current chunk to analyze:
        {chunk.text}

        Please analyze this chunk in the context of the full text and output your analysis as an artifact.
        """
        
        # Prepare the model input using chat template
        messages = [
            {"role": "user", "content": prompt}
        ]
        text_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
        
        # Generate response using Qwen3 model
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1000,
            temperature=0.7,
            do_sample=True
        )
        
        # Extract and decode the response
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        response_content = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        print(f"Qwen3 Response: {response_content}")
        print("="*50)


if __name__ == "__main__":
    # Load the Qwen3 model and tokenizer
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    print(f"Loading model: {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="cpu"
    )
    
    print("Model loaded successfully!")
    chunk_text(text, model, tokenizer)


"""
- Metadata fields

- user_character
- opponent_character
- 
"""