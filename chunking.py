from chonkie import SemanticChunker

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

def chunk_text(text):
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

chunk_text(text)


"""
- Metadata fields

- user_character
- opponent_character
- 
"""