System Architecture

Image → CLIP Vision Encoder → Image Embedding
Caption → CLIP Text Encoder → Text Embedding

Image Embedding + Text Embedding
        ↓
Cosine Similarity
        ↓
Hallucination Score