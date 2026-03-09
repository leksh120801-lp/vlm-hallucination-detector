Hallucination Detection Method

We compute cosine similarity between image and caption embeddings using CLIP.

If similarity score is below a threshold (0.25), the caption is flagged as a potential hallucination.

Similarity > threshold → likely correct caption  
Similarity < threshold → possible hallucination