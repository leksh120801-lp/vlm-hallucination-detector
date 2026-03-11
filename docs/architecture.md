System Architecture

Image → CLIP Vision Encoder → Image Embedding
Caption → CLIP Text Encoder → Text Embedding

Image Embedding + Text Embedding
        ↓
Cosine Similarity
        ↓
Hallucination Score

                IMAGE
                  │
                  ▼
          Vision Transformer
             (CLIP encoder)
                  │
                  ▼
           Image Embedding
                  │
                  ▼
            Cosine Similarity
                  ▲
                  │
           Text Embedding
                  ▲
                  │
           Text Transformer
            (CLIP encoder)
                  ▲
                  │
                TEXT



Dataset
   │
   ▼
Image + Caption
   │
   ▼
CLIP Model
   │
   ├── Image Encoder → Image Embedding
   │
   └── Text Encoder → Text Embedding
   │
   ▼
Cosine Similarity
   │
   ▼
Hallucination Detection
   │
   ▼
Experiment Results