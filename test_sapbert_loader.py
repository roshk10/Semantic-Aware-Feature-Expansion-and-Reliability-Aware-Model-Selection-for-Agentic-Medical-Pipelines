from utils.sapbert import (
    get_embedding,
    cosine_similarity
)

print("\nTesting SapBERT...\n")

blood_pressure = get_embedding(
    "blood pressure"
)

resting_bp = get_embedding(
    "resting blood pressure"
)

cholesterol = get_embedding(
    "serum cholesterol"
)

print(
    "\nEmbedding shape:",
    blood_pressure.shape
)

print(
    "\nBP vs Resting BP:",
    cosine_similarity(
        blood_pressure,
        resting_bp
    )
)

print(
    "\nBP vs Cholesterol:",
    cosine_similarity(
        blood_pressure,
        cholesterol
    )
)

print("\nSapBERT working.")