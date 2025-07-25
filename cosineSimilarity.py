# from sentence_transformers import SentenceTransformer, util
#
# # --- Option 1: hard-code strings here ----------------------------------
# STATEMENT_A = "EXPLORER-HCM: CAMZYOS® (mavacamten), a first-in-class cardiac myosin inhibitor, demonstrated superiority over placebo across all primary and secondary endpoints in patients with symptomatic obstructive hypertrophic cardiomyopathy (HCM).1"
# STATEMENT_B = "EXPLORER-HCM: CAMZYOS® (mavacamten), a first-in-class cardiac myosin inhibitor, showed superiority over placebo on all primary and secondary endpoints in patients with symptomatic obstructive hypertrophic cardiomyopathy (HCM).1."
# # -----------------------------------------------------------------------
#
# # if not (STATEMENT_A and STATEMENT_B):
# #     # --- Option 2: interactive prompt (fallback) -----------------------
# #     STATEMENT_A = input("Enter first sentence: ").strip()
# #     STATEMENT_B = input("Enter second sentence: ").strip()
# #     # -------------------------------------------------------------------
#
# # Load a lightweight sentence-embedding model (downloads once)
# model = SentenceTransformer("all-MiniLM-L6-v2")  # ~80 MB
#
# emb_a = model.encode(STATEMENT_A, convert_to_tensor=True)
# emb_b = model.encode(STATEMENT_B, convert_to_tensor=True)
#
# # Cosine similarity ranges (-1, 1) → scale to (0, 1)
# cos_sim = util.cos_sim(emb_a, emb_b).item()
# similarity_score = (cos_sim + 1) / 2
#
# print(f"\nSimilarity: {similarity_score:.4f}")


"""
Compute semantic similarity between two English sentences with
(1) a lightweight model:  all-MiniLM-L6-v2
(2) a higher-accuracy model: all-mpnet-base-v2

"""

from sentence_transformers import SentenceTransformer, util

# ── 1.  Put sentences here (empty -> will prompt) ───────────────────────
STATEMENT_A = 'CAMZYOS® is a first-in-class cardiac myosin inhibitor that provides sustained relief from symptoms in patients with symptomatic obstructive hypertrophic cardiomyopathy, as demonstrated in the EXPLORER-HCM clinical trial'
STATEMENT_B = 'CAMZYOS® is a first-in-class cardiac myosin inhibitor that provides long-term symptomatic relief in patients with symptomatic obstructive hypertrophic cardiomyopathy, as demonstrated in the EXPLORER-HCM clinical trial.'
# ------------------------------------------------------------------------

if not (STATEMENT_A and STATEMENT_B):
    STATEMENT_A = input("Enter first sentence: ").strip()
    STATEMENT_B = input("Enter second sentence: ").strip()

MODELS = {
    "MiniLM (all-MiniLM-L6-v2)": "all-MiniLM-L6-v2",          #  384-dim
    "MPNet (all-mpnet-base-v2)": "all-mpnet-base-v2",         #  768-dim
}

embeddings = {}
for label, name in MODELS.items():
    print(f"Loading {label} …")
    model = SentenceTransformer(name)
    embeddings[label] = model.encode([STATEMENT_A, STATEMENT_B], convert_to_tensor=True)

print("\nSimilarity scores")
print("------------------")
for label, (emb1, emb2) in embeddings.items():
    score = util.cos_sim(emb1, emb2).item()
    score = (score + 1) / 2        # map from (-1,1) to (0,1)
    print(f"{label:<30}: {score:.4f}")
