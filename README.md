# GenAI
GenAI Models

**1. Laya model of Jev for decision making**

**2. JEPA (Joint Embedding Predictive Architecture)**

JEPA (Joint Embedding Predictive Architecture) is a self-supervised learning approach proposed by Yann LeCun. Instead of reconstructing raw inputs (pixels or tokens), it predicts the representation of one part of the input from the representation of another.

Core idea

A context encoder embeds the visible part of the input (x).
A target encoder embeds the hidden or future part (y).
A predictor maps the context embedding (plus positional or conditioning info) to the predicted target embedding.
The loss is computed in latent space, not in input space.

Why it matters

It ignores unpredictable low-level detail (exact pixels, noise) and focuses on abstract, semantic structure.
It avoids the costs and failure modes of generative reconstruction.
It is a building block in LeCun's proposed path toward world models.

Main challenge
Representation collapse: the encoders can map everything to a constant. Variants avoid this with an EMA target encoder and stop-gradient (I-JEPA, V-JEPA), or with explicit regularization of the embedding distribution (VICReg-style, and newer approaches that aim to remove the heuristics).

Family

I-JEPA: images, predicting masked block embeddings.
V-JEPA / V-JEPA 2: video, with V-JEPA 2 extending to world-model and robot-planning uses.
Newer variants for other modalities and for simplifying the training recipe.
