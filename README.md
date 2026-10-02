# GenAI
GenAI Models

**1. Laya model of Jev for decision making**

Laya is an open-weight, non-autoregressive decision model developed by Convai Innovations. It is not a small chatbot. It takes a state, such as an email, support ticket, agent trace or JSON object, and returns a structured decision with probabilities, rather than generating text token by token.

**Architecture**

It has 421M parameters: a ModernBERT-large encoder, a ~25M decision head, and a small head for "answer or escalate" routing. 

It scores the candidate answers in a single forward pass using masked token scoring, and there is a multilingual router. 

It is trained with RLCD (Reinforcement Learning for Calibrated Decisions), so that its confidence tracks how often it is actually right. 

**Use cases**

It handles small structured calls like routing a ticket (billing, product, security or account), flagging something for review, or deciding whether to escalate to a larger model.

A low-confidence verdict can be passed up to a bigger model, which is the "System 1 / System 2" split.

It is cheap and fast: tens of milliseconds on a Tesla T4, and about 13.4 ms on an M3 Max via an independent MLX port. 

The weights are Apache 2.0 and it can run entirely on local hardware. 


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

step    0  loss 0.4971  pred_std 0.1350
step   50  loss 0.0159  pred_std 0.0009
step  100  loss 0.0047  pred_std 0.0008
step  150  loss 0.0045  pred_std 0.0008
step  200  loss 0.0034  pred_std 0.0007
step  250  loss 0.0031  pred_std 0.0007
step  300  loss 0.0024  pred_std 0.0007
step  350  loss 0.0023  pred_std 0.0007
step  400  loss 0.0020  pred_std 0.0007
step  450  loss 0.0020  pred_std 0.0008
step  500  loss 0.0018  pred_std 0.0008
step  550  loss 0.0019  pred_std 0.0010
step  600  loss 0.0017  pred_std 0.0012
step  650  loss 0.0018  pred_std 0.0012
step  700  loss 0.0017  pred_std 0.0016
step  750  loss 0.0016  pred_std 0.0019
step  800  loss 0.0022  pred_std 0.0025
step  850  loss 0.0018  pred_std 0.0037
step  900  loss 0.0020  pred_std 0.0049
step  950  loss 0.0016  pred_std 0.0070
features: (8, 128)
