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

**Limitations**

Zero-shot is weak. The base model performs only slightly above the random baseline in zero-shot use. The project itself frames it as a base to fine-tune, not an out-of-the-box decision engine. Fine-tuned, it reaches about 0.766 on its typed-decision set. 

Few options at a time. The docs recommend keeping multiple-choice questions to under roughly 20 options, and it did poorly on a 77-class benchmark (0.425). 

Short inputs. The English checkpoint has a 512-token input budget; other variants extend it to 1,024. 

Not hallucination-proof. It does not generate free text, but it can still make the wrong decision. 

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

**HMM** - produce tokens sequentially where label depends upon previous label and token depends upon label

**SG** - Represent sentence in the form of label phrase tree structure

**MEMM** - it predict label on the basis of previous label and sentence

**NAT - Non-autoregressive translation**

It uses XE, AXE, OAXE loss for the improvement of model.
