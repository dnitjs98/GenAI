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

split/train-00000-of-00001.parquet: reconstructing file: 100%
 1.03MB / 1.03MB, 99.1kB/s  
split/train-00000-of-00001.parquet: downloading bytes: 
 1.03MB, 98.2kB/s  
split/validation-00000-of-00001.parquet: reconstructing file: 100%
  127kB /  127kB, 12.3kB/s  
split/validation-00000-of-00001.parquet: downloading bytes: 
  127kB, 12.2kB/s  
split/test-00000-of-00001.parquet: reconstructing file: 100%
  129kB /  129kB, 12.5kB/s  
split/test-00000-of-00001.parquet: downloading bytes: 
  129kB, 12.4kB/s  
Generating train split: 100%
 16000/16000 [00:00<00:00, 248046.99 examples/s]
Generating validation split: 100%
 2000/2000 [00:00<00:00, 73135.20 examples/s]
Generating test split: 100%
 2000/2000 [00:00<00:00, 79500.82 examples/s]
Download complete: : 
  846MB,  205MB/s  
Reconstruction complete: 100%
  846MB /  846MB,  227MB/s  
Fetching 5 files: 100%
 5/5 [00:06<00:00,  1.68s/it]
/usr/local/lib/python3.13/dist-packages/laya/agent.py:1869: RuntimeWarning: laya: this checkpoint ships invalid temperatures or values outside [0.5, 5]; using choice:11+=0.10058280825614929 -> 0.5. Treat confidence from the affected entries as uncalibrated.
  return Agent(model_id_or_path, device=device, token=token, subfolder=subfolder, fast=fast,
Samples        : 500
Accuracy       : 0.590   (random = 0.167)
Mean confidence: 0.768
ECE            : 0.210

Confidence gating (auto-decide vs escalate):
  thr  coverage  acc@auto  escalated
 0.50     0.828     0.633         86
 0.60     0.742     0.658        129
 0.70     0.664     0.681        168
 0.80     0.594     0.690        203
 0.90     0.488     0.709        256

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
