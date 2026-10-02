# GenAI
GenAI Models

1. Laya model of Jev for decision making
**2. JEPA (Joint Embedding Predictive Architecture)**
   _It contains:

  Context encoder: a ViT that sees only the context patches.
  Target encoder: an EMA copy of the context encoder, with no gradients. It encodes the full image, and the outputs are layer-normed.
  Predictor: a narrow ViT that takes the context tokens plus positional mask tokens and outputs embeddings for the target patches.
  Masking: 4 target blocks plus one large context block with the targets removed. The masks are shared across the batch so shapes match.
  Loss: smooth L1 between predicted and target embeddings in latent space.
  Collapse guard: the EMA momentum ramps from 0.996 to 1, and the script prints the prediction std so you can see if it collapses.

  The training loop uses synthetic images of colored shapes so it runs offline. To use real data, replace synthetic_batch with your own loader and keep the 32×32 size or adjust IMG and PATCH.
_
