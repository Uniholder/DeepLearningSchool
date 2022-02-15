# PyTorch Transformer

Transformer model for PyTorch. This repository mostly replicates the architecture of Open AI's [Sparse Transformer](https://arxiv.org/abs/1904.10509) with some minor changes – we use different learning rate scheduler and slightly different weight initialization.

### Requirements:
```
PyTorch >= 1.2
Apex >= 0.1
```

### Initialization:
```python
from transformer import Transformer

transformer = Transformer(
    n_layers=6,
    n_heads=8,
    dropout=0.35,
    encoder_embeddings_size=(32000, 300),
    decoder_embeddings_size=(15000, 300),
)

```

Here we have Transformer with 6 layers and 8 heads with randomly initialized embeddings matrices.

To use pre-trained embeddings, one could pass `embeddings_path` parameter to the model with a path to saved numpy array with token embeddings.

```python
from transformer import Transformer

transformer = Transformer(
    n_layers=6,
    n_heads=8,
    dropout=0.35,
    encoder_embeddings_path="encoder_embeddings.npy",
    decoder_embeddings_path="decoder_embeddings.npy",
)
```

### Usage:

Transformer takes as input two parameters: inputs to encoder and decoder which are `Long` Tensors with shape `batch_size, sequence_length` and returns logits of the distribution over target tokens.

```python
encoder_input = torch.tensor([[1, 2, 3, 4, 5, 0, 0], [6, 7, 8, 1, 2, 3, 4]])
decoder_input = torch.tensor([[10, 11, 12, 13], [14, 15, 16, 0]])
print(transformer(encoder_input, decoder_input).size())  # torch.Size([2, 4, 15000])
```

Here `0` represents padding in the input which would be omitted during model propagation.

When `decoder_input` isn't passed, then the model would generate new sequences from the inputorch.

```python
encoder_input = torch.tensor([[6, 7, 8, 1, 3, 4]])
print(transformer(encoder_input, n_beams=5, n_generations=2))
"""
[[12855, 13565, 6437, 10205, 8335, 2],
[13461, 14500, 11274, 2443, 8421, 7234, 12032, 2]
"""

```
Where `n_beams` is the number of beams in beam search of the most probable sequence under the model and `n_generations` is the number of returned sequences. Note that `n_generations` < `n_beams`.

To omit out-of-memory issues during the training, one could use gradient checkpoints

```python
transformer(encoder_input, decoder_input, checkpoint_gradients=True)
```
Note that there is no sense in adding gradients checkpoints when new sequences are generated since generation is performed in `no_grad` mode.

Also, it's necessary to change learning rate during the training, in this repo we use scheduler from [original work](https://arxiv.org/abs/1706.03762).

```python
from other.scheduler import Scheduler
from torch.optim import Adam

optimizer = Adam(transformer.parameters(), betas=(0.9, 0.999), eps=1e-6)
scheduler = Scheduler(transformer.h_s, n_warmup_steps=4000)

scheduler.update_learning_rate(optimizer) # update lr every train step
```

### Example:
There is example of training Transformer in `train.py`.

```bash
python3 train.py --checkpoint-gradients --opt-level O2
```

This command will train the model with gradients checkpoint and “Almost FP16” Mixed Precision. Other options for mixed-precision could be found in Apex documentation.