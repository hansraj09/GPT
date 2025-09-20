import torch
import torch.nn as nn
from torchtyping import TensorType

class GPT(nn.Module):
  def __init__(self, vocab_size: int, context_length: int, model_dim: int, num_blocks: int, num_heads: int):
    super().__init__()
    self.word_embeddings = nn.Embedding(vocab_size, model_dim)
    self.position_embeddings = nn.Embedding(context_length, model_dim)
    self.transformer_blocks = nn.Sequential()
    for _ in range(num_blocks):
      self.transformer_blocks.append(self.TransformerBlock(model_dim, num_heads))
    self.final_norm = nn.LayerNorm(model_dim)
    self.vocab_projection = nn.Linear(model_dim, vocab_size)

  def forward(self, context: TensorType[int]) -> TensorType[float]:
    embedded = self.word_embeddings(context) # B x T x E
    context_length = embedded.shape[1]
    positions = torch.arange(context_length)
    embedded = embedded + self.position_embeddings(positions)

    raw_output = self.vocab_projection(self.final_norm(self.transformer_blocks(embedded))) # B x T x V

    probs = nn.functional.softmax(raw_output, dim=2)
    return probs

  class TransformerBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
      super().__init__()
      self.mhsa = self.MultiHeadedSelfAttention(model_dim, num_heads)
      self.first_norm = nn.LayerNorm(model_dim)
      self.feed_forward = self.VanillaNeuralNetwork(model_dim)
      self.second_norm = nn.LayerNorm(model_dim)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
      embedded =  embedded + self.mhsa(self.first_norm(embedded))
      embedded = embedded + self.feed_forward(self.second_norm(embedded))
      return embedded

    class MultiHeadedSelfAttention(nn.Module):
      def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.att_heads = nn.ModuleList()
        for _ in range(num_heads):
          self.att_heads.append(self.SingleHeadAttention(model_dim, model_dim // num_heads))

      def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        output = []
        for head in self.att_heads:
          output.append(head(embedded))
        cated = torch.cat(output, dim=2)
        return cated

      class SingleHeadAttention(nn.Module):
        def __init__(self, model_dim: int, head_size: int):
          super().__init__()
          self.get_key = nn.Linear(model_dim, head_size, bias=False)
          self.get_query = nn.Linear(model_dim, head_size, bias=False)
          self.get_value = nn.Linear(model_dim, head_size, bias=False)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
          key = self.get_key(embedded)
          query = self.get_query(embedded)
          value = self.get_value(embedded)

          scores = query @ torch.transpose(key, 1, 2) #q = B x T x A, k= B x T x A -> B x A x T
          context_length, attention_dim = scores.shape[1], scores.shape[2]
          scores = scores / (attention_dim ** 0.5)

          lower_triangular = torch.tril(torch.ones(context_length, context_length)) # T x T
          mask = lower_triangular == 0
          scores = scores.masked_fill(mask, float('-inf')) # replaces 0 with -inf so that softmax does not take into account 1/0
          scores = nn.functional.softmax(scores, dim=2)

          return scores @ value
        
    class VanillaNeuralNetwork(nn.Module):
      def __init__(self, model_dim: int):
        super().__init__()
        self.up_projection = nn.Linear(model_dim, model_dim * 4)
        self.relu = nn.ReLU()
        self.down_projection = nn.Linear(model_dim * 4, model_dim)
        self.dropout = nn.Dropout(0.2) # using p = 0.2

      def forward(self, x: TensorType[float]) -> TensorType[float]:
        return self.dropout(self.down_projection(self.relu(self.up_projection(x))))        

def main():
  model = GPT(vocab_size=3, context_length=2, model_dim=4, num_blocks=2, num_heads=2)
  context = [[1,0],[2,0]] 
  out = model(torch.tensor(context))
  print(out)

if __name__ == "__main__":
  main()