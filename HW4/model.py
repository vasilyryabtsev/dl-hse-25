import torch
from typing import Type
from torch import nn
from dataset import TextDataset
from torch.nn import functional as F

class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super(LanguageModel, self).__init__()
        self.dataset = dataset  # required for decoding during inference
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Create necessary layers
        """
        self.embedding = nn.Embedding(self.vocab_size, embed_size, padding_idx=dataset.pad_id)
        self.rnn = rnn_type(embed_size, hidden_size, num_layers=rnn_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, length, vocab_size)
        """
        # This is a placeholder, you may remove it.
        # logits = torch.randn(
        #     indices.shape[0], indices.shape[1], self.vocab_size,
        #     device=indices.device
        # )
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Convert indices to embeddings, pass them through recurrent layers
        and apply output linear layer to obtain the logits
        """
        x = self.embedding(indices)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            sequence=out,
            padding_value=self.dataset.pad_id,
            batch_first=True
        )
        logits = self.linear(out)
        return logits

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        self.eval()
        # This is a placeholder, you may remove it.
        # generated = prefix + ', а потом купил мужик шляпу, а она ему как раз.'
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Encode the prefix (do not forget the BOS token!),
        pass it through the model to accumulate RNN hidden state and
        generate new tokens sequentially, sampling from categorical distribution,
        until EOS token or reaching self.max_length.
        Do not forget to divide predicted logits by temperature before sampling
        """
        prefix_index = self.dataset.text2ids(prefix)
        if isinstance(prefix_index, str):
            prefix_index = [prefix_index]
        indices = [self.dataset.bos_id] + prefix_index
        generated = [prefix]
        h = None
        for index in indices:
            index_tensor = torch.tensor([index], dtype=torch.long)
            x = self.embedding(index_tensor)
            if h is None:
                out, h = self.rnn(x)
            else:
                out, h = self.rnn(x, h)
        eos_token = self.dataset.ids2text(self.dataset.eos_id)
        for _ in range(max(0, self.max_length - 2 - len(indices))):
            logits = self.linear(out) / temp
            probs = F.softmax(logits, dim=-1)
            out = torch.multinomial(probs, num_samples=1).squeeze(0)
            text = self.dataset.ids2text(out)
            generated.append(text)
            if text == eos_token:
                break
            x = self.embedding(out)
            out, h = self.rnn(x, h)
        return " ".join(generated).replace(eos_token, '')
