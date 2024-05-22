import torch

def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device
):
    """
    Make causal mask used for bi-directional self-attention.
    Copied from Huggingface
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask


class Tree:
    def __init__(self, device :str = 'cpu', max_length = 512, dtype = torch.float16) -> None:
        self.tokens = torch.zeros(max_length, device=device).long()
        self.Successors :list[list[int]] = []
        self.num_nodes = 0
        self.device = device
        self.max_length = max_length
        self.dtype = dtype


    def initialize(self, attn_mask, sequence, new_tokens_buffer, parents_buffer, position_ids, active_mark):
        self.full_attn_mask = attn_mask
        self.sequence = sequence
        self.new_tokens_buffer = new_tokens_buffer
        self.parents_buffer = parents_buffer
        self.position_ids = position_ids
        self.active_mark = active_mark
        self.full_attn_mask = self.full_attn_mask.repeat(2, 2)

    def set_prefix(self, prefix: torch.LongTensor):
        self.tokens[:len(prefix)] = prefix.to(self.device)
        self.position_ids[:len(prefix)] = torch.arange(len(prefix))
        
        self.num_nodes = len(prefix)
        self.full_attn_mask[:self.max_length, :self.max_length] = _make_causal_mask((1, self.max_length),dtype=self.dtype, device=self.device)

        
        

    def collective_expand_position(self, expand_tokens :torch.LongTensor):
        self.tokens = torch.cat([self.tokens, expand_tokens], dim=-1)
        

    def verbose(self):
        print(self.tokens)
        print(self.Successors)






        



