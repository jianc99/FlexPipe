import torch
from pipe.llm_pipe import LLMEngine
import torch.distributed as dist

class LLM_Pipeline:
    def __init__(self, 
                model_name: str,
                pp_config : dict,
                batch_size :int = 1,
                max_length :int = 256, 
                device :str = 'cuda:0',
                dtype = torch.float16,
                ) -> None:
        self.pp_config = pp_config
        self.bsz = batch_size
        self.max_length = max_length
        self.is_first_stage = (self.pp_config["current_stage"] == 0)
        self.is_last_stage = (self.pp_config["current_stage"] == self.pp_config["num_stages"] -1)
        self.dtype = dtype
        self.group_indices = self.pp_config["groups_indices"]
        self.current_stage = self.pp_config["current_stage"]
        self.process_group = self.pp_config["current_group"]
        self.local_rank = dist.get_rank(self.process_group)
        self.num_stages = self.pp_config["num_stages"]

        self.pp_engine = LLMEngine(max_length=max_length, model_name=model_name, device=device, pp_config=pp_config, dtype=dtype, batch_size=batch_size)
        dist.barrier()
        self.hidden_dim = self.pp_engine.llm.hidden_size
        self.pp_engine.initialize_cuda_graph([1, 128])
        dist.barrier()


    def forward(self,input_ids, position_ids, attention_mask, storage_ids):
        if self.num_stages == 1:
            return self.pp_engine.inference(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, storage_ids=storage_ids)

        input_ids=input_ids
        hidden_state = torch.full((self.bsz, input_ids.size(1), self.hidden_dim), 0, dtype=self.dtype, device=input_ids.device)
        output = torch.full((self.bsz, input_ids.size(1), 32000), 0, dtype=torch.float32, device=input_ids.device)

        if self.is_first_stage:
            hidden_state=self.pp_engine.inference(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, storage_ids=storage_ids)
            if self.local_rank == 0:
                dist.send(hidden_state, self.group_indices[self.current_stage+1][0])
        elif self.is_last_stage:
            if self.local_rank == 0:
                dist.recv(hidden_state,self.group_indices[self.current_stage-1][0])
            dist.broadcast(hidden_state, self.group_indices[self.current_stage][0], self.process_group)
            output=self.pp_engine.inference(input_ids=hidden_state, position_ids=position_ids, attention_mask=attention_mask, storage_ids=storage_ids)
        else:
            if self.local_rank == 0:
                dist.recv(hidden_state,self.group_indices[self.current_stage-1][0])
            dist.broadcast(hidden_state, self.group_indices[self.current_stage][0], self.process_group)
            hidden_state=self.pp_engine.inference(input_ids=hidden_state, position_ids=position_ids, attention_mask=attention_mask, storage_ids=storage_ids)
            if self.local_rank == 0:
                dist.send(hidden_state,self.group_indices[self.current_stage+1][0])
        dist.broadcast(output,self.group_indices[-1][0])
        return output

