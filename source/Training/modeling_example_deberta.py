'''
Do not import these models directly.
You need to modify the modeling_deberta_v2.py and the corresponding __init__.py files in the Transformers package to correctly load adapters.
Ensure that the following command does not produce an error before training/evaluation:

"from transformers import DebertaV2ForMaskedLM_Imagine"

Any suggestions to make this process simpler would be helpful.
'''

class ContextPooler_Imagine(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.pooler_hidden_size, config.pooler_hidden_size)
        # self.dense2 = nn.Linear(config.pooler_hidden_size, config.pooler_hidden_size,)
        self.dropout = StableDropout(config.pooler_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.dropout(hidden_states)
        pooled_output = self.dense(pooled_output)
        # pooled_output = torch.nn.functional.gelu(pooled_output)
        # pooled_output = self.dropout(pooled_output)
        # pooled_output = self.dense2(pooled_output)
        pooled_output = torch.nn.functional.tanh(pooled_output) ###
        
        return pooled_output

class DebertaV2ForMaskedLM_Imagine(DebertaV2PreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.deberta = DebertaV2Model(config)
        self.cls = DebertaV2OnlyMLMHead(config)
        self.itm_pooler = ContextPooler_Imagine(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        image_hidden_states = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        pooled_text = self.itm_pooler(sequence_output[-len(image_hidden_states):, 0, :])
        pooled_image = self.itm_pooler(image_hidden_states)
        
        scale = torch.sqrt(torch.tensor(pooled_image.size(-1), dtype=torch.float))

        qk = torch.einsum('ac,adc->ad', pooled_text, pooled_image) / scale.to(dtype=pooled_text.dtype)
        qkv = torch.einsum('ad,adc->ac', qk.softmax(-1), pooled_image)
        
        logit_scale = torch.tensor([2.6592]).exp().to(pooled_text.device)
        cos = nn.CosineSimilarity(dim=1, eps=1e-9)
        itm_scores = cos(pooled_text, qkv) * logit_scale

        prediction_scores = self.cls(sequence_output[:-len(image_hidden_states)])

        output = (prediction_scores,) + outputs[1:] + (qk, itm_scores,)
        return output
    
