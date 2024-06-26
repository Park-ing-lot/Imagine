'''
Do not import these models directly.
You need to modify the modeling_roberta.py and the corresponding __init__.py files in the Transformers package to correctly load adapters.
Ensure that the following command does not produce an error before training/evaluation:

"from transformers import RobertaForMaskedLM_Imagine"

Any suggestions to make this process simpler would be helpful.
'''


class RobertaPooler_Imagine(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.dropout(hidden_states)
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)        
        return pooled_output

class RobertaForMaskedLM_Imagine(RobertaPreTrainedModel):
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        self.itm_pooler = RobertaPooler_Imagine(config)
        # self.how_helpful_head = RobertaPooler_ROMI_how_helpful(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        image_hidden_states = None,
        batch_size=None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
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

        prediction_scores = self.lm_head(sequence_output[:-len(image_hidden_states)])

        output = (prediction_scores,) + outputs[2:] + (qk, itm_scores,)
        return output