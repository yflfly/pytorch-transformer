embedding_output = self.embeddings(
    input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
)
encoder_outputs = self.encoder(
    embedding_output,
    attention_mask=extended_attention_mask,
    head_mask=head_mask,
    encoder_hidden_states=encoder_hidden_states,
    encoder_attention_mask=encoder_extended_attention_mask,
)
sequence_output = encoder_outputs[0]
pooled_output = self.pooler(sequence_output)

outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)
