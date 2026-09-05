from transformers.models.bert import *
from .config import MODELS
import torch

class MyBert(BertPreTrainedModel):
    def __init__(self, config):
        
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()
    
    # @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None
    ):

        # print("++++++++++++++++++++++++++++++++++++++++++++++",head_mask, input_ids,"+++++++++++++++++++++++++++++++++++")
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        output = outputs[0]

        # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^", self.bert,"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        # output = self.dropout(output)

        # print("==================================================",pooled_output,"=================================")
        
        return output


class TextBERTLSTM(torch.nn.Module):
    def __init__(self, pretrained_model, num_class, fine_tune):
        super(TextBERTLSTM, self).__init__()
        # print("jksadhfkjlsh",pretrained_model)
        self.bert = MyBert.from_pretrained(pretrained_model)
        self.pp = self.bert.config
        # self.dp = self.drop
        

        # Freeze bert layers
        if not fine_tune:
            for p in self.bert.parameters():
                p.requires_grad = False

        #new layers
        self.lstm = torch.nn.LSTM(768, 512,2,batch_first=True,bidirectional=True)
        self.dropout = torch.nn.Dropout(.3)
        self.linear = torch.nn.Linear(512*2, 12)

        bert_dim = MODELS[pretrained_model][2]
        self.classifier = torch.nn.Linear(bert_dim, num_class)

    def forward(self, x, attn_masks):
        outputs = self.bert(x, attention_mask=attn_masks)
        lstm_output, (h,c) = self.lstm(outputs)
        lstm_output = self.dropout(lstm_output)
            # print(sequence_output.shape,"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", lstm_output.shape)
        hidden = torch.cat((lstm_output[:,-1,:128], lstm_output[:,0, 128:]), dim=-1)
        linear_output = self.linear(hidden.view(-1,512*2))
        # logits = self.classifier(outputs)
        
        return linear_output
