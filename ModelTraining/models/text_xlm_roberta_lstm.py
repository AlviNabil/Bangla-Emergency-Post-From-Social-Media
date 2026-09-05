from transformers.models.roberta import *
from .config import MODELS
import torch
from  utils import parse_args



class MyRobertaClassificationHead(torch.nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return x


class MyRoBerta(RobertaPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"
    

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.classifier = MyRobertaClassificationHead(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            # inputs_embeds=inputs_embeds,
            output_hidden_states = True,
        )
        
        # pooler_output = outputs['pooler_output']
        # output = self.classifier(sequence_output)
        args_ = parse_args()
        if(args_.use_last_4_hState==True):
            output = outputs
        else:
            output = outputs[0] #shape [32,100,768] - [batch_size, seq_length, hidden_size] output[0] is giving the last hidden state embedding

        # print(outputs)
        # print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        # print(outputs[0])
        # print("5555555555555555555555555555555555555555555555555555555555555",sequence_output.shape)
        return output
        


class TextRoBERTaLSTM(torch.nn.Module):
    def __init__(self, pretrained_model, num_class, fine_tune):
        super(TextRoBERTaLSTM, self).__init__()
        self.bert = MyRoBerta.from_pretrained(pretrained_model)
        # Freeze bert layers
        if not fine_tune:
            for p in self.bert.parameters():
                p.requires_grad = False
        #new layers
        args_ = parse_args()
        if(args_.use_last_4_hState==False):
            self.lstm = torch.nn.LSTM(768, 256,2,batch_first=True, dropout=.2,bidirectional=True)
            self.dropoutt=torch.nn.Dropout(.3)
            self.linear = torch.nn.Linear(256*2, 12)
        else:
            self.lstm = torch.nn.LSTM(768*4, 256,2,batch_first=True, bidirectional=True)
            self.linear = torch.nn.Linear(256*2,12)
        bert_dim = MODELS[pretrained_model][2]
        self.classifier = torch.nn.Linear(bert_dim, num_class)

    def forward(self, x, attn_masks):
        args_ = parse_args()
        if(args_.use_last_4_hState==True):
            outputs = self.bert(x, attention_mask=attn_masks)
            all_hidden_states = torch.stack(outputs[2]) #[13,32,100,768]
            # print("77777777777777777777777777777777777777777",all_hidden_states.detach().shape)
            concatenate_pooling = torch.cat((all_hidden_states[-1],all_hidden_states[-2],all_hidden_states[-3],all_hidden_states[-4]),-1)
            # concatenate_pooling = concatenate_pooling[:,0]
            
            
           

            #lstm
            lstm_output, (h,c) = self.lstm(concatenate_pooling)
            # print(concatenate_pooling.shape,"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", lstm_output.shape)
            hidden = torch.cat((lstm_output[:,-1,:256], lstm_output[:,0, 256:]), dim=-1)
            linear_output = self.linear(hidden.view(-1,256*2))
        
       


        else:
            sequence_output = self.bert(x, attention_mask=attn_masks)
            #lstm
            # print(sequence_output)
            lstm_output, (h,c) = self.lstm(sequence_output)
            lstm_output = self.dropoutt(lstm_output)
            # print(sequence_output.shape,"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", lstm_output.shape)
            hidden = torch.cat((lstm_output[:,-1,:256], lstm_output[:,0, 256:]), dim=-1)
            linear_output = self.linear(hidden.view(-1,256*2))
            # linear_output = self.linear(lstm_output[:-1])
       
        return linear_output


