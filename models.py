import sys

import torch
import torch.nn as nn
from model_utils import sort_batch_by_length, SelfAttentiveSum, SimpleDecoder, MultiSimpleDecoder, CNN
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

sys.path.insert(0, './resources')
import constant

class Model(nn.Module):
  def __init__(self, args, answer_num):
    super(Model, self).__init__()
    self.output_dim = args.rnn_dim * 2
    self.mention_dropout = nn.Dropout(args.mention_dropout)
    self.input_dropout = nn.Dropout(args.input_dropout)
    self.dim_hidden = args.dim_hidden
    self.embed_dim = 300
    self.mention_dim = 300
    self.lstm_type = args.lstm_type
    self.enhanced_mention = args.enhanced_mention
    if args.enhanced_mention:
      self.head_attentive_sum = SelfAttentiveSum(self.mention_dim, 1)
      self.cnn = CNN()
      self.mention_dim += 50
    self.output_dim += self.mention_dim

    # Defining LSTM here.   
    self.attentive_sum = SelfAttentiveSum(args.rnn_dim * 2, 100)
    if self.lstm_type == "two":
      self.left_lstm = nn.LSTM(self.embed_dim, 100, bidirectional=True, batch_first=True)
      self.right_lstm = nn.LSTM(self.embed_dim, 100, bidirectional=True, batch_first=True)
    elif self.lstm_type == 'single':
      self.lstm = nn.LSTM(self.embed_dim + 50, args.rnn_dim, bidirectional=True,
                          batch_first=True)
      self.token_mask = nn.Linear(4, 50)
    self.loss_func = nn.BCEWithLogitsLoss()
    self.sigmoid_fn = nn.Sigmoid()
    self.goal = args.goal
    self.multitask = args.multitask

    if args.data_setup == 'joint' and args.multitask:
      print("Multi-task learning")
      self.decoder = MultiSimpleDecoder(self.output_dim)
    else:
      self.decoder = SimpleDecoder(self.output_dim, answer_num)

  def sorted_rnn(self, sequences, sequence_lengths, rnn):
    sorted_inputs, sorted_sequence_lengths, restoration_indices = sort_batch_by_length(sequences, sequence_lengths)
    packed_sequence_input = pack_padded_sequence(sorted_inputs,
                                                 sorted_sequence_lengths.data.tolist(),
                                                 batch_first=True)
    packed_sequence_output, _ = rnn(packed_sequence_input, None)
    unpacked_sequence_tensor, _ = pad_packed_sequence(packed_sequence_output, batch_first=True)
    return unpacked_sequence_tensor.index_select(0, restoration_indices)

  def rnn(self, sequences, lstm):
    outputs, _ = lstm(sequences)
    return outputs.contiguous()

  def define_loss(self, logits, targets, data_type):
    if not self.multitask or data_type == 'onto':
      loss = self.loss_func(logits, targets)
      return loss
    if data_type == 'wiki':
      gen_cutoff, fine_cutoff, final_cutoff = constant.ANSWER_NUM_DICT['gen'], constant.ANSWER_NUM_DICT['kb'], \
                                              constant.ANSWER_NUM_DICT[data_type]
    else:
      gen_cutoff, fine_cutoff, final_cutoff = constant.ANSWER_NUM_DICT['gen'], constant.ANSWER_NUM_DICT['kb'], None
    loss = 0.0
    comparison_tensor = torch.Tensor([1.0]).cuda()
    gen_targets = targets[:, :gen_cutoff]
    fine_targets = targets[:, gen_cutoff:fine_cutoff]
    gen_target_sum = torch.sum(gen_targets, 1)
    fine_target_sum = torch.sum(fine_targets, 1)
    
    if torch.sum(gen_target_sum.data) > 0:
      gen_mask = torch.squeeze(torch.nonzero(torch.min(gen_target_sum.data, comparison_tensor)), dim=1)
      gen_logit_masked = logits[:, :gen_cutoff][gen_mask, :]
      gen_target_masked = gen_targets.index_select(0, gen_mask)
      gen_loss = self.loss_func(gen_logit_masked, gen_target_masked)
      loss += gen_loss 
    if torch.sum(fine_target_sum.data) > 0:
      fine_mask = torch.squeeze(torch.nonzero(torch.min(fine_target_sum.data, comparison_tensor)), dim=1)
      fine_logit_masked = logits[:,gen_cutoff:fine_cutoff][fine_mask, :]
      fine_target_masked = fine_targets.index_select(0, fine_mask)
      fine_loss = self.loss_func(fine_logit_masked, fine_target_masked)
      loss += fine_loss 

    if not data_type == 'kb':
      if final_cutoff:
        finer_targets = targets[:, fine_cutoff:final_cutoff]
        logit_masked = logits[:, fine_cutoff:final_cutoff]
      else:
        logit_masked = logits[:, fine_cutoff:]
        finer_targets = targets[:, fine_cutoff:]
      if torch.sum(torch.sum(finer_targets, 1).data) >0:
        finer_mask = torch.squeeze(torch.nonzero(torch.min(torch.sum(finer_targets, 1).data, comparison_tensor)), dim=1)
        finer_target_masked = finer_targets.index_select(0, finer_mask)
        logit_masked = logit_masked[finer_mask, :]
        layer_loss = self.loss_func(logit_masked, finer_target_masked)
        loss += layer_loss
    return loss

  def forward(self, feed_dict, data_type):
    if self.lstm_type == 'two':
      left_outputs = self.rnn(self.input_dropout(feed_dict['left_embed']), self.left_lstm)
      right_outputs = self.rnn(self.input_dropout(feed_dict['right_embed']), self.right_lstm)
      context_rep = torch.cat((left_outputs, right_outputs), 1)
      context_rep, _ = self.attentive_sum(context_rep)
    elif self.lstm_type == 'single':
      token_mask_embed = self.token_mask(feed_dict['token_bio'].view(-1, 4))
      token_mask_embed = token_mask_embed.view(feed_dict['token_embed'].size()[0], -1, 50)
      token_embed = torch.cat((feed_dict['token_embed'], token_mask_embed), 2)
      token_embed = self.input_dropout(token_embed)
      context_rep = self.sorted_rnn(token_embed, feed_dict['token_seq_length'], self.lstm)
      context_rep, _ = self.attentive_sum(context_rep)
    # Mention Representation
    if self.enhanced_mention:
      mention_embed, _ = self.head_attentive_sum(feed_dict['mention_embed'])
      span_cnn_embed = self.cnn(feed_dict['span_chars'])
      mention_embed = torch.cat((span_cnn_embed, mention_embed), 1)
    else:
      mention_embed = torch.sum(feed_dict['mention_embed'], dim=1)
    mention_embed = self.mention_dropout(mention_embed)
    output = torch.cat((context_rep, mention_embed), 1)
    logits = self.decoder(output, data_type)
    loss = self.define_loss(logits, feed_dict['y'], data_type)
    return loss, logits
