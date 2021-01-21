
# Assignment 9

Train Seq-to-Seq model and attention model on any of the 4 datasets from <a href="https://kili-technology.com/blog/chatbot-training-datasets/"> Here </a>

## Datasets worked


- SQUAD - Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.

  - <a href="https://github.com/nikshrimali/ENDGAME_MERGER/blob/main/Assignment9/SQUAD_SEQ2SEQ.ipynb">Seq-to-Seq model</a>
  - <a href="https://github.com/nikshrimali/ENDGAME_MERGER/blob/main/Assignment9/SQUAD-Attention.ipynb">Attention Model</a>
  
- Cornell-movie-dialog

  - <a href="https://github.com/nikshrimali/ENDGAME_MERGER/blob/main/Assignment9/Assignment_9_seq2seq-cornell-movie-dialog.ipynb">Seq-to-Seq model</a>
  - <a href="https://github.com/nikshrimali/ENDGAME_MERGER/blob/main/Assignment9/Assignment_9-seq2seq-with-attention-cornell-movie-dialog.ipynb">Attention Model</a>
  
- Twitter_cs

  - <a href=https://github.com/nikshrimali/ENDGAME_MERGER/blob/main/Assignment9/Assignment_9_seq2seq-twitter_cs.ipynb">Seq-to-Seq model</a>
  - <a href="https://github.com/nikshrimali/ENDGAME_MERGER/blob/main/Assignment9/Assignment_9-seq2seq-with-attention-twitter_cs.ipynb">Attention Model</a>
  
 - RecipieQA
  - <a href="https://github.com/nikshrimali/ENDGAME_MERGER/blob/main/Assignment9/SQUAD_SEQ2SEQ.ipynb">Seq-to-Seq model</a>
  - <a href="https://github.com/nikshrimali/ENDGAME_MERGER/blob/main/Assignment9/RecipieQA_ENDS9_M1.ipynb">Attention Model</a>
  
  # Model Summary
  
  - Seq to Seq model
    '''
    Seq2Seq(
    (encoder): Encoder(
      (embedding): Embedding(10004, 128)
      (rnn): GRU(128, 256)
      (dropout): Dropout(p=0.5, inplace=False)
    )
    (decoder): Decoder(
      (embedding): Embedding(10004, 128)
      (rnn): GRU(384, 256)
      (fc_out): Linear(in_features=640, out_features=10004, bias=True)
      (dropout): Dropout(p=0.5, inplace=False)))
    '''
  
- Attention Model
  '''
  Seq2Seq((encoder): Encoder(
    (embedding): Embedding(10004, 64)
    (rnn): GRU(64, 64, bidirectional=True)
    (fc): Linear(in_features=128, out_features=64, bias=True)
    (dropout): Dropout(p=0.5, inplace=False))
    
  (decoder): Decoder(
    (attention): Attention(
      (attn): Linear(in_features=192, out_features=64, bias=True)
      (v): Linear(in_features=64, out_features=1, bias=False))
    (embedding): Embedding(10004, 64)
    (rnn): GRU(192, 64)
    (fc_out): Linear(in_features=256, out_features=10004, bias=True)
    (dropout): Dropout(p=0.5, inplace=False)))
  
  '''
  
