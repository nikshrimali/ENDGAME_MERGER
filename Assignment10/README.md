# Assignment 10

Attention with Packed Padded Sequences for min 20 epochs

PyTorch allows us to pack the sequence, internally packed sequence is a tuple of two lists. One contains the elements of sequences and other contains the length sequence. Elements are interleaved by time steps and other contains the size of each sequence the batch size at each step. This is helpful in recovering the actual sequences as well as telling RNN what is the batch size at each time step. This can be passed to RNN and it will internally optimize the computations.

## Datasets Trained
- <a href= "https://github.com/nikshrimali/ENDGAME_MERGER/blob/main/Assignment10/Assignment_10_Packed_Padded.ipynb">English to German Translation </a>
- <a href = "https://github.com/nikshrimali/ENDGAME_MERGER/blob/main/Assignment10/SQUAD_Attention_PADDED.ipynb"> SQUAD Dataset </a>

## Models Used
```python

Seq2Seq(
  (encoder): Encoder(
    (embedding): Embedding(2918, 256)
    (dropout): Dropout(p=0.5, inplace=False)
    (rnn): GRU(256, 512, bidirectional=True)
    (fc): Linear(in_features=1024, out_features=512, bias=True)
  )
  (decoder): Decoder(
    (attention): Attention(
      (attn): Linear(in_features=1536, out_features=512, bias=True)
      (v): Linear(in_features=512, out_features=1, bias=False)
    )
    (embedding): Embedding(502, 256)
    (dropout): Dropout(p=0.5, inplace=False)
    (rnn): GRU(1280, 512)
    (fc_out): Linear(in_features=1792, out_features=502, bias=True)
  )
)

```
## Results

```

Training started
training complete
1611250053.286074
Epoch: 01 | Time: 0m 6s
	Train Loss: 3.917 | Train PPL:  50.234
	 Val. Loss: 1.885 |  Val. PPL:   6.586
Training started
training complete
1611250059.6753592
Epoch: 02 | Time: 0m 6s
	Train Loss: 3.646 | Train PPL:  38.327
	 Val. Loss: 2.250 |  Val. PPL:   9.486
Training started
training complete
1611250065.9895358
Epoch: 03 | Time: 0m 6s
	Train Loss: 3.601 | Train PPL:  36.638
	 Val. Loss: 2.214 |  Val. PPL:   9.149
Training started
training complete
1611250072.3503861
Epoch: 04 | Time: 0m 6s
	Train Loss: 3.773 | Train PPL:  43.521
	 Val. Loss: 2.112 |  Val. PPL:   8.265
Training started
training complete
1611250078.7238147
Epoch: 05 | Time: 0m 6s
	Train Loss: 3.455 | Train PPL:  31.651
	 Val. Loss: 2.118 |  Val. PPL:   8.315
Training started
training complete
1611250085.0924964
Epoch: 06 | Time: 0m 6s
	Train Loss: 3.421 | Train PPL:  30.602
	 Val. Loss: 2.174 |  Val. PPL:   8.798
Training started
training complete
1611250091.4957643
Epoch: 07 | Time: 0m 6s
	Train Loss: 3.157 | Train PPL:  23.508
	 Val. Loss: 2.123 |  Val. PPL:   8.354
Training started
training complete
1611250097.9256227
Epoch: 08 | Time: 0m 6s
	Train Loss: 3.131 | Train PPL:  22.905
	 Val. Loss: 2.087 |  Val. PPL:   8.063
Training started
training complete
1611250104.339232
Epoch: 09 | Time: 0m 6s
	Train Loss: 2.929 | Train PPL:  18.704
	 Val. Loss: 2.091 |  Val. PPL:   8.096
Training started
training complete
1611250110.7697022
Epoch: 10 | Time: 0m 6s
	Train Loss: 2.976 | Train PPL:  19.607
	 Val. Loss: 2.143 |  Val. PPL:   8.524
Training started
training complete
1611250117.1950333
Epoch: 11 | Time: 0m 6s
	Train Loss: 3.072 | Train PPL:  21.575
	 Val. Loss: 2.172 |  Val. PPL:   8.776
Training started
training complete
1611250123.6300805
Epoch: 12 | Time: 0m 6s
	Train Loss: 4.803 | Train PPL: 121.862
	 Val. Loss: 2.138 |  Val. PPL:   8.485
Training started
training complete
1611250130.056084
Epoch: 13 | Time: 0m 6s
	Train Loss: 3.652 | Train PPL:  38.544
	 Val. Loss: 3.428 |  Val. PPL:  30.821
Training started
training complete
1611250136.4938004
Epoch: 14 | Time: 0m 6s
	Train Loss: 3.283 | Train PPL:  26.666
	 Val. Loss: 2.497 |  Val. PPL:  12.142
Training started
training complete
1611250142.9540548
Epoch: 15 | Time: 0m 6s
	Train Loss: 3.347 | Train PPL:  28.421
	 Val. Loss: 2.440 |  Val. PPL:  11.477
Training started
training complete
1611250149.4100351
Epoch: 16 | Time: 0m 6s
	Train Loss: 4.259 | Train PPL:  70.725
	 Val. Loss: 2.320 |  Val. PPL:  10.180
Training started
training complete
1611250155.881719
Epoch: 17 | Time: 0m 6s
	Train Loss: 4.377 | Train PPL:  79.598
	 Val. Loss: 2.892 |  Val. PPL:  18.029
Training started
training complete
1611250162.3550816
Epoch: 18 | Time: 0m 6s
	Train Loss: 4.533 | Train PPL:  93.051
	 Val. Loss: 2.792 |  Val. PPL:  16.319
Training started
training complete
1611250168.8399339
Epoch: 19 | Time: 0m 6s
	Train Loss: 3.510 | Train PPL:  33.452
	 Val. Loss: 2.167 |  Val. PPL:   8.732
Training started
training complete
1611250175.34098
Epoch: 20 | Time: 0m 6s
	Train Loss: 4.132 | Train PPL:  62.318
	 Val. Loss: 2.460 |  Val. PPL:  11.700
```
