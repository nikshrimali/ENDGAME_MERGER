# Chatbot Using Transformers

In this notebook, we will train a simple chatbot using movie
scripts from the [Cornell Movie-Dialogs
Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html).

Conversational models are a hot topic in artificial intelligence
research. Chatbots can be found in a variety of settings, including
customer service applications and online helpdesks. These bots are often
powered by retrieval-based models, which output predefined responses to
questions of certain forms. In a highly restricted domain like a
company’s IT helpdesk, these models may be sufficient, however, they are
not robust enough for more general use-cases. Teaching a machine to
carry out a meaningful conversation with a human in multiple domains is
a research question that is far from solved. Recently, the deep learning
boom has allowed for powerful generative models like Google’s [Neural
Conversational Model](https://arxiv.org/abs/1506.05869), which marks
a large step towards multi-domain generative conversational models. In
this notebook, we will implement this kind of model in PyTorch.



- Handle loading and preprocessing of [Cornell Movie-Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) dataset
- Refer the code from this [Pytorch tutorial](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html) for integrating the dataset. 
- GRU based encoder-decoder architecture mentioned in the PyTorch code. However, we will be reimplementing above code using transformers to make it into a chatbot
- Achieve a loss of less than 2.4 in any epoch.  
- the same loss function/methodology as used in the PyTorch sample code will be used, else your evaluation would not be comparable and 2.4 doesn't make sense 


## Training Log

    Epoch: 01 | Time: 0m 27s
      Train Loss: 4.029 | Train PPL:  56.222
       Val. Loss: 3.691 |  Val. PPL:  40.100
    Epoch: 02 | Time: 0m 27s
      Train Loss: 3.549 | Train PPL:  34.768
       Val. Loss: 3.580 |  Val. PPL:  35.871
    Epoch: 03 | Time: 0m 28s
      Train Loss: 3.376 | Train PPL:  29.262
       Val. Loss: 3.559 |  Val. PPL:  35.128
    Epoch: 04 | Time: 0m 27s
      Train Loss: 3.245 | Train PPL:  25.658
       Val. Loss: 3.552 |  Val. PPL:  34.876
    Epoch: 05 | Time: 0m 27s
      Train Loss: 3.134 | Train PPL:  22.956
       Val. Loss: 3.561 |  Val. PPL:  35.187
    Epoch: 06 | Time: 0m 27s
      Train Loss: 3.035 | Train PPL:  20.803
       Val. Loss: 3.589 |  Val. PPL:  36.185
    Epoch: 07 | Time: 0m 28s
      Train Loss: 2.946 | Train PPL:  19.025
       Val. Loss: 3.608 |  Val. PPL:  36.904
    Epoch: 08 | Time: 0m 26s
      Train Loss: 2.866 | Train PPL:  17.565
       Val. Loss: 3.644 |  Val. PPL:  38.252
    Epoch: 09 | Time: 0m 27s
      Train Loss: 2.792 | Train PPL:  16.319
       Val. Loss: 3.702 |  Val. PPL:  40.547
    Epoch: 10 | Time: 0m 27s
      Train Loss: 2.728 | Train PPL:  15.301
       Val. Loss: 3.725 |  Val. PPL:  41.466
    Epoch: 11 | Time: 0m 27s
      Train Loss: 2.671 | Train PPL:  14.460
       Val. Loss: 3.775 |  Val. PPL:  43.577
    Epoch: 12 | Time: 0m 26s
      Train Loss: 2.617 | Train PPL:  13.693
       Val. Loss: 3.820 |  Val. PPL:  45.608
    Epoch: 13 | Time: 0m 27s
      Train Loss: 2.571 | Train PPL:  13.073
       Val. Loss: 3.835 |  Val. PPL:  46.292
    Epoch: 14 | Time: 0m 27s
      Train Loss: 2.526 | Train PPL:  12.503
       Val. Loss: 3.879 |  Val. PPL:  48.358
    Epoch: 15 | Time: 0m 28s
      Train Loss: 2.489 | Train PPL:  12.051
       Val. Loss: 3.915 |  Val. PPL:  50.165
    Epoch: 16 | Time: 0m 27s
      Train Loss: 2.454 | Train PPL:  11.640
       Val. Loss: 3.935 |  Val. PPL:  51.156
    Epoch: 17 | Time: 0m 27s
      Train Loss: 2.420 | Train PPL:  11.243
       Val. Loss: 3.976 |  Val. PPL:  53.291
    Epoch: 18 | Time: 0m 27s
      Train Loss: 2.389 | Train PPL:  10.900
       Val. Loss: 3.988 |  Val. PPL:  53.970
    Epoch: 19 | Time: 0m 27s
      Train Loss: 2.359 | Train PPL:  10.583
       Val. Loss: 4.038 |  Val. PPL:  56.728
    Epoch: 20 | Time: 0m 27s
      Train Loss: 2.335 | Train PPL:  10.327
       Val. Loss: 4.068 |  Val. PPL:  58.451


## Chatbot output
    > hii
    Bot: you re not gon na be seein me shortly .
    > why?
    Bot: yeah .
    > what's the matter?
    Bot: yes sir .
    > what happened?
    Bot: it s not a short word .
    > thats ok, tell me
    Bot: what ?
    > you said i am not gon na see you
    Bot: it s a .
    > ok, bye then
    Bot: it s a combat action .
    > ok
    Bot: what ?
    > bye
    Bot: hey .
    > i quit
    Bot: you re not gon na be seein me shortly .
    > quit
