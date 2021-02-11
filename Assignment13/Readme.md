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



