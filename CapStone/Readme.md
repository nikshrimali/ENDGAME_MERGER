# Transformer based model to translate English text to Python code

The goal is to  write a transformer-based model that can translats English text to python code(with proper whitespace indentations)

The training dataset contains around 4600+ examples of English text to python code. 
- must use transformers with self-attention, multi-head, and scaled-dot product attention in the model
- There is no limit on the number of training epochs or total number of parameters in the model
- should have trained a separate embedding layer for python keywords and paid special attention to whitespaces, colon and other things (like comma etc)
- model should to do proper indentation
- model should to use newline properly
- model should understand how to use colon (:)
- model should generate proper python code that can run on a Python interpreter and produce proper results


Some preprocessing checks on the dataset should be carried out like:
- the dataset provided is divided into English and "python-code" pairs properly
- the dataset does not have anomalies w.r.t. indentations (like a mixed-use of tabs and spaces, or use of either 4 or 3 spaces, it should be 4 spaces only). Either use tabs only or 4 spaces only, not both
- the length of the "python-code" generated is not out of your model's capacity
