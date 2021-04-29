# Raug

Raug is a simple pipeline to train deep neural models using Pytorch. Those are the main steps you should take to use this pipeline:

1. Loading the dataset
To load your dataset you must call [get_data_loader()](https://github.com/paaatcha/raug/blob/6752e811b4a3367252881bfdc394e96b6beff359/loader.py#L87) function, which will create the dataloader.

2. Create/Define a model
You must create/define a model. The easiest way is to load a model from `torchvision`, for example:
```
from torchvision import models
my_model = models.resnet50(pretrained=_pretrained)
``` 
3. Define the loss function and the optimizer
The third step is to define the loss function and the optimizer. Again, you can call them from Pytorch. Example:
```
import torch.optim as optim
import torch.nn as nn
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(my_model.parameters(), lr=0.001)
```
4. Training the model
Now, to train the model, you must call the function [fit_model()](https://github.com/paaatcha/raug/blob/6752e811b4a3367252881bfdc394e96b6beff359/train.py#L130). 

5. Testing the model
Finally, to test the model in a test/validation partition, you just call the function [test_model()](https://github.com/paaatcha/raug/blob/6752e811b4a3367252881bfdc394e96b6beff359/eval.py#L124)


## Example
You may see an example of the use of this package in the [ISIC script](https://github.com/paaatcha/MetaBlock/blob/main/benchmarks/isic/isic.py) in which I use it to train CNNs to classify skin cancer.

____

For all functions/methods it is included the documentation to described each parameter. Please, refer to them to understand the parameters properly. Also, in [Utils](https://github.com/paaatcha/raug/tree/master/utils) folder you find some codes to compute metrics, to load, or to use a telegram bot to follow the training phase. 

## Dependencies
To install the dependecies you just need to run `pip install -r requirements.txt`.

## What is Raug?
Well, I'm a fan of J. R. R. Tolkien (the author of The Lord of the rings and The Hobbit) who creates some Elf languages. Raug means a powerful creature in [Sindarin](https://www.jrrvf.com/hisweloke/sindar/online/sindar/dict-en-sd.html). As I'm not that creative to create names to my codes, I just choose some elf names.
