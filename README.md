# SC4002 NLP Project

## How to install

1. Create a virtual environment with either conda or venv or any environment manager of your choice
   * We try to use python 3.12, but if needed, we will up/downgrade python
2. Run pip install -r requirements.txt
3. Create a .env file in the root folder, and add in the following:

```
MODEL_PATH=<YOUR FOLDER PATH HERE>/modelfiles/
```

## Install pytorch

1. Go to this [LINK](https://pytorch.org/get-started/locally/) and select the correct options for your device
   * We try to use CUDA 12.4 and pip where possible

## File structure

* Note: you will have to create these folders to store your files
* Will NOT be uploaded into git:
  * /csv - holds all working csv files
  * /notebooks - holds notebooks. these are the testing notebooks that we may need individually, but may not need to be uploaded to keep the git clean
  * /modelfiles - holds model files. these are more of the intermediary files that may be generated. if the model is the final one that we know we are using, then copy it into /datapool/modelfiles to upload to git
* Will be uploaded into git:
  * /utils - stores utilities (functions, model structure, etc)
  * /datapool - stores bulk data that is to be pushed to git (model files, datasets, etc)
