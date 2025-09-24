# Onboarding a CausalLM Model

## Prerequisites

Before beginning the onboarding process, ensure you have `qefficient-transformers` library installed in editable mode.


## Introduction

This README provides a step-by-step guide on how to on-board a CausalLM model. The process includes setting up the environment, modifying configuration files, and running the model.
We will use a dummy model named `Blueprint` as an example and walk through the changes that have to be provided in the `qefficient_transformers` library to enable such a model.


## Step 1: Checking the classes in the modeling file from `transformers`library

1. **Look into the original modelling files in the `transformers` library:**
    - Locate the original model in the `transformers` library.
        - `/src/transformers/models/blueprint/modeling_blueprint.py` has all the modeling classes used to construct the model.
    - Locate the `pytorch_transforms.py` file in `qefficient_transformers` to see if the corresponding classes are already implemented in `qefficient_transformers`.
        - It's a good reference point to see if some functionalities have already been used in a prior model.
    - Check the architecture class of the model that you want to on-board.
        - If it is not in `pytorch_transforms.py`, you will need to implement them and then map the class along with the other required classes in the `pytorch_transforms.py` file.

## Step 2: Creating the custom modeling file and mappings in pytorch_tranforms.py

1. **Adding the required modified modeling file in the `qefficient-transformers` library:**
    - For our example we will create the following directory :
        `/QEfficient/transformers/models/blueprint`
    - Then we will add the modeling and __init__ files in this directory.
    - The modeling file 'modeling_blueprint.py` will have all the necessary modified modeling classes.
    - The file has been annotated to explain why the changes are required for the model.

2. **Add the mapping to the corresponding classes in `pytorch_transforms.py` file:**
    - You will need to map the classes of the model to the ones in the `pytorch_transforms.py` file.
     - If you look into `dummy_pytorch_transforms.py` file, you can see an example case for our `Blueprint` model. 
     - Every Mapping Class serves a specific purpose :-
        - **CustomOpsTransform** 
            - This class has mapping for the RMSNorm class that we use for a model. 
            - Most of the models have the same RMSNorm, in case you need to change the RMSNorm classes, you will need to make changes in this class as we do for Gemma models.
            - To add your own custom RMSNorm class when required, you can add it in `QEfficient.customop` file.
        - **KVCacheTransform**
            - This class handles mappings for almost all models that use a KV cache and thus generate text.
            - All the custom classes that we define for a model, we add the mappings with the corresponding transformers class in this section.
            - For the exception of models that don't have their modeling files in transformers library, we create this mapping via a different mapping class called **KVCacheExternalModuleMapperTransform**
        - **KVCacheExternalModuleMapperTransform**
            - This class is used to map to a class that is not in the transformers library.
            - Here we don't perform a class replacement as we do in other mapping classes.
            - We simply map the class name of the original model and then we map the methods of those classes with the custom methods that we had defined in our custom modeling file in qefficient.

3. **Testing the implementation:**
    - Once the implementation is complete, you can test it via the following instructions
        - Go to the file `/efficient-transformers/tests/transformers/models/test_causal_lm_models.py` and add the appropriate model card for your model.
            - For example, for `Blueprint` model, we might have a model card like `Blueprint/Blueprint-70B` on huggingface.
        - Add the model card in the list `test_models_causal` and then run the test to ensure that the classes are mapped correctly.


## References
- [Hugging Face Transformers GitHub Repository](https://github.com/huggingface/transformers)
- [Qefficient Transformers GitHub Repository](https://github.com/quic/efficient-transformers)

