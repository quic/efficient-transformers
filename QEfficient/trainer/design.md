Onnx Transform Pipeline:

+-------------------+     +-------------------+     +-------------------+
|   ONNX Model      |     |  Transform Queue  |     | Transformed Model |
+-------------------+     +-------------------+     +-------------------+
|                   |     | - FP16Clip        |     |                   |
|   Original ONNX   | --> | - SplitWeights    | --> |   Modified ONNX   |
|      Model        |     | - LoraAdapters    |     |      Model        |
|                   |     | - FixTrainingOps  |     |                   |
+-------------------+     | - CustomTransform |     +-------------------+
                          +-------------------+
                                    ^
                                    |
                          +-------------------+
                          |  New Transform    |
                          +-------------------+
                          | Easy to add new   |
                          | transforms here   |
                          +-------------------+


QEffRefine Pipeline:

+-------------------+     +-------------------+     +-------------------+
|    QEffConfig     |     | QEffTrainingArgs  |     |    QEfficient     |
+-------------------+     +-------------------+     +-------------------+
| - model_id        |     | - output_dir      |     | - config          |
| - dataset_name    |     | - num_train_epochs|     | - data_manager    |
| - train_frac      |     | - batch_size      |     | - model_manager   |
| - lora_config     |     | - learning_rate   |     | - trainer         |
| - max_ctx_len     |     | ...               |     +-------------------+
+-------------------+     +-------------------+     | + refine()        |
                                                    | + generate()      |
                                                    +-------------------+

                +--------------------+--------------------+  
                |                    |                    |  
    +-----------v----------+ +-------v--------+  +--------v-------+
    |   QEffDataManager    | | QEffModelManager|  |  QEffTrainer   |
    +----------------------+ +----------------+  +----------------+
    | - prepare_dataset()  | | - init_model() |  | - train()      |
    | - get_dataloader()   | | - prepare_for_ |  | - evaluate()   |
    | - collate()          |   training()     |  | - save_model() |
    +----------------------+ | - generate()   |  +----------------+
                             +----------------+
                                     
                      +---------------+---------------+
                      |                               |
            +---------v---------+           +---------v---------+
            |     ONNXModel     |           |   QAICCompiler    |
            +-------------------+           +-------------------+
            | - export()        |           | - compile()       |
            | - modify()        |           +-------------------+
            | - validate()      |
            | - save()          |           +-------------------+
            +-------------------+           |    QAICLoader     |
                     |                       +-------------------+
                     |                       | - load_model()    |
                     |                       | - get_session()   |
                     v                       +-------------------+
        +-------------------------+
        |     ONNX Transforms     |
        +-------------------------+
        | - FP16Clip              |
        | - SplitWeights          |
        | - LoraAdapters          |
        | - FixTrainingOps        |
        +-------------------------+


QEffRefine Workflow:

+-------------+     +----------------+     +------------------+
| User Input  | --> | QEfficient     | --> | QEffDataManager  |
+-------------+     | (refine)       |     +------------------+
                    +----------------+              |
                            |                       v
                            |            +----------------------+
                            |            | Prepare Dataset      |
                            |            +----------------------+
                            |                       |
                            v                       v
                  +-------------------+    +------------------+
                  | QEffModelManager  | <- | Get Dataloader   |
                  +-------------------+    +------------------+
                            |
                            v
                  +----------------------+
                  | Initialize Model     |
                  | (Pytorch Transforms) |
                  +----------------------+
                            |
                            v
                  +--------------------+
                  | Generate Artifacts |
                  +--------------------+
                            |
                            v
                  +-------------------+
                  | Export to ONNX    |
                  +-------------------+
                            |
                            v
                  +-------------------+
                  | Modify/Fix ONNX   |
                  +-------------------+
                            |
                            v
                  +-------------------+
                  | Compile for QAIC  |
                  +-------------------+
                            |
                            v
                  +-------------------+
                  | Load QAIC Model   |
                  +-------------------+
                            |
                            v
                  +-------------------+
                  | QEffTrainer       |
                  | (train)           |
                  +-------------------+
                            |
                            v
                  +-------------------+
                  | Save Checkpoint   |
                  +-------------------+


Custom Dataset Integration Diagram:
+-------------------+     +-------------------+     +-------------------+
|   Custom Dataset  |     | QEffDataManager   |     |    QEffTrainer    |
+-------------------+     +-------------------+     +-------------------+
| - load_data()     |     | - prepare_dataset()|    | - train()         |
| - preprocess()    | --> | - get_dataloader() | --> | - evaluate()     |
| - split()         |     | - collate()        |    | - save_model()    |
+-------------------+     +-------------------+     +-------------------+
        ^
        |
+-------------------+
| User-defined      |
| Dataset Class     |
+-------------------+


Modular Architecture Overview:
+------------------+    +------------------+    +------------------+
|  Configuration   |    |  Data Pipeline   |    | Model Pipeline   |
+------------------+    +------------------+    +------------------+
| - QEffConfig     |    | - Custom Dataset |    | - Model Init     |
| - TrainingArgs   | -> | - DataManager    | -> | - ONNX Export    |
|                  |    | - Dataloader     |    | - ONNX Transforms|
+------------------+    +------------------+    | - QAIC Compile   |
         |                       |              +------------------+
         |                       |                       |
         v                       v                       v
+------------------+    +------------------+    +------------------+
|   QEfficient     |    |   QEffTrainer    |    |  Model Outputs   |
+------------------+    +------------------+    +------------------+
| - refine()       | -> | - train()        | -> | - Refined Model  |
| - generate()     |    | - evaluate()     |    | - Tokenizer      |
+------------------+    +------------------+    +------------------+

Directory Structure::

QEfficient/Finetuning
│
├── __init__.py
├── config.py
├── data_manager.py
├── model_manager.py
├── models.py
├── onnx_utils.py
├── qaic_utils.py
├── refine.py
├── trainer.py
└── utils.py
