from .config import QEffConfig, QEffTrainingArguments
from .data_utils import QEffDataManager
from .model_manager import QEffModelManager
from .trainer import QEffTrainer


class QEfficient:
    def __init__(self, config: QEffConfig):
        self.config = config
        self.data_manager = QEffDataManager(config)
        self.model_manager = QEffModelManager(config)
        self.trainer = None

    def refine(self, training_args: QEffTrainingArguments):
        train_dataset, eval_dataset = self.data_manager.prepare_dataset()
        self.model_manager.initialize_model()
        self.model_manager.prepare_for_training()

        self.trainer = QEffTrainer(
            model_manager=self.model_manager,
            data_manager=self.data_manager,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        self.trainer.train()
        return self.model_manager.get_model(), self.model_manager.get_tokenizer()

    def generate(self, prompt: str, max_length: int = 100):
        return self.model_manager.generate(prompt, max_length)
