from transformers import Trainer

class MultimodalTrainer(Trainer):

    def printOutput(self, outputs):
        tokens = outputs.logits.argmax(dim=-1)
        output = self.tokenizer.decode(
            tokens[0],
            skip_special_tokens=True
            )

    def printLabels(self, labels):
        output = self.tokenizer.decode(
            labels[0],
            skip_special_tokens=True
            )

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss