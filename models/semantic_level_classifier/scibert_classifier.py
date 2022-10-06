import os
import sys
import json
import argparse
from sklearn.model_selection import KFold

import torch
import torchmetrics
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn.modules.loss import BCELoss
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


class AltTextSentenceDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Example data entry
        {
          "corpus_id": 14745329,
          "sent_id": 0,
          "text": "A graph of the latencies for each transcript (professional, automatic and crowd).",
          "labels": [
            1,
            0,
            0,
            0
          ]
        }
        """
        current_item = self.data[idx]
        text = current_item['text']
        token_ids = self.tokenizer.encode(text, max_length=512, truncation=True)
        labels = current_item['labels']
        labels_float = [float(l) for l in labels]

        return {
            "text": token_ids,
            "labels": labels,
            "labels_float": labels_float
        }

    @staticmethod
    def collate_fn(data):
        token_ids = [torch.tensor(entry["text"]) for entry in data]
        labels = [torch.tensor(entry["labels"]) for entry in data]
        labels_float = [torch.tensor(entry["labels_float"]) for entry in data]
        token_ids_tensor = torch.nn.utils.rnn.pad_sequence(token_ids, batch_first=True, padding_value=0)
        labels_tensor = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        labels_float_tensor = torch.nn.utils.rnn.pad_sequence(labels_float, batch_first=True, padding_value=-100)
        return {
            "input_ids": token_ids_tensor,
            "labels": labels_tensor,
            "labels_float": labels_float_tensor
        }


class DataModule(LightningDataModule):
    def __init__(
            self,
            train_file: str,
            val_file: str,
            model_name_or_path: str,
            max_seq_length: int = 512,
            batch_size: int = 32,
            **kwargs,
    ):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, use_fast=True, max_length=max_seq_length
        )

    def setup(self, stage="fit"):
        def load_data(path):
            data = []
            with open(path) as fin:
                for line in fin:
                    data.append(json.loads(line))
            return data

        train_data = load_data(self.train_file)
        val_data = load_data(self.val_file)
        self.train_dataset = AltTextSentenceDataset(train_data, self.tokenizer)
        self.val_dataset = AltTextSentenceDataset(val_data, self.tokenizer)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=AltTextSentenceDataset.collate_fn,
            num_workers=2
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=AltTextSentenceDataset.collate_fn,
            num_workers=2
        )


class TransformerModule(LightningModule):
    def __init__(
            self,
            model_name_or_path: str,
            num_labels: int = 4,
            learning_rate: float = 3e-5,
            adam_epsilon: float = 1e-8,
            warmup_steps: int = 0,
            weight_decay: float = 0.0,
            max_seq_length: int = 512,
            batch_size: int = 32,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        self.classifier = torch.nn.Linear(768, 4)
        self.sigmoid = torch.nn.Sigmoid()
        self.loss_fn = BCELoss()
        self.metric_acc = torchmetrics.Accuracy()
        self.metric_f1 = torchmetrics.F1()
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True, max_length=max_seq_length
        )

    def forward(self, **inputs):
        # inputs['input_ids'].shape -> [batch_size, max_len]
        output = self.model(inputs["input_ids"])
        # cls_output_state.shape -> [batch_size, 768]
        cls_output_state = output["last_hidden_state"][inputs["input_ids"] == self.tokenizer.cls_token_id]
        # logits.shape -> [batch_size, num_labels] -> [num_labels * batch_size]
        logits = self.classifier(cls_output_state)
        probs = self.sigmoid(logits)
        probs_flat = probs.view(-1)
        # labels_flat.shape -> [num_labels * batch_size]
        labels = inputs["labels"]
        labels_float = inputs["labels_float"]
        labels_flat = labels_float.view(-1)
        loss = self.loss_fn(probs_flat, labels_flat)
        return loss, probs, labels

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        acc = self.metric_acc(outputs[1].view(-1), outputs[2].view(-1))
        self.log("loss", loss)
        self.log("acc", acc)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, probs, labels = outputs
        preds = torch.round(probs)
        return {"loss": val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss)
        val_acc = self.metric_acc(preds.view(-1), labels.view(-1))
        val_f1 = self.metric_f1(preds.view(-1), labels.view(-1))
        self.log("val_acc", val_acc, prog_bar=True)
        self.log("val_f1", val_f1, prog_bar=True)
        return loss

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return
        train_loader = self.train_dataloader()
        tb_size = self.hparams.batch_size * max(1, self.trainer.gpus if self.trainer.gpus else 0)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


def train(model, train_file, val_file, outdir, logname, device_type, devices):
    dm = DataModule(train_file=train_file, val_file=val_file, model_name_or_path=model)
    dm.setup(stage="fit")
    model = TransformerModule(warmup_steps=200, model_name_or_path=model)
    logger = TensorBoardLogger(outdir, name=logname)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(outdir, 'checkpoints'),
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    trainer = Trainer(
        accelerator=device_type,
        devices=devices,
        progress_bar_refresh_rate=5,
        max_epochs=16,
        default_root_dir=outdir,
        logger=logger,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, dm)
    trainer.validate(model, dm.val_dataloader())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Name of model")
    parser.add_argument("--data", type=str, help="Path to data directory")
    parser.add_argument("--outdir", type=str, help="Path to save output")
    args = parser.parse_args()

    model_name = args.model
    data_dir = args.data
    out_dir = os.path.join(args.outdir, model_name.replace('/', '_'))

    if not os.path.exists(data_dir):
        print('Data path does not exist!')
        sys.exit(-1)
    os.makedirs(out_dir, exist_ok=True)

    # check if GPUs available
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        device_type = 'cpu'
        devices = None
    else:
        device_type = 'gpu'
        devices = [0]

    # get folds
    for i in range(5):
        print(f'Fold {i}')
        train_file = os.path.join(data_dir, f'{i:02d}', 'train.jsonl')
        val_file = os.path.join(data_dir, f'{i:02d}', 'val.jsonl')
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"{train_file} not found!")
        if not os.path.exists(val_file):
            raise FileNotFoundError(f"{val_file} not found!")
        out_subdir = os.path.join(out_dir, f'fold_{i:02d}')
        os.makedirs(out_subdir, exist_ok=True)
        logger_name = 'logs'
        train(model_name, train_file, val_file, out_subdir, logger_name, device_type, devices)

    print('done.')