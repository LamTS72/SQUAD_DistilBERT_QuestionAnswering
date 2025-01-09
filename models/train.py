
from transformers import (
    get_scheduler,
)
import evaluate
import torch
import os
import numpy as np
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from huggingface_hub import Repository, HfApi, HfFolder
import math
from torch.nn.utils.rnn import pad_sequence
from nltk.tokenize import sent_tokenize
import collections
from configs.squad_config import ConfigDataset, ConfigModel, ConfigHelper
from squad_model import CustomModel
from preprocessing import Preprocessing
from data.custom_data import CustomDataset
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Used Device: ", device)

class Training():
    def __init__(self, model_name=ConfigModel.MODEL_TOKENIZER,
                 learning_rate=ConfigModel.LEARNING_RATE,
                 epoch=ConfigModel.EPOCHS,
                 num_warmup_steps=ConfigModel.NUM_WARMUP_STEPS,
                 name_metric=ConfigModel.METRICS,
                 path_tensorboard=ConfigModel.PATH_TENSORBOARD,
                 path_save=ConfigModel.PATH_SAVE,
                 dataset=None,
                 process=None
                ):
        self.dataset = dataset
        self.process = process
        self.model = CustomModel(model_name).model
        self.epochs = epoch
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )
        self.lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=(self.epochs * len(self.process.train_loader))
        )
        self.metric = evaluate.load(name_metric)
        self.writer = SummaryWriter(path_tensorboard)

        # Define necessary variables
        self.api = HfApi(token=ConfigHelper.TOKEN_HF)
        self.repo_name = path_save  # Replace with your repo name
        self.author = ConfigHelper.AUTHOR
        self.repo_id = self.author + "/" + self.repo_name
        self.token = HfFolder.get_token()
        self.repo = self.setup_hf_repo(self.repo_name, self.repo_id, self.token)

    def setup_hf_repo(self, local_dir, repo_id, token):
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        try:
            self.api.repo_info(repo_id)
            print(f"Repository {repo_id} exists. Cloning...")
        except Exception as e:
            print(f"Repository {repo_id} does not exist. Creating...")
            self.api.create_repo(repo_id=repo_id, token=token, private=True)

        repo = Repository(local_dir=local_dir, clone_from=repo_id)
        return repo

    def save_and_upload(self, epoch, final_commit=False):
        # Save model, tokenizer, and additional files
        for param in self.model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        self.model.save_pretrained(self.repo_name)
        self.process.tokenizer.save_pretrained(self.repo_name)

        # Push to Hugging Face Hub
        self.repo.git_add(pattern=".")
        commit_message = "Final Commit: Complete fine-tuned model" if final_commit else f"Epoch {epoch}: Update fine-tuned model and metrics"
        self.repo.git_commit(commit_message)
        self.repo.git_push()

        print(f"Model and files pushed to Hugging Face Hub for epoch {epoch}: {self.repo_id}")

    def compute_metrics(self, start_logits, end_logits, features, examples):
      n_best = 20
      max_answer_length = 30
      example_to_features = collections.defaultdict(list)
      for idx, feature in enumerate(features):
          example_to_features[feature["example_id"]].append(idx)

      predicted_answers = []
      for example in tqdm(examples):
          example_id = example["id"]
          context = example["context"]
          answers = []

          # Lặp qua tất cả các đặc trưng liên quan tới mẫu đó
          for feature_index in example_to_features[example_id]:
              start_logit = start_logits[feature_index]
              end_logit = end_logits[feature_index]
              offsets = features[feature_index]["offset_mapping"]

              start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
              end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
              for start_index in start_indexes:
                  for end_index in end_indexes:
                      # Bỏ qua câu trả lời không xuất hiện hoàn toàn trong ngữ cảnh
                      if offsets[start_index] is None or offsets[end_index] is None:
                          continue
                      # Bỏ qua những câu trả lời với độ dài < 0 hoặc > max_answer_length
                      if (
                          end_index < start_index
                          or end_index - start_index + 1 > max_answer_length
                      ):
                          continue

                      answer = {
                          "text": context[offsets[start_index][0] : offsets[end_index][1]],
                          "logit_score": start_logit[start_index] + end_logit[end_index],
                      }
                      answers.append(answer)

          # Chọn câu trả lời có điểm cao nhất
          if len(answers) > 0:
              best_answer = max(answers, key=lambda x: x["logit_score"])
              predicted_answers.append(
                  {"id": example_id, "prediction_text": best_answer["text"]}
              )
          else:
              predicted_answers.append({"id": example_id, "prediction_text": ""})

      theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
      return self.metric.compute(predictions=predicted_answers, references=theoretical_answers)

    def fit(self, flag_step=False):
        progress_bar = tqdm(range(self.epochs * len(self.process.train_loader)))
        interval = 200
        for epoch in range(self.epochs):
            # training
            self.model.train()
            n_train_samples = 0
            total_train_loss = 0
            for i, batch in enumerate(self.process.train_loader):
                n_train_samples += len(batch)
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model.to(device)(**batch)
                losses = outputs.loss
                self.optimizer.zero_grad()
                losses.backward()
                total_train_loss += round(losses.item(),4)
                self.optimizer.step()
                self.lr_scheduler.step()

                progress_bar.update(1)
                if (i + 1) % interval == 0 and flag_step == True:
                    print("Epoch: {}/{}, Iteration: {}/{}, Train Loss: {}".format(
                        epoch + 1,
                        self.epochs,
                        i + 1,
                        len(self.process.train_loader),
                        losses.item())
                    )
                    self.writer.add_scalar('Train/Loss', round(losses.item(),4), epoch * len(self.process.train_loader) + i)

            # evaluate
            self.model.eval()
            start_logits = []
            end_logits = []
            for batch in self.process.val_loader:
                with torch.no_grad():
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = self.model.to(device)(**batch)

                    start_logits.append(outputs.start_logits.cpu().numpy())
                    end_logits.append(outputs.end_logits.cpu().numpy())

            start_logits = np.concatenate(start_logits)
            end_logits = np.concatenate(end_logits)
            start_logits = start_logits[: len(self.process.tokenized_dataset_val)]
            end_logits = end_logits[: len(self.process.tokenized_dataset_val)]

            self.metrics = self.compute_metrics(
                start_logits, end_logits, self.process.tokenized_dataset_val, self.dataset.raw_data["validation"]
            )
            print(f"epoch {epoch}:", self.metrics)


            epoch_train_loss = total_train_loss / n_train_samples
            print(f"train_loss: {epoch_train_loss}")


            # Save and upload after each epoch
            final_commit = ((epoch+1) == self.epochs)
            self.save_and_upload((epoch+1), final_commit)


if __name__ == '__main__':
    dataset = CustomDataset()
    process = Preprocessing(dataset=dataset.raw_data)
    train = Training(dataset=dataset,process=process)
    train.fit()