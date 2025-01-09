from transformers import(
    AutoTokenizer,
    default_data_collator
)
from torch.utils.data import DataLoader
from configs.squad_config import ConfigModel

class Preprocessing():
  def __init__(self, model_tokenizer=ConfigModel.MODEL_TOKENIZER,
                batch_size=ConfigModel.BATCH_SIZE,
                max_input_length=ConfigModel.MAX_INPUT_LENGTH,
                stride=ConfigModel.STRIDE,
                model=None,
                dataset=None,
                flag_training=True):
    self.tokenizer = AutoTokenizer.from_pretrained(model_tokenizer)
    self.max_input_length = max_input_length
    self.stride = stride
    if flag_training:
      print("-"*50, "Information of Tokenizer", "-"*50)
      print(self.tokenizer)
      print("-"*50, "Information of Tokenizer", "-"*50)

      self.tokenized_dataset_train = self.map_tokenize_dataset_train(dataset)
      self.tokenized_dataset_val = self.map_tokenize_dataset_val(dataset)
      self.train_loader, self.val_loader = self.data_loader(batch_size)

  def tokenize_dataset_train(self, examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = self.tokenizer(
        questions,
        examples["context"],
        max_length=self.max_input_length,
        truncation="only_second",
        stride=self.stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Tìm điểm bắt đầu và kết thúc của ngữ cảnh
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # Nếu câu trả lời không hoàn toàn nằm trong ngữ cảnh, nhãn là (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Nếu không nó sẽ là vị trí token bắt đầu và kết thúc
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

  def map_tokenize_dataset_train(self, dataset):
    print(type(dataset))
    train_dataset = dataset["train"].shuffle().select(range(10000)).map(
      self.tokenize_dataset_train,
      batched=True,
      remove_columns=dataset["train"].column_names
    )
    #train_dataset.save_to_disk("/kaggle/working/data")
    return train_dataset

  def tokenize_dataset_val(self, examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = self.tokenizer(
        questions,
        examples["context"],
        max_length=self.max_input_length,
        truncation="only_second",
        stride=self.stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs

  def map_tokenize_dataset_val(self, dataset):
    validation_dataset = dataset["validation"].shuffle().select(range(1000)).map(
      self.tokenize_dataset_val,
      batched=True,
      remove_columns=dataset["validation"].column_names,
    )
    #validation_dataset.save_to_disk("/kaggle/working/data")
    return validation_dataset


  def data_loader(self, batch_size):
    train_loader = DataLoader(
      self.tokenized_dataset_train,
      shuffle=True,
      batch_size=batch_size,
      collate_fn=default_data_collator,
    )
    tokenized_dataset_val= self.tokenized_dataset_val.remove_columns(["example_id", "offset_mapping"])
    tokenized_dataset_val.set_format("torch")
    val_loader = DataLoader(
      tokenized_dataset_val,
      batch_size=batch_size,
      collate_fn=default_data_collator,
    )

    return train_loader, val_loader

