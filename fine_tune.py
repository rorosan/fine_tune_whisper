#Create an environment and install all the packaages in requirments
# git clone https://github.com/libsndfile/libsndfile.git
# cd libsndfile/
# sudo apt install autoconf autogen automake build-essential libasound2-dev   libflac-dev libogg-dev libtool libvorbis-dev libopus-dev libmp3lame-dev   libmpg123-dev pkg-config python3
# autoreconf -vif
# ./configure --enable-werror
# sudo make
# sudo make check
# sudo mkdir /usr/local/lib/python3.8/dist-packages/_soundfile_data/

from huggingface_hub.hf_api import HfFolder
HfFolder.save_token('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')

from datasets import load_dataset, DatasetDict

common_voice = DatasetDict()

common_voice["train"] = load_dataset(
    "mozilla-foundation/common_voice_13_0", "ur", split="train+validation"
)
common_voice["test"] = load_dataset(
    "mozilla-foundation/common_voice_13_0", "ur", split="test"
)

print(common_voice)

common_voice = common_voice.select_columns(["audio", "sentence"])

from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="urdu", task="transcribe"
)

common_voice["train"].features

from datasets import Audio

sampling_rate = processor.feature_extractor.sampling_rate
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=sampling_rate))

def prepare_dataset(example):
    audio = example["audio"]
    example = processor(audio=audio["array"],sampling_rate=audio["sampling_rate"],text=example["sentence"],)
    example["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    return example

common_voice = common_voice.map(
    prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1
)

max_input_length = 30.0

def is_audio_in_length_range(length):
    return length < max_input_length

common_voice["train"] = common_voice["train"].filter(
    is_audio_in_length_range,
    input_columns=["input_length"],
)

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"][0]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

import evaluate

metric = evaluate.load("wer")

from transformers.models.whisper.english_normalizer import BasicTextNormalizer

normalizer = BasicTextNormalizer()


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)
    pred_str_norm = [normalizer(pred) for pred in pred_str]
    label_str_norm = [normalizer(label) for label in label_str]
    pred_str_norm = [pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0]
    label_str_norm = [label_str_norm[i] for i in range(len(label_str_norm)) if len(label_str_norm[i]) > 0 ]
    wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)
    return {"wer_ortho": wer_ortho, "wer": wer}

from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

from functools import partial
model.config.use_cache = False
model.generate = partial(model.generate, language="urdu", task="transcribe", use_cache=True)


from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-ur",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=50,
    max_steps=500,
    gradient_checkpointing=True,
    fp16=True,
    fp16_full_eval=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

trainer.train()

from datasets import load_dataset
from datasets import Audio

minds = load_dataset("mozilla-foundation/common_voice_13_0", name="ur", split="train")
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))

example = minds[0]
print(example["sentence"])

from transformers import pipeline
pipe = pipeline(task="automatic-speech-recognition", model="./whisper-small-ur")
print(pipe(example["audio"]["array"]))
