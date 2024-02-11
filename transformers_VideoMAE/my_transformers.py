from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from pytorchvideo.data import LabeledVideoDataset
import pytorchvideo.data

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)

import os
import pathlib
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
print(os.environ['PYTORCH_ENABLE_MPS_FALLBACK'])

# initialize model
class_labels = {'0','1','2','3','4','5','6','7','8','9'}
label2id = {label: i for i, label in enumerate(class_labels)}
id2label = {i: label for label, i in label2id.items()}

print(f"Unique classes: {list(label2id.keys())}.")


model_ckpt = "MCG-NJU/videomae-base"
batch_size = 8

dataset_root_path = pathlib.Path('dataset_root_path')
video_count_train = len(list(dataset_root_path.glob("train/*/*.avi")))
video_count_val = len(list(dataset_root_path.glob("val/*/*.avi")))
video_count_test = len(list(dataset_root_path.glob("test/*/*.avi")))
video_total = video_count_train + video_count_val + video_count_test
print(f"Total videos: {video_total}")

image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
model = VideoMAEForVideoClassification.from_pretrained(
    model_ckpt,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)

# image preprozessing
mean = image_processor.image_mean
std = image_processor.image_std
if "shortest_edge" in image_processor.size:
    height = width = image_processor.size["shortest_edge"]
else:
    height = image_processor.size["height"]
    width = image_processor.size["width"]
resize_to = (height, width)

num_frames_to_sample = model.config.num_frames
sample_rate = 8
fps = 60
clip_duration = num_frames_to_sample * sample_rate / fps

#training
train_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(resize_to),
                    RandomHorizontalFlip(p=0.5),
                ]
            ),
        ),
    ]
)

train_dataset = pytorchvideo.data.Ucf101(
    data_path='dataset_3/dataset_transformer',
    clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
    decode_audio=False,
    transform=train_transform,
)

#workflow for test and validation
val_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    Resize(resize_to),
                ]
            ),
        ),
    ]
)

val_dataset = pytorchvideo.data.Ucf101(
    data_path='/Users/juliakisela/HKA/8.Semester/Thesis/talking_in_the_disco/dataset_transformers/val',
    clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
    decode_audio=False,
    transform=val_transform,
)

#test_dataset = pytorchvideo.data.Ucf101(
#    data_path='/Users/juliakisela/HKA/8.Semester/Thesis/talking_in_the_disco/dataset_transformers/test',
#    clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
#    decode_audio=False,
#    transform=val_transform,
#)

##visualize
#import imageio
#import numpy as np
#from IPython.display import Image
#
#def unnormalize_img(img):
#    """Un-normalizes the image pixels."""
#    img = (img * std) + mean
#    img = (img * 255).astype("uint8")
#    return img.clip(0, 255)
#
#def create_gif(video_tensor, filename="sample.gif"):
#    """Prepares a GIF from a video tensor.
#    
#    The video tensor is expected to have the following shape:
#    (num_frames, num_channels, height, width).
#    """
#    frames = []
#    for video_frame in video_tensor:
#        frame_unnormalized = unnormalize_img(video_frame.permute(1, 2, 0).numpy())
#        frames.append(frame_unnormalized)
#    kargs = {"duration": 0.25}
#    imageio.mimsave(filename, frames, "GIF", **kargs)
#    return filename
#
#def display_gif(video_tensor, gif_name="sample.gif"):
#    """Prepares and displays a GIF from a video tensor."""
#    video_tensor = video_tensor.permute(1, 0, 2, 3)
#    gif_filename = create_gif(video_tensor, gif_name)
#    return Image(filename=gif_filename)
#
#sample_video = next(iter(train_dataset))
#video_tensor = sample_video["video"]
#display_gif(video_tensor)

#train
from transformers import TrainingArguments, Trainer

model_name = model_ckpt.split("/")[-1]
new_model_name = f"{model_name}-finetuned-ucf101-subset"
num_epochs = 4
batch_size = 16

args = TrainingArguments(
    new_model_name,
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    max_steps=(train_dataset.num_videos // batch_size) * num_epochs,
)

#predict
import evaluate
import numpy as np
import torch

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

# pixel_values and labels
def collate_fn(examples):
    # permute to (num_frames, num_channels, height, width)
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels} 

#it's time for training
trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

train_results = trainer.train()