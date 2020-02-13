import argparse
from pathlib import Path

import torch
from fastai.vision import *
from fastai.distributed import *
from fastai.callbacks import SaveModelCallback
from efficientnet_pytorch import EfficientNet

parser = argparse.ArgumentParser()
parser.add_argument("--image-root-dir", type=Path, required=True)
parser.add_argument("--train-indices-path", type=Path, required=True)
parser.add_argument("--val-indices-path", type=Path, required=True)
parser.add_argument("--pretrain-model", type=Path, help="must be absolute path without .pth suffix")
parser.add_argument("--n-max-epochs", type=int, required=True)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--max-lr", type=float)
# parser.add_argument("--resume-from-epoch", type=int, help="training will start from `resume_from_epoch` + 1")
parser.add_argument("--model-checkpoint-dir", type=Path, default="/tmp/fastai-models")
parser.add_argument("--local_rank", type=int)


def read_train_val_indices(train_indices_path, val_indices_path, image_root_dir=None):
    train_df = pd.read_csv(train_indices_path, header=None, sep=" ", names=["name", "label"])
    val_df = pd.read_csv(val_indices_path, header=None, sep=" ", names=["name", "label"])
    train_df["is_valid"] = False
    val_df["is_valid"] = True
    trainval_df = pd.concat([train_df, val_df])
    if image_root_dir is not None:
        trainval_df.loc[:, "name"] = trainval_df.name.apply(lambda x: Path(x).relative_to(image_root_dir))
    return trainval_df


def main(args):
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    tfms = get_transforms(
        do_flip=True,  # default True
        flip_vert=False,  # default False
        max_rotate=10.0,  # default 10.0
        max_zoom=1.1,  # default 1.1
        max_lighting=0.2,  # default 0.2
        max_warp=0.2,  # default 0.2
        p_affine=0.75,  # default 0.75
        p_lighting=0.75,  # default 0.75
    )

    dataset = read_train_val_indices(args.train_indices_path, args.val_indices_path, image_root_dir=args.image_root_dir)

    data_bunch = (
        ImageList.from_df(dataset, args.image_root_dir)
        .split_from_df()
        .label_from_df()
        .transform(tfms, size=256, resize_method=ResizeMethod.PAD, padding_mode="zeros")
        .databunch(bs=args.batch_size, num_workers=4)
        .normalize(imagenet_stats)
    )

    model = EfficientNet.from_pretrained("efficientnet-b5", num_classes=197)
    learn = Learner(data_bunch, model, metrics=accuracy, path=args.model_checkpoint_dir).to_distributed(args.local_rank)

    if args.pretrain_model is not None:
        learn = learn.load(args.pretrain_model)
        print(f"Loaded user specified pretrain-model {args.pretrain_model}")

    # learn.lr_find()
    # learn.recorder.plot(suggestion=True)
    # max_lr = learn.recorder.min_grad_lr
    # print(max_lr)

    # if args.resume_from_epoch is not None:
    #     learn.fit_one_cycle(
    #         args.n_max_epochs,
    #         max_lr=args.max_lr,
    #         start_epoch=args.resume_from_epoch + 1,
    #         callbacks=[SaveModelCallback(learn, every="epoch", monitor="accuracy")],
    #     )
    # else:

    learn.fit_one_cycle(
        args.n_max_epochs,
        max_lr=args.max_lr,
        callbacks=[SaveModelCallback(learn, every="epoch", monitor="accuracy")],
    )


if __name__ == "__main__":
    main(parser.parse_args())
