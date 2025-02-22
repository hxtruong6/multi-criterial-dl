import pytorch_lightning
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import random
import numpy as np
import pytorch_lightning as pl
import logging
from argparse import ArgumentParser
import resource

import torch
from model.SE_XLNet import SEXLNet
from model.data import ClassificationData

logging.basicConfig(level=logging.INFO)


def get_train_steps(dm, args):
    total_devices = args.num_gpus * args.num_nodes
    if total_devices == 0:
        logging.warning("No GPUs found, using 1 device")
        total_devices = 1

    logging.info(f"Total devices: {total_devices}")
    train_batches = len(dm.train_dataloader()) // total_devices
    if args.accumulate_grad_batches is None:
        return args.max_epochs * train_batches
    else:
        return args.max_epochs * train_batches // args.accumulate_grad_batches


def get_parse_args():

    # argparser
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument(
        "--dataset_basedir", help="Base directory where the dataset is located.", type=str
    )
    parser.add_argument("--concept_store", help="Concept store file", type=str)
    parser.add_argument("--model_name", default="xlnet-base-cased", help="Model to use.")
    parser.add_argument("--gamma", default=0.01, type=float, help="Gamma parameter")
    parser.add_argument("--lamda", default=0.01, type=float, help="Lamda Parameter")
    parser.add_argument("--topk", default=5, type=int, help="Topk GIL concepts")
    # run.py: error: unrecognized arguments: --lr 2e-5 --max_epochs 5 --gpus 1 --accelerator ddp
    parser.add_argument("--lr", default=2e-5, type=float, help="Initial learning rate.")
    parser.add_argument("--max_epochs", default=5, type=int, help="Max epochs")
    parser.add_argument("--accelerator", default="ddp", type=str, help="Accelerator")

    # For SE_XLNet
    parser.add_argument("--min_lr", default=0, type=float, help="Minimum learning rate.")
    parser.add_argument("--h_dim", type=int, help="Size of the hidden dimension.", default=768)
    parser.add_argument("--n_heads", type=int, help="Number of attention heads.", default=1)
    parser.add_argument("--kqv_dim", type=int, help="Dimensionality of the each attention head.", default=256)
    parser.add_argument("--num_classes", type=float, help="Number of classes.", default=2)
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay rate.")
    parser.add_argument("--warmup_prop", default=0.01, type=float, help="Warmup proportion.")

    return parser


def main():
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    # init: important to make sure every node initializes the same weights
    SEED = 18
    np.random.seed(SEED)
    random.seed(SEED)
    pytorch_lightning.seed_everything(SEED)

    args = get_parse_args().parse_args()
    args.num_gpus = torch.cuda.device_count()
    args.num_nodes = 1
    args.accumulate_grad_batches = 1

    logging.info(f"Args: {args}")

    # Step 1: Init Data
    logging.info(
        "--------------------------------Loading the data module--------------------------------"
    )
    dm = ClassificationData(
        basedir=args.dataset_basedir,
        tokenizer_name=args.model_name,
        batch_size=args.batch_size,
        num_workers=8,
    )

    # Step 2: Init Model
    logging.info(
        "--------------------------------Initializing the model--------------------------------"
    )
    model = SEXLNet(hparams=args)
    model.hparams.warmup_steps = int(
        get_train_steps(dm, args) * model.hparams.warmup_prop
    )

    # Init Learning Rate Monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Step 3: Start training
    logging.info(
        "--------------------------------Starting the training--------------------------------"
    )
    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-{step}-{val_acc_epoch:.4f}",
        save_top_k=3,
        verbose=True,
        monitor="val_acc_epoch",
        mode="max",
    )

    trainer = pytorch_lightning.Trainer(
        callbacks=[checkpoint_callback, lr_monitor],
        val_check_interval=0.5,
        gradient_clip_val=args.clip_grad,
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        strategy="auto",
    )

    logging.info(
        "--------------------------------Fitting the model--------------------------------"
    )
    trainer.fit(model, dm)
    logging.info(
        "--------------------------------Training completed--------------------------------"
    )


if __name__ == "__main__":
    main()
