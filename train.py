import argparse
import collections
import warnings

import numpy as np
import torch

import random
import os

import asr.loss as module_loss
import asr.metrics as module_metrics
import asr.models as module_arch
from asr.datasets.utils import get_dataloaders
from asr.text_encoder.ctc_text_encoder import CTCTextEncoder
from asr.text_encoder.text_encoder import get_simple_alphabet
from asr.trainer import Trainer
from asr.utils import prepare_device
from asr.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# Fix random seeds for reproducibility
seed = 42 
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"


def main(config):
    logger = config.get_logger("train")

    # Text_encoder
    text_encoder = CTCTextEncoder(get_simple_alphabet())

    # Setup data_loader instances
    dataloaders = get_dataloaders(config, text_encoder)
    dataloaders["val"] = None if config["overfit"] else dataloaders["val"]

    # Build model architecture, then print to console
    model = config.init_obj(config["arch"], module_arch, n_class=len(text_encoder.char_to_index))
    if not config["is_first_to_train_model"]:
        model.load_state_dict(torch.load(config["previous_model_path"])["state_dict"])
        print("Downloaded the pretrained model.")
    logger.info(model)

    # Prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # Get function handles of loss and metrics
    loss_module = config.init_obj(config["loss"], module_loss).to(device)
    metrics = [config.init_obj(metric_dict, module_metrics, text_encoder=text_encoder)
               for metric_dict in config["metrics"]]

    # Build optimizer, learning rate scheduler. Delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = config.init_obj(config["optimizer"], torch.optim, trainable_params)
    lr_scheduler = config.init_obj(config["lr_scheduler"], torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model=model,
                      criterion=loss_module,
                      metrics=metrics,
                      optimizer=optimizer,
                      config=config,
                      device=device,
                      text_encoder=text_encoder,
                      dataloader=dataloaders["train"],
                      valid_dataloader=dataloaders["val"],
                      lr_scheduler=lr_scheduler,
                      len_epoch=config["trainer"].get("len_epoch", None),
                      skip_oom=True)

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")

    args.add_argument("-c",
                      "--config",
                      default=None,
                      type=str,
                      help="config file path (default: None)")

    args.add_argument("-r",
                      "--resume",
                      default=None,
                      type=str,
                      help="path to latest checkpoint (default: None)")

    args.add_argument("-d",
                      "--device",
                      default=None,
                      type=str,
                      help="indices of GPUs to enable (default: all)")

    # Custom command line options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
               CustomArgs(["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size")]

    config = ConfigParser.from_args(args, options)
    main(config)
