import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from asr.datasets.utils import get_dataloaders
from asr.text_encoder.ctc_text_encoder import CTCTextEncoder
import asr.models as module_model
import asr.loss as module_loss
import asr.metrics as module_metric
from asr.trainer import Trainer
from asr.utils import ROOT_PATH
from asr.utils.parse_config import ConfigParser

DEFAULT_TEST_CONFIG_PATH = ROOT_PATH / "default_test_model" / "config.json"
DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_file=None):
    logger = config.get_logger("test")

    # Text encoder
    text_encoder = CTCTextEncoder.get_simple_alphabet()

    # Setup dataloader instances
    dataloaders = get_dataloaders(config, text_encoder)

    # Build model architecture
    model = config.init_obj(config["arch"], module_model, n_class=len(text_encoder))
    logger.info(model)

    # Get function handles of loss and metrics
    loss_fn = getattr(module_loss, config["loss"])
    metric_fns = [getattr(module_metric, met) for met in config["metrics"]]

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint["state_dict"]

    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(state_dict)

    # Prepare model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    results = []

    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloaders["test"])):
            batch = Trainer.move_batch_to_device(batch, device)

            batch["logits"] = model(**batch)
            batch["log_probs"] = torch.nn.functional.log_softmax(batch["logits"], dim=-1)
            batch["log_probs_length"] = model.transform_input_lengths(batch["spectrogram_length"])
            batch["probs"] = batch["log_probs"].exp().cpu()
            batch["argmax"] = batch["probs"].argmax(-1)

            for i in range(len(batch["text"])):
                results.append({"ground_trurh": batch["text"][i],
                                "pred_text_argmax": text_encoder.ctc_decode(batch["argmax"][i]),
                                "pred_text_beam_search": text_encoder\
                                                         .ctc_beam_search(batch["probs"], 
                                                                          beam_size=100)[:10]})

    out_file = "default_test_model/results.json" if out_file is None else out_file

    with Path(out_file).open('w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument("-c",
                      "--config",
                      default=str(DEFAULT_TEST_CONFIG_PATH.absolute().resolve()),
                      type=str,
                      help="config file path (default: None)")

    args.add_argument("-r",
                      "--resume",
                      default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
                      type=str,
                      help="path to latest checkpoint (default: None)")

    args.add_argument("-d",
                      "--device",
                      default=None,
                      type=str,
                      help="indices of GPUs to enable (default: all)")

    args.add_argument("-o",
                      "--output",
                      default='output.json',
                      type=str,
                      help="File to write results (.json)")

    args.add_argument("-t",
                      "--test-data-folder",
                      default=None,
                      type=str,
                      help="Path to dataset")

    args.add_argument("-b",
                      "--batch-size",
                      default=20,
                      type=int,
                      help="Test dataset batch size")

    args.add_argument("-j",
                      "--jobs",
                      default=1,
                      type=int,
                      help="Number of workers for test dataloader")

    test_data_folder = Path(args.test_data_folder)

    config = ConfigParser.from_args(DEFAULT_TEST_CONFIG_PATH)
    main(config, args.output)
