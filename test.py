import argparse
import json
from pathlib import Path

import torch.nn.functional as F

import torch
from tqdm import tqdm

from asr.datasets.utils import get_dataloaders
from asr.text_encoder.ctc_text_encoder import CTCTextEncoder
from asr.text_encoder.text_encoder import get_simple_alphabet
import asr.models as module_model
from asr.trainer import Trainer
from asr.utils import ROOT_PATH
from asr.utils.parse_config import ConfigParser
from asr.metrics.utils import calc_cer, calc_wer

DEFAULT_TEST_CONFIG_PATH = ROOT_PATH / "default_test_config.json"
DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(config, out_file):
    logger = config.get_logger("test")

    # Text encoder
    text_encoder = CTCTextEncoder(get_simple_alphabet())

    # Setup dataloader instances
    dataloaders = get_dataloaders(config, text_encoder)

    # Build model architecture
    model = config.init_obj(config["arch"], module_model, n_class=len(text_encoder))
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint["state_dict"]

    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(state_dict)

    # Prepare model for testing
    model = model.to(device)
    model.eval()

    results = []

    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloaders["test"])):
            batch = Trainer.move_batch_to_device(batch, device)

            outputs = model(**batch)

            if type(outputs) is dict:
                batch.update(outputs)
            else:
                batch["logits"] = outputs

        
            batch["log_probs"] = F.log_softmax(batch["logits"], dim=-1)
            batch["log_probs_length"] = model.transform_input_lengths(batch["spectrogram_length"])
            batch["probs"] = batch["log_probs"].exp().cpu()
            batch["argmax"] = batch["probs"].argmax(-1)

            for i in range(len(batch["text"])):

                ground_trurh = batch["text"][i]
                pred_text_argmax = text_encoder.ctc_decode(batch["argmax"][i])
                pred_text_beam_search = text_encoder.ctc_beam_search(batch["probs"][i], beam_size=100)[:10]

                results.append({"ground_trurh": ground_trurh,
                                "pred_text_argmax": pred_text_argmax,
                                "argmax_wer": calc_wer(ground_trurh, pred_text_argmax) * 100,
                                "argmax_cer": calc_cer(ground_trurh, pred_text_argmax) * 100,
                                "pred_text_beam_search": pred_text_beam_search, 
                                "beam_search_wer": calc_wer(ground_trurh, pred_text_beam_search[0][0]) * 100,
                                "beam_search_cer": calc_cer(ground_trurh, pred_text_beam_search[0][0]) * 100})

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

    config = ConfigParser.from_args(args)

    args = args.parse_args()

    config.config["data"] = {
        "test": {
            "batch_size": args.batch_size,
            "num_workers": args.jobs,
            "datasets": [
                {
                    "type": "LibrispeechDataset",
                    "args": {
                        "part": "test-clean"
                    }
                }
            ]
        }
    }

    main(config, args.output)