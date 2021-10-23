import json
import logging
import os
from datetime import datetime
from functools import reduce, partial
from operator import getitem
from pathlib import Path

from asr.logger import setup_logging
from asr.utils import read_json, write_json, ROOT_PATH


class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None):
        """
        Class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.

        :param config: Dict containing configurations, hyperparameters for training. Contents of `config.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default.
        """
        # Load config file and apply modification
        self._config = update_config(config, modification)
        self.resume = resume

        # Set save_dir where trained model and log will be saved.
        save_dir = Path(self._config["trainer"]["save_dir"])

        experiment_name = self._config["name"]
        if run_id is None:  # use timestamp as default run-id
            run_id = datetime.now().strftime(r"%m%d_%H%M%S")
            
        self._save_dir = save_dir / "models" / experiment_name / run_id
        self._log_dir = save_dir / "log" / experiment_name / run_id

        # Make directory for saving checkpoints and log.
        exist_ok = run_id == ""
        self._save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self._log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # Save updated config file to the checkpoint dir
        write_json(self._config, self._save_dir / "config.json")

        # Configure logging module
        setup_logging(self._log_dir)
        self.log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

    @classmethod
    def from_args(cls, args, options=""):
        """
        Initialize this class from some command line arguments. Used in train, test.
        """
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)

        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device

        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / "config.json"
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)

        config = read_json(cfg_fname)
        if args.config and resume:
            # update new config for fine-tuning
            config.update(read_json(args.config))

        # parse custom cli options into dictionary
        modification = {opt.target: getattr(args, _get_opt_name(opt.flags)) for opt in options}
        return cls(config, resume, modification)

    def init_obj(self, obj_dict, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj(config['param'], module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = obj_dict["type"]
        module_args = dict(obj_dict["args"])
        assert all([k not in module_args for k in kwargs]), "Overwriting kwargs given in config file is not allowed"
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]["type"]
        module_args = dict(self[name]["args"])
        assert all([k not in module_args for k in kwargs]), "Overwriting kwargs given in config file is not allowed"
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """
        Access items like ordinary dict.
        """
        return self._config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = f"verbosity option {verbosity} is invalid. Valid options are {self.log_levels.keys()}."
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

    @classmethod
    def get_default_configs(cls):
        config_path = ROOT_PATH / 'defualt_config.json'
        with config_path.open() as f:
            return cls(json.load(f))


# helper functions to update config dict with custom cli options
def update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            set_by_path(config, k, v)
            
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith("--"):
            return flg.replace("--", "")
    return flags[0].replace("--", "")


def set_by_path(tree, keys, value):
    """
    Set a value in a nested object in tree by sequence of keys.
    """
    keys = keys.split(";")
    get_by_path(tree, keys[:-1])[keys[-1]] = value


def get_by_path(tree, keys):
    """
    Access a nested object in tree by sequence of keys.
    """
    return reduce(getitem, keys, tree)
