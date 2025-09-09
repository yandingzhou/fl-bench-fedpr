import json
import os
from enum import Enum
from pathlib import Path

from torch import optim

FLBENCH_ROOT = Path(__file__).parent.parent.parent.absolute()
OUT_DIR = FLBENCH_ROOT / "out"
TEMP_DIR = FLBENCH_ROOT / "temp"


class MODE(Enum):
    SERIAL = 0
    PARALLEL = 1


DEFAULTS = {
    "method": "fedavg",
    "dataset": {"name": "mnist"},
    "model": {
        "name": "lenet5",
        "use_torchvision_pretrained_weights": True,
        "external_model_weights_path": None,
    },
    "lr_scheduler": {
        "name": None,
        "step_size": 10,
        "gamma": 0.1,
        "T_max": 10,
        "eta_min": 0,
        "factor": 0.3334,
        "total_iters": 5,
        "mode": "min",
        "patience": 10,
        "threshold": 1.0e-4,
        "threshold_mode": "rel",
        "cooldown": 0,
        "min_lr": 0,
        "eps": 1.0e-8,
        "last_epoch": -1,
    },
    "optimizer": {
        "name": "sgd",
        "lr": 0.01,
        "dampening": 0,
        "weight_decay": 0,
        "momentum": 0,
        "alpha": 0.99,
        "nesterov": False,
        "betas": [0.9, 0.999],
        "amsgrad": False,
    },
    "mode": "serial",
    "parallel": {
        "ray_cluster_addr": None,
        "num_cpus": None,
        "num_gpus": None,
        "num_workers": 2,
    },
    "common": {
        "seed": 42,
        "join_ratio": 0.1,
        "global_epoch": 100,
        "local_epoch": 5,
        "batch_size": 32,
        "reset_optimizer_on_global_epoch": True,
        "straggler_ratio": 0,
        "straggler_min_local_epoch": 0,
        "buffers": "global",
        "client_side_evaluation": True,
        "test": {
            "client": {
                "interval": 100,
                "finetune_epoch": 0,
                "train": False,
                "val": False,
                "test": True,
            },
            "server": {
                "interval": -1,
                "train": False,
                "val": False,
                "test": False,
                "model_in_train_mode": False,
            },
        },
        "verbose_gap": 10,
        "monitor": None,
        "use_cuda": True,
        "save_log": True,
        "save_model": False,
        "save_learning_curve_plot": True,
        "save_metrics": True,
        "delete_useless_run": True,
    },
}


INPUT_CHANNELS = {
    "mnist": 1,
    "medmnistS": 1,
    "medmnistC": 1,
    "medmnistA": 1,
    "covid19": 3,
    "fmnist": 1,
    "emnist": 1,
    "femnist": 1,
    "cifar10": 3,
    "cinic10": 3,
    "svhn": 3,
    "cifar100": 3,
    "celeba": 3,
    "usps": 1,
    "tiny_imagenet": 3,
    "domain": 3,
}


def _get_domainnet_args():
    if os.path.isfile(FLBENCH_ROOT / "data" / "domain" / "metadata.json"):
        with open(FLBENCH_ROOT / "data" / "domain" / "metadata.json", "r") as f:
            metadata = json.load(f)
        return metadata
    else:
        return {}


def _get_synthetic_args():
    if os.path.isfile(FLBENCH_ROOT / "data" / "synthetic" / "args.json"):
        with open(FLBENCH_ROOT / "data" / "synthetic" / "args.json", "r") as f:
            metadata = json.load(f)
        return metadata
    else:
        return {}


# (C, H, W)
DATA_SHAPE = {
    "mnist": (1, 28, 28),
    "medmnistS": (1, 28, 28),
    "medmnistC": (1, 28, 28),
    "medmnistA": (1, 28, 28),
    "fmnist": (1, 28, 28),
    "svhn": (3, 32, 32),
    "emnist": 62,
    "femnist": 62,
    "cifar10": (3, 32, 32),
    "cinic10": (3, 32, 32),
    "cifar100": (3, 32, 32),
    "covid19": (3, 244, 224),
    "usps": (1, 16, 16),
    "celeba": (3, 218, 178),
    "tiny_imagenet": (3, 64, 64),
    "synthetic": _get_synthetic_args().get("dimension", 0),
    "domain": (3, *(_get_domainnet_args().get("image_size", (0, 0)))),
}

NUM_CLASSES = {
    "mnist": 10,
    "medmnistS": 11,
    "medmnistC": 11,
    "medmnistA": 11,
    "fmnist": 10,
    "svhn": 10,
    "emnist": 62,
    "femnist": 62,
    "cifar10": 10,
    "cinic10": 10,
    "cifar100": 100,
    "covid19": 4,
    "usps": 10,
    "celeba": 2,
    "tiny_imagenet": 200,
    "synthetic": _get_synthetic_args().get("class_num", 0),
    "domain": _get_domainnet_args().get("class_num", 0),
}


DATA_MEAN = {
    "mnist": [0.1307],
    "cifar10": [0.4914, 0.4822, 0.4465],
    "cifar100": [0.5071, 0.4865, 0.4409],
    "emnist": [0.1736],
    "fmnist": [0.286],
    "femnist": [0.9637],
    "medmnist": [124.9587],
    "medmnistA": [118.7546],
    "medmnistC": [124.424],
    "covid19": [125.0866, 125.1043, 125.1088],
    "celeba": [128.7247, 108.0617, 97.2517],
    "synthetic": [0.0],
    "svhn": [0.4377, 0.4438, 0.4728],
    "tiny_imagenet": [122.5119, 114.2915, 101.388],
    "cinic10": [0.47889522, 0.47227842, 0.43047404],
    "domain": [0.485, 0.456, 0.406],
}


DATA_STD = {
    "mnist": [0.3015],
    "cifar10": [0.2023, 0.1994, 0.201],
    "cifar100": [0.2009, 0.1984, 0.2023],
    "emnist": [0.3248],
    "fmnist": [0.3205],
    "femnist": [0.155],
    "medmnist": [57.5856],
    "medmnistA": [62.3489],
    "medmnistC": [58.8092],
    "covid19": [56.6888, 56.6933, 56.6979],
    "celeba": [67.6496, 62.2519, 61.163],
    "synthetic": [1.0],
    "svhn": [0.1201, 0.1231, 0.1052],
    "tiny_imagenet": [58.7048, 57.7551, 57.6717],
    "cinic10": [0.24205776, 0.23828046, 0.25874835],
    "domain": [0.229, 0.224, 0.225],
}

OPTIMIZERS = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "rmsprop": optim.RMSprop,
    "adagrad": optim.Adagrad,
}

LR_SCHEDULERS = {
    "step": optim.lr_scheduler.StepLR,
    "cosine": optim.lr_scheduler.CosineAnnealingLR,
    "constant": optim.lr_scheduler.ConstantLR,
    "plateau": optim.lr_scheduler.ReduceLROnPlateau,
}

DATASET_CLASSES = {
    "cifar10": [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ],
    "cifar100": [
        "apple", "aquarium_fish", "baby", "bear", "beaver",
        "bed", "bee", "beetle", "bicycle", "bottle",
        "bowl", "boy", "bridge", "bus", "butterfly",
        "camel", "can", "castle", "caterpillar", "cattle",
        "chair", "chimpanzee", "clock", "cloud", "cockroach",
        "couch", "crab", "crocodile", "cup", "dinosaur",
        "dolphin", "elephant", "flatfish", "forest", "fox",
        "girl", "hamster", "house", "kangaroo", "keyboard",
        "lamp", "lawn_mower", "leopard", "lion", "lizard",
        "lobster", "man", "maple_tree", "motorcycle", "mountain",
        "mouse", "mushroom", "oak_tree", "orange", "orchid",
        "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
        "plain", "plate", "poppy", "porcupine", "possum",
        "rabbit", "raccoon", "ray", "road", "rocket",
        "rose", "sea", "seal", "shark", "shrew",
        "skunk", "skyscraper", "snail", "snake", "spider",
        "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
        "tank", "telephone", "television", "tiger", "tractor",
        "train", "trout", "tulip", "turtle", "wardrobe",
        "whale", "willow_tree", "wolf", "woman", "worm"
    ],
}

# Create dictionary configuration instead of using YACS ConfigNode
CONFIG = {
    'SEED': 42,
    'dist_url': 'env://',
    'world_size': 1,
    'lam': 1,
    'beta': 100,

    'DATASET': {
        'ROOT': [
            os.path.join(os.path.expanduser('~'), 'data/AAA-Download-Datasets/Fed/Data/JiangSu'),
            os.path.join(os.path.expanduser('~'), 'data/AAA-Download-Datasets/Fed/Data/lianying'),
            os.path.join(os.path.expanduser('~'), 'data/AAA-Download-Datasets/Fed/Data/FeTS2022'),
            os.path.join(os.path.expanduser('~'), 'data/AAA-Download-Datasets/Fed/Data/IXI-NIFTI'),
        ],
        'CLIENTS': ['JiangSu', 'lianying', 'FedTS01', 'FedTS02', 'FedTS03', 'FedTS04', 'FedTS05', 'FedTS06',
                    'FedTS07', 'FedTS08', 'FedTS09', 'BraTS10', 'ixiHH', 'ixiGuys', 'ixiIOP'],
        'PATTERN': ['T2'] * 15,
        'SAMPLE_RATE': [1] * 15,
        'CHALLENGE': 'singlecoil',
        'NUM_TRAIN': [432, 360, 120, 304, 304, 240, 160, 160, 280, 224, 264, 184, 360, 328, 296],
        'NUM_VAL': [184, 154, 80, 130, 130, 100, 64, 64, 120, 96, 112, 78, 152, 120, 128],
        'NUM_SERVER_VAL': 1712
    },

    'DATA_ROOT': os.path.join(os.path.expanduser('~'), 'data/AAA-Download-Datasets'),

    'TRANSFORMS': {
        'MASK_FILE': ["1D-Cartesian_3X_320.mat"] * 15,
        'MASK_DIR': os.path.join(os.path.expanduser('~'), 'data/AAA-Download-Datasets/Fed/Data/masks'),
        'MASK_SPEC': ["1D-Cartesian_3X"] * 15
    },

    'FL': {
        'CLIENTS_NUM': 100,
        'MODEL_NAME': 'swin_vpt_nullspace',
        'SHOW_SIZE': True
    },

    'MODEL': {
        'TRANSFER_TYPE': "prompt",  # one of linear, end2end, prompt, adapter, side, partial-1, tinytl-bias
        'WEIGHT_PATH': None,
        'SUBTYPE': "swin_320",
        'INPUTSIZE': 320,
        'FINALSIZE': 320,
        'HEAD_NUM_CONV': 4,
        'INPUT_DIM': 1,
        'OUTPUT_DIM': 1,
        'PROMPT': {
            'NUM_TOKENS': 20,
            'LOCATION': "prepend",
            'INITIATION': "random",
            'PROJECT': -1,
            'DEEP': True,  # whether do deep prompt or not, only for prepend location
            'DROPOUT': 0.0
        }
    },

    'SOLVER': {
        'DEVICE': 'cuda:1',
        'DEVICE_IDS': [0, 1, 2, 3],
        'LR': [0.1] * 15,
        'WEIGHT_DECAY': 0,
        'LR_DROP': 20,
        'BATCH_SIZE': 8,
        'NUM_WORKERS': 4,
        'PRINT_FREQ': 10,
        'LR_GAMMA': 0.1,
        'RATIO': 0.8
    },

    'RESUME': '',  # model resume path
    'PRETRAINED_FASTMRI_CKPT': 'path/to/pretrained/fastmri_ckpt.pth',
    'OUTPUTDIR': './saved',

    'TRAIN': {
        'EPOCHS': 50,
        'LOCAL_EPOCHS': 10
    },

    'meta_client_num': 15,
    'DISTRIBUTION_TYPE': 'in-distribution'  # alternative: 'out-of-distribution'
}