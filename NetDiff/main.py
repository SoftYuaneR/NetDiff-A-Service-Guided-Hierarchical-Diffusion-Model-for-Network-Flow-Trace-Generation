
import argparse
import torch
import datetime
import json
import yaml
import os
import scipy.sparse as sp
from main_model_upload import CSDI_Physio
from new_csdi import get_dataloader
from utils import train,  evaluate

os.environ['CUDA_VISIBLE_DEVICES']='0'


parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cpu', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.1)
#n-fold（通常称为k-fold）是一种交叉验证（Cross-Validation）的方法，它是用来评估机器学习模型对于独立数据集的泛化能力的。
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
# parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--unconditional", default=True)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=1)
parser.add_argument("--c_in", type=int, default=64)
parser.add_argument("--graph_conv_type", type=str, default='graph_conv')


args = parser.parse_args()




path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/physio_fold" + str(args.nfold) + "_" + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader = get_dataloader(
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train"]["batch_size"],
)

model = CSDI_Physio(config, args.device).to(args.device)

if args.modelfolder == "":
    print('begin train')
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)
