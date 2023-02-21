#questo pacchetto ha a che fare con gli argomenti che si passano 
#quando si chiama il file.py da terminale come function
from argparse import Namespace, ArgumentParser 

from datetime import datetime
import json
import logging
from pathlib import Path

import yaml
import torch
import torchvision
import pytorch_lightning as pl
import pandas as pd

from classification import utils_global
from classification.s2_utils import Partitioning, Hierarchy
from classification.dataset import ImageDataset

#in questa classe vengono utilizzate altre 8 classi definte dagli autori del paper negli altri scirpt
class MultiPartitioningClassifier(pl.LightningModule):
    #MultiPartitioningClassifier è una classe figlia della classe pl.LightningModule
    def __init__(self, hparams: Namespace):
        #con questo comando MultiPartitioningClassifier eredita tutti gli attributi e tutti i metodi di pl.LightningModule
        super().__init__()
        #il nuovo attributo .hparams grazie a Namespace poi risulta avere
        #una miriade di sotto attributi: hparams.optim, hparams.batch_size, ecc (11 attributi)
        self.hparams = hparams
        #questi 4 attributi sono tutti output dei 2 metodi successivi
        self.partitionings, self.hierarchy = self.__init_partitionings()
        model, classifier = self.__build_model()
        self.model, self.intermediate, self.classifier = self.__build_intermediate_layer(model, classifier,
                                                                                         self.nfeatures,
                                                                                         option=self.hparams.architecture_number)

#in totale ci sono 13 (nuovi) metodi all'interno di questa classe. 
    def __init_partitionings(self):

        partitionings = []
        for shortname, path in zip(
            self.hparams.partitionings["shortnames"],
            self.hparams.partitionings["files"],
        ):
            partitionings.append(Partitioning(Path(path), shortname, skiprows=2))

        if len(self.hparams.partitionings["files"]) == 1:
            return partitionings, None

        return partitionings, Hierarchy(partitionings)

    def __build_model(self):
        logging.info("Build model")
        model, nfeatures = utils_global.build_base_model(self.hparams.arch)

        self.nfeatures = nfeatures

        classifier = torch.nn.ModuleList(
            [
                torch.nn.Linear(nfeatures, len(self.partitionings[i]))
                for i in range(len(self.partitionings))
            ]
        )
        #se stiamo usando un modello pretrainato allora avremo già dei pesi
        if self.hparams.weights:
            logging.info("Load weights from pre-trained model")
            model, classifier = utils_global.load_weights_if_available(
                model, classifier, self.hparams.weights, self.hparams.load_also_weights_classifier
            )

        return model, classifier

    def __build_intermediate_layer(self, model, classifier, nfeatures, option):
        if option == 0: #keep original architecture
            inter = False
        elif option == 1: #add fully connected layer in-between model and classifier
            inter = torch.nn.Sequential(torch.nn.Linear(nfeatures, nfeatures), torch.nn.ReLU())
        elif option == 2: #add fully connected layer initialized close to identity
            inter = torch.nn.Sequential(torch.nn.Linear(nfeatures, nfeatures), torch.nn.Tanh())
            inter[0].weight.data.copy_(torch.normal(torch.eye(nfeatures), 0.05))
        elif option == 3:  #add fully connected layer initialized close to last layer weights
            inter = torch.nn.Sequential(torch.nn.Linear(nfeatures, nfeatures), torch.nn.ReLU())
            last_layer = list(model.children())[-1]
            inter[0].weight.data = last_layer.weight.data.clone()
            inter[0].bias.data = last_layer.bias.data.clone()
        return model, inter, classifier

    def forward(self, x):
        if self.hparams.train_all:
            fv = self.model(x)
        else:
            self.model.eval()
            with torch.no_grad():
                fv = self.model(x)
        if self.intermediate:
            inter = self.intermediate(fv)
        else:
            inter = fv
        yhats = [self.classifier[i](inter) for i in range(len(self.partitionings))]
        return yhats

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        images, target = batch

        if not isinstance(target, list) and len(target.shape) == 1:
            target = [target]

        # forward pass
        output = self(images) #self(...) è la stessa cosa di chiamare forward(...)

        # individual losses per partitioning
        losses = [
            torch.nn.functional.cross_entropy(output[i], target[i])
            for i in range(len(output))
        ]

        loss = sum(losses)

        # stats
        losses_stats = {
            f"loss_train/{p}": l
            for (p, l) in zip([p.shortname for p in self.partitionings], losses)
        }
        for metric_name, metric_value in losses_stats.items():
            self.log(metric_name, metric_value, prog_bar=True, logger=True)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, **losses_stats}

    def validation_step(self, batch, batch_idx):
        images, target, true_lats, true_lngs = batch

        if not isinstance(target, list) and len(target.shape) == 1:
            target = [target]

        # forward
        output = self(images)

        # loss calculation
        losses = [
            torch.nn.functional.cross_entropy(output[i], target[i])
            for i in range(len(output))
        ]

        loss = sum(losses)

        # log top-k accuracy for each partitioning
        individual_accuracy_dict = utils_global.accuracy(
            output, target, [p.shortname for p in self.partitionings]
        )
        # log loss for each partitioning
        individual_loss_dict = {
            f"loss_val/{p}": l
            for (p, l) in zip([p.shortname for p in self.partitionings], losses)
        }

        # log GCD error@km threshold
        distances_dict = {}

        if self.hierarchy is not None:
            hierarchy_logits = [
                yhat[:, self.hierarchy.M[:, i]] for i, yhat in enumerate(output)
            ]
            hierarchy_logits = torch.stack(hierarchy_logits, dim=-1,)
            hierarchy_preds = torch.prod(hierarchy_logits, dim=-1)

        pnames = [p.shortname for p in self.partitionings]
        if self.hierarchy is not None:
            pnames.append("hierarchy")
        for i, pname in enumerate(pnames):
            # get predicted coordinates
            if i == len(self.partitionings):
                i = i - 1
                pred_class_indexes = torch.argmax(hierarchy_preds, dim=1)
            else:
                pred_class_indexes = torch.argmax(output[i], dim=1)
            pred_latlngs = [
                self.partitionings[i].get_lat_lng(idx)
                for idx in pred_class_indexes.tolist()
            ]
            pred_lats, pred_lngs = map(list, zip(*pred_latlngs))
            pred_lats = torch.tensor(pred_lats, dtype=torch.float)
            pred_lngs = torch.tensor(pred_lngs, dtype=torch.float)
            # calculate error
            distances = utils_global.vectorized_gc_distance(
                pred_lats,
                pred_lngs,
                true_lats.type_as(pred_lats),
                true_lngs.type_as(pred_lats),
            )
            distances_dict[f"gcd_{pname}_val"] = distances

        output = {
            "loss_val/total": loss,
            **individual_accuracy_dict,
            **individual_loss_dict,
            **distances_dict,
        }
        return output

    def validation_epoch_end(self, outputs):
        pnames = [p.shortname for p in self.partitionings]

        # top-k accuracy and loss per partitioning
        loss_acc_dict = utils_global.summarize_loss_acc_stats(pnames, outputs)

        # GCD stats per partitioning
        gcd_dict = utils_global.summarize_gcd_stats(pnames, outputs, self.hierarchy)

        metrics = {
            "val_loss": loss_acc_dict["loss_val/total"],
            **loss_acc_dict,
            **gcd_dict,
        }
        for metric_name, metric_value in metrics.items():
            self.log(metric_name, metric_value, logger=True)

    def _multi_crop_inference(self, batch):
        images, meta_batch = batch
        cur_batch_size = images.shape[0]
        ncrops = images.shape[1]

        # reshape crop dimension to batch
        images = torch.reshape(images, (cur_batch_size * ncrops, *images.shape[2:]))

        # forward pass
        yhats = self(images)
        yhats = [torch.nn.functional.softmax(yhat, dim=1) for yhat in yhats]

        # respape back to access individual crops
        yhats = [
            torch.reshape(yhat, (cur_batch_size, ncrops, *list(yhat.shape[1:])))
            for yhat in yhats
        ]

        # calculate max over crops
        yhats = [torch.max(yhat, dim=1)[0] for yhat in yhats]

        hierarchy_preds = None
        if self.hierarchy is not None:
            hierarchy_logits = torch.stack(
                [yhat[:, self.hierarchy.M[:, i]] for i, yhat in enumerate(yhats)],
                dim=-1,
            )
            hierarchy_preds = torch.prod(hierarchy_logits, dim=-1)

        return yhats, meta_batch, hierarchy_preds

    def inference(self, batch):

        yhats, meta_batch, hierarchy_preds = self._multi_crop_inference(batch)

        if self.hierarchy is not None:
            nparts = len(self.partitionings) + 1
        else:
            nparts = len(self.partitionings)

        pred_class_dict = {}
        pred_lat_dict = {}
        pred_lng_dict = {}
        for i in range(nparts):
            # get pred class indices
            if self.hierarchy is not None and i == len(self.partitionings):
                pname = "hierarchy"
                pred_classes = torch.argmax(hierarchy_preds, dim=1)
                i = i - 1
            else:
                pname = self.partitionings[i].shortname
                pred_classes = torch.argmax(yhats[i], dim=1)

            # calculate GCD
            pred_lats, pred_lngs = map(
                list,
                zip(
                    *[
                        self.partitionings[i].get_lat_lng(c)
                        for c in pred_classes.tolist()
                    ]
                ),
            )
            pred_lats = torch.tensor(pred_lats, dtype=torch.float)
            pred_lngs = torch.tensor(pred_lngs, dtype=torch.float)
            pred_lat_dict[pname] = pred_lats
            pred_lng_dict[pname] = pred_lngs
            pred_class_dict[pname] = pred_classes

        return meta_batch["img_path"], pred_class_dict, pred_lat_dict, pred_lng_dict

    def test_step(self, batch, batch_idx, dataloader_idx=None):

        yhats, meta_batch, hierarchy_preds = self._multi_crop_inference(batch)

        distances_dict = {}
        if self.hierarchy is not None:
            nparts = len(self.partitionings) + 1
        else:
            nparts = len(self.partitionings)

        for i in range(nparts):
            # get pred class indices
            if self.hierarchy is not None and i == len(self.partitionings):
                pname = "hierarchy"
                pred_classes = torch.argmax(hierarchy_preds, dim=1)
                i = i - 1
            else:
                pname = self.partitionings[i].shortname
                pred_classes = torch.argmax(yhats[i], dim=1)

            # calculate GCD
            pred_lats, pred_lngs = map(
                list,
                zip(
                    *[
                        self.partitionings[i].get_lat_lng(c)
                        for c in pred_classes.tolist()
                    ]
                ),
            )
            pred_lats = torch.tensor(pred_lats, dtype=torch.float)
            pred_lngs = torch.tensor(pred_lngs, dtype=torch.float)

            distances = utils_global.vectorized_gc_distance(
                pred_lats,
                pred_lngs,
                meta_batch["latitude"].type_as(pred_lats),
                meta_batch["longitude"].type_as(pred_lngs),
            )
            distances_dict[pname] = distances

        return distances_dict

    def test_epoch_end(self, outputs):
        result = utils_global.summarize_test_gcd(
            [p.shortname for p in self.partitionings], outputs, self.hierarchy
        )
        return {**result}

    def configure_optimizers(self):

        optim_feature_extrator = torch.optim.SGD(
            self.parameters(), **self.hparams.optim["params"]
        )

        return {
            "optimizer": optim_feature_extrator,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                    optim_feature_extrator, **self.hparams.scheduler["params"]
                ),
                "interval": "epoch",
                "name": "lr",
            },
        }

    def train_dataloader(self):

        with open(self.hparams.train_label_mapping, "r") as f:
            target_mapping = json.load(f)

        tfm = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomResizedCrop(224, scale=(0.66, 1.0)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

        dataset = ImageDataset(
            path=self.hparams.msgpack_train_dir,
            target_mapping=target_mapping,
            shuffle=True,
            transformation=tfm,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers_per_loader,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):

        with open(self.hparams.val_label_mapping, "r") as f:
            target_mapping = json.load(f)

        tfm = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )
        dataset = ImageDataset(
            path=self.hparams.msgpack_val_dir,
            target_mapping=target_mapping,
            shuffle=False,
            transformation=tfm,
            meta_path=self.hparams.val_meta_path,
            cache_size=128,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers_per_loader,
            pin_memory=True,
        )

        return dataloader


def parse_args():
    args = ArgumentParser()
    args.add_argument("-c", "--config", type=Path, default=Path("config/prova.yml"))
    args.add_argument("--progbar", action="store_true")
    return args.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_params = config["model_params"]
    trainer_params = config["trainer_params"]

    utils_global.check_is_valid_torchvision_architecture(model_params["arch"])

    out_dir = Path(config["out_dir"]) / datetime.now().strftime("%y%m%d-%H%M")
    out_dir.mkdir(exist_ok=True, parents=True)
    logging.info(f"Output directory: {out_dir}")

    # init classifier
    model = MultiPartitioningClassifier(hparams=Namespace(**model_params))

    logger = pl.loggers.TensorBoardLogger(save_dir=str(out_dir), name="tb_logs")
    checkpoint_dir = out_dir / "ckpts" / "{epoch:03d}-{val_loss:.4f}"
    checkpointer = pl.callbacks.model_checkpoint.ModelCheckpoint(checkpoint_dir)

    progress_bar_refresh_rate = 0
    if args.progbar:
        progress_bar_refresh_rate = 1

    trainer = pl.Trainer(
        **trainer_params,
        logger=logger,
        val_check_interval=model_params["val_check_interval"],
        checkpoint_callback=checkpointer,
        progress_bar_refresh_rate=progress_bar_refresh_rate,
    )

    trainer.fit(model)

#cosa fa questo??
if __name__ == "__main__":
    main()
