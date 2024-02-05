# IMPORT PYTORCH LIGHTNING LIBRARY APIs
import lightning.pytorch as pl
import torch
import torch.nn as nn
import timm


class cnnModel(pl.LightningModule):
    def __init__(
        self,
        model_name,
        num_classes
        ):
        super(cnnModel, self).__init__()
        self.save_hyperparameters() # tell the model to save the hyperparameters into the checkpoint file

        self.num_classes = num_classes

        self.model = timm.create_model(model_name, pretrained=True) # create model
        num_in_features = self.model.get_classifier().in_features # get number of penultimate layer's output

        ################# New Head ########################
        self.model.fc = nn.Sequential(
                    nn.Dropout(0.4),  # Move dropout before batch norm for potential regularization
                    nn.Linear(in_features=num_in_features, out_features=512, bias=False),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(in_features=512, out_features=self.num_classes, bias=False)
                    )# modify classify's head




    def forward(self, x):
        # batch -> ([list of images],[list of targets])
        # this forward method can be refer to model(x)
        # print("\nbatch \n", batch)

        logits = self.model(x)
        predictions = torch.nn.functional.softmax(logits, dim=1)  # Apply softmax
        predictions = predictions.argmax(dim=1)  # Get class indices
        # print(x)
        return predictions
