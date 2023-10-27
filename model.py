from lightning.pytorch.core import LightningModule
from transformers import SwinForImageClassification
from torch import nn
import torch
from einops import rearrange
import torch.onnx


class SwinModel(LightningModule):
    def __init__(self,aus,
                 variant='base',
                 lr=1e-4,
                 ckpt_path=None,
                 loss_reduction='mean'):                 
        super().__init__()
        variants = {'base': 'microsoft/swin-base-patch4-window7-224',
                    'tiny': 'microsoft/Swin-tiny-patch4-window7-224'}
        self.model = SwinForImageClassification.from_pretrained(variants[variant])
        self.AUs = aus
        num_aus = len(aus)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_aus)
        self.criterion = nn.BCEWithLogitsLoss(reduction = loss_reduction)
        self.learning_rate = lr
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self,path):
        sd = torch.load(path,map_location='cpu')
        if 'state_dict' in sd:
            sd = sd['state_dict']
        self.load_state_dict(sd,strict=False)

    def forward(self, x):
        outputs= self.model(x)
        return outputs.logits
    
    def compute_loss(self,outputs,labels):
        loss = self.criterion(outputs, labels)
        mask = labels != -1
        loss = loss * mask
        loss = loss.sum() / mask.sum()
        return loss
    
    def common_step(self, batch, batch_idx):
        x = batch['image']
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        outputs = self(x)
        labels = batch['aus']
        loss = self.compute_loss(outputs,labels)
        return loss,outputs

    def training_step(self, batch, batch_idx):
        loss,outputs = self.common_step(batch, batch_idx)
        self.log('train/loss', loss,sync_dist=True,prog_bar=True)
        return {'loss':loss,'logits':outputs}
    
    def validation_step(self, batch, batch_idx):
        loss,outputs = self.common_step(batch, batch_idx)
        self.log('val/loss', loss,sync_dist=True,prog_bar=True)
        return {'loss':loss,'logits':outputs}
    
    def test_step(self, batch, batch_idx):
        loss,outputs = self.common_step(batch, batch_idx)
        self.log('test/loss', loss,sync_dist=True,prog_bar=True)
        return {'loss':loss,'logits':outputs}
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = batch['image']
        if not x:
            return None
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        outputs = self(x)
        return outputs

    def configure_optimizers(self):
        lr = self.learning_rate
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        return optimizer
a=['AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU9', 'AU10','AU12', 'AU14', 'AU15', 'AU17','AU20', 'AU23', 'AU24', 'AU25','AU26','AU27','AU43']
dummy_input = torch.randn(1, 3, 224, 224)
path= './bestsofar.ckpt'

model = SwinModel(a,'base',1e-4,path,'mean')

onnx_path = "swin_transformer.onnx"
torch.onnx.export(model, dummy_input, onnx_path, verbose=True)