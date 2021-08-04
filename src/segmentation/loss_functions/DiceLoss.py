class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, bce_weight=0.0):
        
        bce = F.binary_cross_entropy_with_logits(inputs, targets)
        

        #pred = torch.sigmoid(inputs)
        pred = inputs
        dice = dice_loss(inputs, targets)

        loss = bce * bce_weight + dice * (1 - bce_weight)
        
        return bce
