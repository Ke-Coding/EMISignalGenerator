from torch.utils.data import DataLoader
from .emi_ds import LabeledEMIDataSet, InLabeledEMIDataSet


def get_dataloaders(data_dir, num_classes, vae_batch_size, cls_batch_size, lg):
    lg.info('==> Preparing data..')
    
    if data_dir is None or len(data_dir) == 0:
        raise AttributeError(f'data file {data_dir} not found!')
    
    lg.info(f'==> Reading dataset from {data_dir} ...')
    vae_train_set = InLabeledEMIDataSet(
        data_dir=data_dir, train=True, num_classes=num_classes, normalize=True)
    vae_test_set = InLabeledEMIDataSet(
        data_dir=data_dir, train=False, num_classes=num_classes, normalize=True)
    cls_train_set = LabeledEMIDataSet(
        data_dir=data_dir, train=True, num_classes=num_classes, normalize=True)
    cls_test_set = LabeledEMIDataSet(
        data_dir=data_dir, train=False, num_classes=num_classes, normalize=True)
    
    lg.info(f'==> Getting dataloader from {data_dir} ...')
    vae_train_loader = DataLoader(
        dataset=vae_train_set, batch_size=vae_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    vae_test_loader = DataLoader(
        dataset=vae_test_set, batch_size=vae_batch_size, shuffle=False, num_workers=2, pin_memory=True)
    cls_train_loader = DataLoader(
        dataset=cls_train_set, batch_size=cls_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    cls_test_loader = DataLoader(
        dataset=cls_test_set, batch_size=cls_batch_size, shuffle=False, num_workers=2, pin_memory=True)

    classes = ('pink', 'brown', 'laplace', 'uniform', 'exponential')
    assert num_classes == len(classes)
    lg.info(f'==> Preparing data complete, classes:{classes} .\n')
    
    return vae_train_loader, vae_test_loader, cls_train_loader, cls_test_loader, classes
