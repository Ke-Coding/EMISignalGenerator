from torch.utils.data import DataLoader
from .emi_ds import EMIDataset


def build_dataloader(args, cfg, lg):
    lg.info('==> Preparing data..')
    
    if args.data_dir is None or len(args.data_dir) == 0:
        raise AttributeError(f'data file {args.data_dir} not found!')
    
    lg.info(f'==> Reading dataset from {args.data_dir} ...')
    
    train_set = EMIDataset(
        data_dir=args.data_dir, train=True, num_classes=cfg.model.kwargs.num_classes, normalize=True)
    test_set = EMIDataset(
        data_dir=args.data_dir, train=False, num_classes=cfg.model.kwargs.num_classes, normalize=True)
    
    lg.info(f'==> Getting dataloader from {args.data_dir} ...')
    train_loader = DataLoader(
        dataset=train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(
        dataset=test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    classes = ('pink', 'brown', 'laplace', 'uniform', 'exponential')
    assert cfg.model.kwargs.num_classes == len(classes)
    lg.info(f'==> Preparing data complete, classes:{classes} .\n')
    
    return train_loader, test_loader
