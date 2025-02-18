import logging

from methods.er_baseline import ER
# from methods.ewc import EWCpp
# from methods.mir import MIR
# from methods.clib import CLIB
# from methods.gss import GSS
# from methods.der import DER
from methods.sdp import *


from methods.baseline import BASELINE

logger = logging.getLogger()


def select_method(args, criterion, device, train_transform, test_transform, n_classes):
    kwargs = vars(args)
    if args.mode == "er":
        method = ER(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )
    # elif args.mode == "baseline":
    #     method = BASELINE(
    #         criterion=criterion,
    #         device=device,
    #         train_transform=train_transform,
    #         test_transform=test_transform,
    #         n_classes=n_classes,
    #         **kwargs,
    #     )
    # elif args.mode == "twf":
    #     method = TWF(
    #         criterion=criterion,
    #         device=device,
    #         train_transform=train_transform,
    #         test_transform=test_transform,
    #         n_classes=n_classes,
    #         **kwargs,
    #     )
    # elif args.mode == "gdumb":
    #     from methods.gdumb import GDumb
    #     method = GDumb(
    #         criterion=criterion,
    #         device=device,
    #         train_transform=train_transform,
    #         test_transform=test_transform,
    #         n_classes=n_classes,
    #         **kwargs,
    #     )
    # elif args.mode == "rm":
    #     method = RM(
    #         criterion=criterion,
    #         device=device,
    #         train_transform=train_transform,
    #         test_transform=test_transform,
    #         n_classes=n_classes,
    #         **kwargs,
    #     )
    # elif args.mode == "bic":
    #     method = BiasCorrection(
    #         criterion=criterion,
    #         device=device,
    #         train_transform=train_transform,
    #         test_transform=test_transform,
    #         n_classes=n_classes,
    #         **kwargs,
    #     )
    # elif args.mode == "ewc++":
    #     method = EWCpp(
    #         criterion=criterion,
    #         device=device,
    #         train_transform=train_transform,
    #         test_transform=test_transform,
    #         n_classes=n_classes,
    #         **kwargs,
    #     )
    # elif args.mode == "mir":
    #     method = MIR(
    #         criterion=criterion,
    #         device=device,
    #         train_transform=train_transform,
    #         test_transform=test_transform,
    #         n_classes=n_classes,
    #         **kwargs,
    #     )
    # elif args.mode == "clib":
    #     method = CLIB(
    #         criterion=criterion,
    #         device=device,
    #         train_transform=train_transform,
    #         test_transform=test_transform,
    #         n_classes=n_classes,
    #         **kwargs,
    #     )
    # elif args.mode == "gss":
    #     method = GSS(
    #         criterion=criterion,
    #         device=device,
    #         train_transform=train_transform,
    #         test_transform=test_transform,
    #         n_classes=n_classes,
    #         **kwargs,
    #     )
    # elif args.mode == "der":
    #     method = DER(
    #         criterion=criterion,
    #         device=device,
    #         train_transform=train_transform,
    #         test_transform=test_transform,
    #         n_classes=n_classes,
    #         **kwargs,
    #     )
    elif args.mode == "SDP":
        method = SDP(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )
    else:
        raise NotImplementedError("Choose the args.mode in [er, gdumb, rm, bic, ewc++, mir, clib]")

    return method