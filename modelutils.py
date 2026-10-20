import torch
import torch.nn as nn


DEV = torch.device('cuda:0')


def get_device(name):
    '''Return the torch device for `name`; any name starting with "xla" selects the PyTorch/XLA (TPU) device.'''
    if name.startswith('xla'):
        import torch_xla
        return torch_xla.device()
    return torch.device(name)


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def device_sync(dev):
    '''Wait for pending work on `dev`: cut and run the lazy graph on XLA, synchronize on CUDA.'''
    if dev.type == 'xla':
        import torch_xla
        torch_xla.sync(wait=True)
    elif dev.type == 'cuda':
        torch.cuda.synchronize()


def mark_step(dev):
    '''On XLA, cut the lazy graph here so loops do not build one ever-growing graph; no-op elsewhere.'''
    if dev.type == 'xla':
        import torch_xla
        torch_xla.sync()


def log_xla_metrics(dev, tag, report_path=None):
    '''On XLA, print the compile count and every aten:: CPU-fallback counter; optionally save the full report.'''
    if dev.type != 'xla':
        return
    import torch_xla.debug.metrics as met
    compile_data = met.metric_data('CompileTime')
    fallbacks = {n: met.counter_value(n) for n in met.counter_names() if n.startswith('aten::')}
    print(f"[xla-metrics] {tag} compiles={compile_data[0] if compile_data else 0} fallbacks={fallbacks}", flush=True)
    if report_path:
        with open(report_path, 'w') as f:
            f.write(met.metrics_report())
