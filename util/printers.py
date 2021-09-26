import torch
import time
import math
from pytorch_pretrained_bert import WEIGHTS_NAME, CONFIG_NAME


def checkpoint(model, output_model_file):
    print('saving checkpoint to', output_model_file)
    # Only save the model itself
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), output_model_file)


def current_timestamp() -> str:
    # timestamp format from https://github.com/tensorflow/tensorflow/blob/
    # 155b45698a40a12d4fef4701275ecce07c3bb01a/tensorflow/core/platform/default/logging.cc#L80
    current_seconds = time.time()
    remainder_micros = int(1e6 * (current_seconds - int(current_seconds)))
    time_str = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(current_seconds))
    return time_str


def get_fps(args):
    output_model_file = args.output_dir + args.model_name + "_"
    if args.prune_type is not None:
        #if args.prune_perc != 0.0:
        #    output_model_file += args.prune_type + "_" + str(args.prune_perc) + "_"
        #else:
        output_model_file += args.prune_type + "_"
        output_model_file += WEIGHTS_NAME
    else:
        output_model_file += WEIGHTS_NAME

    output_config_file = args.output_dir + args.model_name + "_"
    if args.prune_type is not None:
        output_config_file += args.prune_type + "_" + CONFIG_NAME
    else:
        output_config_file += CONFIG_NAME
    output_model_file = output_model_file.replace(" ", "")
    output_config_file = output_config_file.replace(" ", "")
    return output_model_file, output_config_file


def format_log(loss, split, prune_type='', prune_perc=''):
    log_str = '{2} - {0} - {1} | {0} loss {3:5.2f} | {0} ppl {4:9.3f} '.format(
        prune_type, prune_perc, split, loss, math.exp(loss))
    return log_str


def display_loss(log_str, valid_loss=None, test_loss=None, prune_type='', prune_perc=''):
    if valid_loss is not None:
        log_str += format_log(valid_loss, 'valid',
                              prune_type=prune_type, prune_perc=prune_perc)
    if test_loss is not None:
        log_str += format_log(test_loss, 'test',
                              prune_type=prune_type, prune_perc=prune_perc)
    return log_str + "\n"
