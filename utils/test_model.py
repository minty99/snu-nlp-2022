import os

import torch

from utils.utils import calc_metric, get_logger, get_pytorch_kobert_model, is_logging_process
from utils.writer import Writer
from functools import lru_cache


@lru_cache(maxsize=None)
def get_vocab():
    _, vocab = get_pytorch_kobert_model()
    return vocab


def test_model(cfg, model, test_loader, writer: Writer):
    logger = get_logger(cfg, os.path.basename(__file__))
    model.net.eval()
    total_test_loss = 0
    test_loop_len = 0
    vocab = get_vocab()
    texts = dict()
    with torch.no_grad():
        for data in test_loader:
            model_input, model_target, qa_id = data[:3], data[3:5], data[5]
            output = model.inference(model_input)
            # Calculate test_loss
            model_target = torch.stack([item.to(cfg.device) for item in model_target], dim=-1)
            loss_v = model.loss_f(output, model_target.to(cfg.device))
            if cfg.dist.gpus > 0:
                # Aggregate loss_v from all GPUs. loss_v is set as the sum of all GPUs' loss_v.
                torch.distributed.all_reduce(loss_v)
                loss_v /= torch.tensor(float(cfg.dist.gpus))
            total_test_loss += loss_v.to("cpu").item()

            # Calculate QA metrics
            output = output.mT  # [batch, num_words, 2] -> [batch, 2, num_words]
            pred = torch.argmax(output, dim=-1)  # [batch, 2]
            batch_size = model_input[0].shape[0]
            for i in range(batch_size):
                input_text = model_input[0][i]
                pred_start, pred_end = pred[i]
                pred_text = input_text[pred_start : pred_end + 1].tolist()
                decoded_text = vocab.decode(pred_text)
                texts[qa_id[i]] = decoded_text

            test_loop_len += 1

        total_test_loss /= test_loop_len
        metric = calc_metric(cfg, texts)

        if writer is not None:
            writer.logging_with_step(total_test_loss, model.step, "test_loss")
            writer.logging_with_step(metric["exact_match"], model.step, "EM")
            writer.logging_with_step(metric["f1"], model.step, "F1")
        if is_logging_process():
            logger.info(
                "Test Loss %.04f / EM %.04f / F1 %.04f at step %d"
                % (total_test_loss, metric["exact_match"], metric["f1"], model.step)
            )
