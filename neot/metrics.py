from collections import Counter


def preproc(text):
    text = text.lower().strip()
    # TODO strip punct and stop words
    return text


def em(pred, tgt):
    return pred == tgt


def f1(pred, tgt):
    pred_tokens = pred.split()
    tgt_tokens = tgt.split()
    common = Counter(pred_tokens) & Counter(tgt_tokens)
    tp = sum(common.values())
    if tp == 0:
        return 0
    precision = tp / len(pred_tokens)
    recall = tp / len(tgt_tokens)
    return (2 * precision * recall) / (precision + recall)


def compute_metrics(predictions, targets):
    ems, f1s = [], []
    for pred, tgt in zip(predictions, targets):
        pred, tgt = preproc(pred), preproc(tgt)
        ems.append(em(pred, tgt))
        f1s.append(f1(pred, tgt))
    return {
        "em": sum(ems)/len(ems),
        "f1": sum(f1s)/len(f1s),
        "ems": ems,
        "f1s": f1s
    }
