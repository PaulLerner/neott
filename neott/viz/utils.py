import matplotlib.pyplot as plt
import numpy as np


def pie(counter, rotatelabels=True, rotatepct=True, **kwargs):
    def annote_total(pct):
        return f"{(pct / 100) * total:,.0f}"
    total = sum(counter.values())
    patches, labels, pct_texts = plt.pie(
        list(counter.values()),
        labels=counter,
        autopct=annote_total,
        rotatelabels=rotatelabels,
        **kwargs
    )
    if rotatepct:
        for label, pct_text in zip(labels, pct_texts):
            pct_text.set_rotation(label.get_rotation())
    return patches, labels, pct_texts


def decile_eg(data):
    deciles = np.quantile(data, np.arange(0, 1.1, 0.1), method="nearest")
    for decile in deciles:
        yield np.where(data == decile)[0][0]
