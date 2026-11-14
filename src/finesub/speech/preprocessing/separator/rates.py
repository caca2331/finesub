"""The rates the separator may be told to decode and emit at.

Its own module, torch-free, because the pipeline's argument parser offers them
as choices: reading them from `separation` cost every `--help` the torch import.
"""

#: 44100 is the model's own and the only one its weights were trained for; the
#: other two buy fewer chunks per second of audio -- the chunk is a fixed 352800
#: samples, so halving the rate halves the chunk count -- and pay for it in
#: separation quality. Neither is recommended; see docs/separator-optimization.md
#: "E12" for what each costs and why 16000 is not on this list at all.
SEPARATOR_SAMPLE_RATES = (44100, 32000, 22050)
DEFAULT_SEPARATOR_SAMPLE_RATE = 44100
