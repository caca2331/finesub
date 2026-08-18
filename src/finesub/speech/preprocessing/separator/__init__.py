"""The vocal separator: the stage, its compile cache, and the AOTI packages.

Three modules that only ever talk to each other -- `separation` runs the stage,
`accel` decides and caches which compiled backend this machine gets, and
`separator_aoti` builds and loads the packages `accel` chooses. They sat beside
the VAD and the audio readers, which share nothing with them.
"""
