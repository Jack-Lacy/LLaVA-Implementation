# HowMany-QA data splits

HowMany-QA is a combination of counting questions from popular visual question answering datasets.
It is introduced in:

    Alexander Trott, Caiming Xiong, & Richard Socher. "Interpretable
    Counting for Visual Question Answering" in ICLR (2018)

The associated file question_ids.json provides the lists of question IDs used in training,
development, and testing. These IDs reference the questions found in the "VQA 2.0" and "Visual Genome"
datasets, which must be downloaded separately.

Development and testing questions are taken exclusively from VQA 2.0. Training questions are taken from
both. The training IDs are separated according to the source dataset.
