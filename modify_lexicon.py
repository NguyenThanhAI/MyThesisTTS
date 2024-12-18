# see: https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/pull/480
import re
lexicon = open("lexicon/librispeech-lexicon.txt").readlines()
sp = re.compile("\s+")
with open("lexicon/modified_librispeech-lexicon.txt", "w") as f:
    for line in lexicon:
        print("line: {}".format(line))
        word, *phonemes = sp.split(line.strip())
        phonemes = " ".join(phonemes)
        f.write(f"{word}\t{phonemes}\n")