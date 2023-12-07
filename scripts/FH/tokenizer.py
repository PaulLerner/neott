
from .basis_funcs import *
import sentencepiece as spm

def tok_sentencePiece(textPath,df = None, name = "franceTerm"):
	modelLoc = dumpPath + "/" + name
	spm.SentencePieceTrainer.train("--input=" + textPath + " --model_prefix=" + modelLoc + " --vocab_size=2000");
	sp = spm.SentencePieceProcessor()
	sp.load(modelLoc + ".model")
	print(sp.encode_as_pieces("This is a test"))
	print(sp.encode_as_ids("This is a test"))
	df["term_tokenized"] = df.term.apply(lambda term: sp.encode_as_pieces(term))
	return df

