# coding=utf-8
import codecs
from .basis_funcs import *;
from .daille import getMorphoType
from .process_xml import get_mainUnits

def df_to_derif(df):
	lines = set()
	nlp = loadNlp()
	print("mapping to derif with steaze",df)
	def writeWordToDerif(row):
		# print(row)
		if row.name % 100 == 0:
			print(row.name/len(df));
		print("seeyun",row.term,row.basic_pos)
		if row.basic_pos != "":
			tt = ""
			term = row.term.replace(r"œ","oe").replace("’","'").replace('\u2033',"''")
			_, morphoType = getMorphoType(term)
			if type(morphoType) == type(None):
				#simple term, no problemo
				if row.basic_pos == "noun":
					tt = "NOM"
				elif row.basic_pos == "adj":
					tt = "ADJ"
				elif row.basic_pos == "verb":
					tt = "VERBE"
				elif row.basic_pos == "adv":
					tt = "ADV"
				else:
					if nlp(term)[0].pos_ in spacyPOSDict:
						tt = spacyPOSDict[nlp(term)[0].pos_]
					else:
						print("defaulting to noun just cuz",term)
						tt = "NOM"
				try:
					lines.add(term+","+tt)
				except Exception as e:
					print("flubbing on",term,tt)
					raise(e)
			else:
				doc = nlp(term)
				for tok in doc:
					if tok.pos_ in ["NOUN","ADJ","ADV","VERB"]:
						lines.add(str(tok) + "," + spacyPOSDict[tok.pos_])
	fp = codecs.open(dataPath + "/derifTerms.txt", "w","ISO-8859-1")
	df.apply(writeWordToDerif,axis=1)
	for line in list(sorted(lines)):
		try:
			fp.write(line + "\n");
		except:
			print("can't write",line);
	fp.close()
	print("wrote to",dataPath + "/derifTerms.txt")

def process_derif_analysis(analysis):
	try:
		steps = analysis.find("Analyses").find("Analyse").find("Steps").findall("Step")
	except:
		# no DeriF analysis for this term!
		return None, set()
	steps = sorted(steps, key = lambda step: int(step.attrib["number"]))
	decidingStep = steps[0]
	morphType = decidingStep.find("MorphologicalProcessType").text
	toks = set()
	for tag in ("MorphologicalProcess","Base"):
		tok = decidingStep.find(tag).text
		if type(tok) != type(None) and len(tok) > 0:
			if tag == "MorphologicalProcess":
				if morphType == "suf":
					toks.add("-" + tok)
				elif morphType == "pre":
					toks.add(tok + "-")
				else:
					toks.add(tok)
	if morphType in ("pre","suf","???"):
		morphTypeRet = (morphType+"fix" if not morphType == "???" else "fix")
		toksRet = toks
	elif morphType == "conv":
		morphTypeRet = morphType
		toksRet = set()
		toksRet.add(decidingStep.find("Derived").text)
	elif morphType == "comp":
		morphTypeRet = morphType
		toksRet = toks
	else:
		morphTypeRet = None
		toksRet = set()
	return morphTypeRet, toksRet


def incorporate_derif_etym(df, corpusName, suffix = "", genDerif = False):
	'''
	TODO:
	For each term:
		• Get its component tokens re DeriF
		• Get its construction type re DeriF
	:param df:
	:param corpusName:
	:return:
	'''
	print("loading spacy")
	try:
		type(nlp)
		print("cha")
	except:
		nlp = loadNlp()
	xmlPath = dataPath + "/" "derif" + "/" + corpusName + "Derif.xml"
	try:
		singleTerms = get_mainUnits("DerifResult", xml_path=xmlPath)
		if genDerif:
			raise Exception("Do derif tho")
	except:
		print("no derif terms exist for this corpus, re-mapping with steaze (with style and ease)",xmlPath)
		df_to_derif(df)
		exit()
	lemmaToAnalysis = {term.find("Lemme").text + term.find("Category").text: term for term in singleTerms}
	print("keys knees",[x for x in lemmaToAnalysis.keys() if "gha" in x]);
	dumpVar(lemmaToAnalysis,"lemmaToAnalysis","","")
	dumpVar(singleTerms, "singleTerms", "", "")
	allTerms = loadVar("wordSet", "", "_allWords");
	print("lalennifer jedanniston",len(allTerms))
	classicalFixes = set([x for x in loadVar("classicalPrefixes","","") if len(x) >= 2])
	allKnownWiktFixes = loadVar("allFixesWikt", "", "")
	# equivs = termXml.find("Analyses").findall("Analyse")
	def apply_derif_and_morpho_results(row):
		if row.name % 100 == 0:
			print(row.name/len(df))
		# print("origterm",row.term);
		term = row.term.replace(r"œ", "oe").replace("’", "'").replace('\u2033', "''")
		morphoTok, morphoType = getMorphoType(term)
		print("we doin", term, morphoType)



		if type(morphoType) == type(None):
			# simple term, no problemo
			if row.basic_pos == "noun":
				tt = "NOM"
			elif row.basic_pos == "adj":
				tt = "ADJ"
			elif row.basic_pos == "verb":
				tt = "VERBE"
			elif row.basic_pos == "adv":
				tt = "ADV"
			else:
				if nlp(term)[0].pos_ in spacyPOSDict:
					tt = spacyPOSDict[nlp(term)[0].pos_]
				else:
					print("defaulting to noun just cuz", term)
					tt = "NOM"

			if not (term + tt) in lemmaToAnalysis and term.lower() + tt in lemmaToAnalysis:
				analysis = lemmaToAnalysis[term.lower() + tt]
			elif (term + tt) in lemmaToAnalysis:
				analysis = lemmaToAnalysis[term + tt]
			else:
				print(term,"How am I not in that movie!", term, tt, row.basic_pos)
				morphoType, toks = None, set()
				analysis = None
			if type(analysis) != type(None):
				morphoType, toks = process_derif_analysis(analysis)
			realToks = set()
			if not type(toks) == type(None):
				def processTok(tok):
					print("ima processin",tok)
					subToks = tok.split(":")
					realTok = [subTok for subTok in subToks if not "*" in subTok][0]
					realToks.add(realTok)
				for tok in toks:
					processTok(tok)
				toks = realToks
			print("morphological",term,morphoType,morphoTok,"hinei toks",toks)
			extraToks = findExtraToks(term, allTerms,classicalFixes)

			nativeCompounds = []
			for index in range(3, len(term) - 3):
				secondHalf = term[index:]
				firstHalf = term[:index]
				if not ("-" + secondHalf in allKnownWiktFixes or firstHalf + "-" in allKnownWiktFixes):
					if firstHalf in allTerms and secondHalf in allTerms:
						nativeCompounds.extend([firstHalf, secondHalf])
			# toks = toks.union(extraToks)
		elif morphoTok == "-":
			twoToks = term.split("-");
			if not (twoToks[0] in allKnownWiktFixes or twoToks[1] in allKnownWiktFixes):
				nativeCompounds = [twoToks[0],twoToks[1]];
			else:
				nativeCompounds = []
			toks = nativeCompounds
			extraToks = set()
		else:
			#It's a syntagmic compound
			toks = set(term.split(morphoTok))
			extraToks = set()
			nativeCompounds = []
		print("retting",toks,morphoType,morphoTok,extraToks)

		return toks, morphoType, morphoTok, extraToks, nativeCompounds
	df[["subToks_derif","morpho_type_derif","morpho_tok","extraToks", "nativeToks"]] = df.apply(lambda row:
								apply_derif_and_morpho_results(row),axis=1,result_type='expand')
	print("post malone",df)
	print(df.columns)
	print("snuffix",suffix)
	dumpVar(df, "dfFinal", "combined", suffix)
	return df



def get_english_etymologies():
	#todo@feh: this lol
	import lxml.html as lh
	import requests
	for word in ["preordain","gregarity now"]:
		word = word.replace(" ","+").replace("'","%27")
		req = requests.get('https://www.etymonline.com/search?q=' + word)
		elem = lh.fromstring(req.text)
	elem.getchildren()





'''
TODO@feh:
finish logic for language of origin for Daille classification
start some fine-tuning
'''