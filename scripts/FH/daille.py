
from .basis_funcs import *
try:
	from nltk.corpus import words
	nltkEnglishWords = set(words.words())
except:
	print("damn conda...")

def analyzeClassicalFixDist(df, suffix = ""):
	outLoc = imgOutLoc + "/" + "basicVis"
	numTake = 35
	for dataType, columnName in [("Daille-types","daille_type"),("classical affixes","classicalFixes")]:
		if columnName == "daille_type":
			fig = plt.figure(figsize=(10, 6), dpi=80)
		else:
			fig = plt.figure(figsize=(14, 9), dpi=80)
		fig.tight_layout()
		np.random.seed(42)
		firstFixList = []
		for corpusIndex, corpus in enumerate(("all","wiktionnaire","ft")):
			print("corping",corpus)
			if not corpus == "all":
				dfThis = df[df.source==corpus];
			else:
				dfThis = df
			affixFreqDict = {}
			def populateAffixFreqDict(row):
				colVals = row[columnName]
				if type(row[columnName]) == str:
					colVals = [colVals]
				for fix in colVals:
					if "set()" in fix: continue
					if not fix in affixFreqDict:
						affixFreqDict[fix] = 0
					affixFreqDict[fix] += 1;
			dfThis.apply(lambda row: populateAffixFreqDict(row),axis=1)
			fixList = sorted([(fix, count) for fix, count in affixFreqDict.items()], key = lambda tup: tup[1], reverse=True)
			print("applicado", fixList)
			if len(firstFixList) == 0:
				firstFixList = [tup[0] for tup in fixList]
				print("ffl",firstFixList)
			else:
				justFixes = [tup[0] for tup in fixList]
				for x in firstFixList:
					if not x in justFixes:
						print("appending",x,0)
						fixList.append((x,0))
				justFixes = [tup[0] for tup in fixList]
				idx = [justFixes.index(fix) for fix in firstFixList[:len(justFixes)]]
				fixList = [fixList[i] for i in idx]
				# print("normy",fixList,idx)
			allFixCount = np.sum([tup[1] for tup in fixList])
			allFixCount25 = np.sum([tup[1] for tup in fixList[:numTake]])
			fixList = [(tup[0],tup[1]/allFixCount) for tup in fixList]
			fixList = fixList[:numTake]
			color = generateRandomColor()
			ratio25 = round(allFixCount25 / allFixCount, 2)
			plt.bar(np.arange(len(fixList))+corpusIndex/4, [tup[1] for tup in fixList], color=color,label=corpus + (" ("+str(ratio25)+")" if ratio25 < 1 else ""),width=1/4)
		plt.ylabel("Percentage over all " + dataType, fontsize=17)
		firstFixList = firstFixList[:numTake]
		plt.xticks(np.arange(len(firstFixList)) + 1 / 4, [fix for fix in firstFixList], rotation=90, fontsize=17)
		plt.title("Distribution of " + str(len(firstFixList)) + " most common " + dataType + " in corpus", fontsize=20)  # ("+str(100*round(percMissing,2))+"% missing)")
		plt.legend()
		fig.tight_layout()
		picPath = outLoc + "/" + columnName + "_dist" + suffix + ".png"
		print("saving image to",picPath)
		plt.savefig(picPath)
		plt.close()


def hasNoNeoAffix(term):
	'''
	For each non-neoclassical affix, determine if it is a sub-string of the term
	:param term:
	:return:
	'''
	for fix in noNeoFixesLen:
		if fix[0] == "-":
			fix = fix[1:]
			if term[-1*(len(fix)):] == fix:
				return True
		else:
			fix = fix[:-1]
			if term[:len(fix)] == fix:
				return True;
	return False




def get_daille_type(df, suffix):
	'''
	This function determines the daille type for every term in the dataframe
		*
	:param df: dataframe of terms to analyze
	:param suffix: useful for dumping results
	:return:
	'''

	#this method aggregates all classical pre- and suffixes scraped from various sources
	#and discards those which have become sufficiently french-ified
	knownClassicalFixes = getKnownClassicalFixes();

	#all affixes, whether classical or otherwise
	allKnownWiktFixes = loadVar("allFixesWikt", "", "")

	#all terms from Wiktionnaire, neonyms and otherwise
	allTerms = loadVar("wordSet", "", "_allWords");

	def get_daille_type_row(row):
		classicalFixes = set()
		# morphological token is the token separating various neonym segments, for neoclassical- and native compounding ("à" or "de", for example)
		if type(row.morpho_tok) == type(None):
			#no morphological token means that the term is a single unit, not broken up by spaces or other separating characters
			allFixesForTerm = set()

			### START OF EASY CLASSIFICATION CRITERIA

			#first glean all affixes pertinent to this term
			# 1: Take all sub-tokens returned by Derif that have a - in them
			if not type(row.subToks_derif) == float:
				allFixesForTerm = allFixesForTerm.union(set([x for x in row.subToks_derif if "-" in x]))

			#2: take all affixes that Wiktionnaire has for this term
			if not type(row.fixes_wikt) == float:
				allFixesForTerm = allFixesForTerm.union(set([x for x in row.fixes_wikt]))

			#3: take all extra tokens which were recognized as prefixes, based on the following criteria:
				# • either have an affix of sufficient length that is known as an affix (based on my dictionary of affixes)
				# • have an affix of short length such that the remainder is also in the set of all Wiktionnaire terms
			if not type(row.extraToks) == float:
				allFixesForTerm = allFixesForTerm.union(set([x for x in row.extraToks]))

			#retain only classical affixes
			classicalFixes = allFixesForTerm.intersection(knownClassicalFixes)


			#the following are a list of rules, whose order I deemed logical, but which is certainly not superior to any other manually constructed
			# decision tree

			# IF the term has a classical affix, THEN it is neoclassical
			if len(classicalFixes) > 0:
				retName= "neoClass"
			# IF the term has an affix which is not neoclassical, THEN it is an affix term
			elif len(allFixesForTerm.intersection(noNeoFixes)) > 0:
				retName = "affix"
			# IF DeriF determined that it was a conversion term (grammatical category changed, word syntax remains unperturbed), THEN it is a conv term
			elif row.morpho_type_derif == "conv":
				retName = "conv"
			# IF the term is comprised of more than one subtoken which is also a Wiktionnaire word, THEN it is a native compounding
			elif len(row.nativeToks) > 1:
				retName = "native"
			# IF the term is a mot valise (based on Wiktoinnaire), THEN it is a native compounding
			elif row.isMotValise:
				retName = "native"
			# IF Wiktionnaire specifies that the word was borrowed, THEN it is of type borrow
			elif row.isBorrowed:
				retName = "borrow"
			# IF the row has an English equivalent, specified by Wiktionnaire in the "Traduction" section at the bottom of some Wiktionnaire pages,
			# THEN it is borrowed
			elif (row.englishEquiv is not None and len(row.englishEquiv) > 0):
				retName = "borrow"
			# IF the word has a language of descent, specified by Wiktoinnaire, that is neither latin nor greek
			elif (len(row.descLang) > 0 and not row.descLang in ("la","grc")):
				retName = "borrrow"
			# IF the word is in the NLTK list of english words
			elif row.term in nltkEnglishWords:
				retName = "borrow"
			else:
				### END OF EASY CLASSIFICATION CRITERIA
				#if no criterion has yet been matched, rather than throw up our hands and say unknown, we
				# search for further evidence of affixes which were not known by Wiktionnaire (or perhaps were off by a letter, and thus didn't match
				# exactly, but are morphologically equivalent)

				#consider each DeriF subtoken that is NOT an affix; perhaps DeriF gave us a subtoken without denoting it as an affix,
				# based on, for example, its not being at the beginning of the word, but which still qualifies as an affixation upon further scrutiny
				derifCompTokenMaybePrefixes = set([x for x in row.subToks_derif if not "-" in x])
				derifCompTokenMaybePrefixes = set([y for x in derifCompTokenMaybePrefixes if not "-" in x for y in (x + "-", "-" + x)])

				#determine respective classical, non-classical affixes
				classicalDerifCompTokenMaybePrefixes = 	derifCompTokenMaybePrefixes.intersection(knownClassicalFixes)
				nonClassicalDerifCompTokenMaybePrefixes = derifCompTokenMaybePrefixes.intersection(noNeoFixes)

				# IF there is a classical affix, THEN denote the term neoclassical compounding
				if len(classicalDerifCompTokenMaybePrefixes) > 0:
					classicalFixes = classicalDerifCompTokenMaybePrefixes
					retName = "neoClass"
				# IF the term has a non-classical affix, THEN denote as affix
				elif len(nonClassicalDerifCompTokenMaybePrefixes) > 0:
					retName = "affix"
				# IF the term has an affix denoted in our dictionary of all affixes (this should be redundant, but it is a fail-safe in case some
				# logic failed previously), THEN it is an affix
				elif len(allFixesForTerm.intersection(allKnownWiktFixes)) > 0:
					retName = "affix"
				# IF the term has a non-classical affix based on the algorithm described in the below function, THEN it is an affix
				# (Again, this should be redundant, but is a fail-safe)
				elif hasNoNeoAffix(row.term):
					retName = "affix"
				else:
					### Still no match has been made; we now consider whether there might be a conversion identifiable

					# IF the word is used for at least one other meaning on Wiktionnaire (réseau, for example), THEN it is a conversion
					# (Note: this might lead to an over-classification of conversions; this is the reason why this clause is only considered
					# after none of the previous criteria have held)
					if (type(row.otherCatLens) != str) and row.otherCatLens > 1:
						retName = "conv";
					# IF the term ends with an é, denoting past tense for common +er verbs, and its equivalent +er is also a Wiktionnaire term,
					# THEN it is a conv. (Note: such logic could be expanded for other verb/word classes)
					elif row.term[-1] == "é" and row.term[:-1]+"er" in allTerms:
						retName = "conv"
					# IF no other criterion has been met, THEN the term is of unknown daille type.
					# Note: each time I (@Felix) looked at the terms with unknown daille type, I had an idea for another sub-method
					# to catch a few more. This tree can thus surely be widened, particularly by someone whose knowledge of French
					# is better than mine
					else:
						retName = "UNKNOWN"
		# IF the term is separated by a hyphen, THEN it is some type of compounding. We analyze its components.
		elif row.morpho_tok == "-":

			#Consider whether either component is a neoclassical affix
			toks = row.term.split("-");
			fixes = {toks[0] + "-", "-" + toks[1]}
			# IF a component is a neoclassical affix, THEN denote as neoclassical
			if len(fixes.intersection(knownClassicalFixes)) > 0:
				retName = "neoClass"
			# IF a component is a non-neoclassical affix, THEN denote as affix
			elif len(fixes.intersection(noNeoFixes)) > 0:
				retName = "affix";
			# The compounding is neither neoclassical nor affix; thus, it must be native, according to our daille tree
			else: retName = "native"
		else:
			# The term is separated by spaces; this is a syntagmatic compounding.
			#Note: this category could certainly be further fleshed out
			retName = "syntag"
		return retName, classicalFixes

	df[["daille_type","classicalFixes"]] = df.apply(get_daille_type_row,axis=1,result_type='expand')

	def reassign_daille_types(row):
		'''
		This function is used to catch any final terms which escaped with UNKNOWN daille type;
		After the initial assigning, for every term that was classified as UNKNOWN, we consider whether it has a buddy
		of identical spelling and POS with a not-UNKNOWN daille type (primarily for cases where a Wiktoinnaire term was classified
		but its corresponding FranceTerme was not, or vice-versa). Assign the unknown term the most common tag given to all of these
		buddies.
		:param row:
		:return:
		'''
		if row.daille_type != "UNKNOWN":
			return row.daille_type
		otherRows = df[(df.term==row.term) & (df.basic_pos == row.basic_pos) & (df.index!=row.name) & (df.daille_type != "UNKNOWN")]
		if len(otherRows) > 0:
			return otherRows.groupby("daille_type")["daille_type"].count().sort_values(ascending=False).index[0]
		return "UNKNOWN"


	df["daille_type"] = df.apply(lambda row: reassign_daille_types(row),axis=1)

	dumpVar(df, "dfFinal", "combined", suffix)
	return df;


def getMorphoType(term):
	de = re.search("\sd(?:'|es{0,1}\s)",term)
	aGrave = re.search("\saux{0,1}\s|à l(?:a\s|')|au\s", term)
	hyphen = re.search("-", term)
	space = re.search("\s", term)
	for splitIndex, (splitType, splitName) in enumerate(((de,"de"), (aGrave,"aGrave"), (hyphen,"hyphen"), (space,"space"))):
		if type(splitType) != type(None):
			splitter = splitType.group(0)
			return splitter, splitName
	return None, None

def analyzeDailleResults():
	pass


# def getCompoundingType(toks):
# 	origLangs = set((getOrigLang(tok) for tok in toks))
# 	#if latin or greek and french, it's neoclassical
# 	if "la" in origLangs or "grc" in origLangs or "latin":
# 		if "fr" in origLangs:
# 			return "neo"
# 	#if only french, it's native
# 	if "fr" in origLangs and len(origLangs) == 1:
# 		return "nat"
