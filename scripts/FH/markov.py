
from .basis_funcs import *;

def initMarkov(df):
	mm = MarkovModel(df);
	df["markovLikelihood"] = df.apply(lambda row: mm.getLikelihood(row.term),axis=1)
	return df, mm


class MarkovModel():
	def __init__(self, df, upToOrder = 2, quickie = False):
		self.notSeenPunishment = 100
		self.markovDicts = {}
		self.upToOrder = upToOrder
		thisDf = df[df.repn == 0]
		if quickie:
			thisDf = thisDf[:500]
		self.minProb = 1
		self.generateMarkovModel(thisDf, upToOrder)
		self.compile()
	def mapStr(self, str):
		str = re.sub("[0-9]","0",str)
		str = re.sub("[^a-zA-Z0-9.\-\s]","/",str)
		return str
	def generateMarkovModel(self,df, upToOrder = 3):
		print("generating")
		upToOrder = min(self.upToOrder, upToOrder);
		for order in range(upToOrder+1):
			markovDict = {}
			def addTermToMarkovDict(term, order):
				print("adding",term)
				for i in range(order,len(term)):
					nextChar = self.mapStr(term[i])
					prefix = self.mapStr(term[max(i-order,0):i])
					if not prefix in markovDict:
						markovDict[prefix] = {}
						markovDict[prefix]["__totCountForPrefix__"] = 0
					if not nextChar in markovDict[prefix]:
						markovDict[prefix][nextChar] = 0
					markovDict[prefix]["__totCountForPrefix__"] += 1
					markovDict[prefix][nextChar] += 1;
			df[df.repn==0].apply(lambda row: addTermToMarkovDict(row.term, order),axis=1);
			self.markovDicts[order] = markovDict
		print("generated",self.markovDicts)
	def compile(self):
		print("compiling")
		self.minProb = 1
		for order, d in self.markovDicts.items():
			totCountsForOrder = 0
			for prefix in d.keys():
				for nextChar in d[prefix].keys():
					if nextChar == "__totCountForPrefix__": continue;
					print("before",d[prefix][nextChar],d[prefix]["__totCountForPrefix__"])
					d[prefix][nextChar] /= d[prefix]["__totCountForPrefix__"]
					print("afta", d[prefix][nextChar])
					self.minProb = min(self.minProb, d[prefix][nextChar])
					totCountsForOrder += d[prefix]["__totCountForPrefix__"]
			d["__totCountForOrder__"] = totCountsForOrder
		print("compiled",self.markovDicts)
	def getLikelihood(self,term, upToOrder=-1, verbose = False):
		if upToOrder == -1:
			upToOrder = self.upToOrder
		allProbs = []
		for order in range(upToOrder+1):
			if verbose:
				print("order",order)
			for i in range(order, len(term)):
				nextChar = self.mapStr(term[i])
				prefix = self.mapStr(term[max(i - order, 0):i])
				try:
					prob = self.markovDicts[order][prefix][nextChar]*self.markovDicts[order][prefix]["__totCountForPrefix__"]/self.markovDicts[order]["__totCountForOrder__"]
				except:
					try:
						minProb = np.min(list(self.markovDicts[order][prefix].values()))/(self.markovDicts[order]["__totCountForOrder__"]*self.notSeenPunishment)
					except:
						minProb = self.minProb/(self.markovDicts[order]["__totCountForOrder__"]*self.notSeenPunishment)
					prob = minProb
				if verbose:
					print("|"+prefix+"|", "|"+nextChar+"|", prob)
				allProbs.append(prob)
		if len(allProbs) == 0:
			totProb = np.infty
		else:
			totProb = -1*np.mean(np.log(allProbs));
		return totProb









