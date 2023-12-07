
from .basis_funcs import *

def convertPrefixPDFToList():
lines = open("/Users/f002nb9/Downloads/nose.txt").read()
fixes = re.findall("[a-z]+-",lines) + re.findall("-[a-z]+",lines)
bothFixes  = set(re.findall("-[a-z]+-",lines))
for reapp in ("-midi","-plos","-clus","-scen","ré-","dé-","télé"):
	fixes.append(reapp)
	dumpVar(fixes,"fixes","","")




# ubiquitousAffixes =