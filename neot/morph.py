# -*- coding: utf-8 -*-
from typing import List


class Term:    
    def __init__(self, term: str, pos=None):
        self.term = term
        self.pos = pos
    
    def __str__(self):
        return self.term
    
    def __repr__(self):
        return self.__str__()
    
    def signature(self):
        return "[M]"
    
    
class Inflected(Term):
    def __init__(self, stem: Term, inflection: str, *args, **kwargs):
        self.stem = stem
        self.inflection = inflection
        super().__init__(*args, **kwargs)
        
    def __str__(self):
        return f"{self.stem}-{self.inflection}"
    
    def signature(self):
        # discount inflections
        return self.stem.signature()


class Prefixed(Term):
    def __init__(self, stem: Term, prefix: str, *args, **kwargs):
        self.stem = stem
        self.prefix = prefix
        super().__init__(*args, **kwargs)
        
    def __str__(self):
        return f"{self.prefix}-{self.stem}"
    
    def signature(self):
        return f"[P{self.stem.signature()}]"
        
        
class Suffixed(Term):
    def __init__(self, stem: Term, suffix: str, *args, **kwargs):
        self.stem = stem
        self.suffix = suffix
        super().__init__(*args, **kwargs)
        
    def __str__(self):
        return f"{self.stem}-{self.suffix}"
    
    def signature(self):
        return f"[S{self.stem.signature()}]"
    

class Converted(Term):
    def __init__(self, stem: Term, *args, **kwargs):
        self.stem = stem
        super().__init__(*args, **kwargs)
        # self.term == self.stem.term
        # self.pos != self.stem.pos
        
    def signature(self):
        return f"[0{self.stem.signature()}]" 
    
        
class Compound(Term):
    def __init__(self, stem_l: Term, stem_r: Term, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __str__(self):
        return f"{self.stem_l}|{self.stem_r}"
    
    def signature(self):
        return f"[{self.stem_l}.signature(){self.stem_r}.signature()]" 
    

class Neoclassical(Compound):
    def signature(self):
        return f"[C{self.stem_l}.signature(){self.stem_r}.signature()]" 


class Native(Compound):
    def signature(self):
        return f"[N{self.stem_l}.signature(){self.stem_r}.signature()]" 


class Syntagmatic(Term):
    def __init__(self, terms: List[Term], *args, **kwargs):
        self.terms = terms
        super().__init__(*args, **kwargs)
        
    def __str__(self):
        return " ".join(t.__str__() for t in self.terms)
    
    def signature(self):
        return f"[T{''.join(t.signature() for t in self.terms)}]" 

    
        
if __name__ == "__main__":
    ps = Suffixed(
        term="retournement", 
        pos="NOUN", 
        suffix="ment",
        stem=Prefixed(
            term="retourner", 
            pos="VERB",
            prefix="re", 
            stem=Term(
                "tourner", 
                "VERB"
            )
        )
    )
    print(ps)