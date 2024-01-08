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
    
    def __len__(self):
        return 1
    
    def to_dict(self):
        d = self.__dict__.copy()
        d["_class"] = self.__class__.__name__
        for k, v in d.items():
            if isinstance(v, Term):
                d[k] = v.to_dict()
        return d
    
    @classmethod
    def from_dict(cls, *args, **kwargs):
        return cls(*args, **kwargs)    
    
    
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
    
    def __len__(self):
        return 1 + len(self.stem)
    
    @classmethod
    def from_dict(cls, stem: dict, *args, **kwargs):
        Class = CLASSES[stem.pop("_class")]
        stem = Class(**stem)
        return cls(*args, stem=stem, **kwargs)
    

class Prefixed(Term):
    def __init__(self, stem: Term, prefix: str, *args, **kwargs):
        self.stem = stem
        self.prefix = prefix
        super().__init__(*args, **kwargs)
        
    def __str__(self):
        return f"{self.prefix}-{self.stem}"
    
    def signature(self):
        return f"[P{self.stem.signature()}]"
    
    def __len__(self):
        return 1 + len(self.stem)
    
    @classmethod
    def from_dict(cls, stem: dict, *args, **kwargs):
        Class = CLASSES[stem.pop("_class")]
        stem = Class(**stem)
        return cls(*args, stem=stem, **kwargs)
        
        
class Suffixed(Term):
    def __init__(self, stem: Term, suffix: str, *args, **kwargs):
        self.stem = stem
        self.suffix = suffix
        super().__init__(*args, **kwargs)
        
    def __str__(self):
        return f"{self.stem}-{self.suffix}"
    
    def signature(self):
        return f"[S{self.stem.signature()}]"
    
    def __len__(self):
        return 1 + len(self.stem)
    
    @classmethod
    def from_dict(cls, stem: dict, *args, **kwargs):
        Class = CLASSES[stem.pop("_class")]
        stem = Class(**stem)
        return cls(*args, stem=stem, **kwargs)
    

class Converted(Term):
    def __init__(self, stem: Term, *args, **kwargs):
        self.stem = stem
        super().__init__(*args, **kwargs)
        # self.term == self.stem.term
        # self.pos != self.stem.pos
        
    def signature(self):
        return f"[0{self.stem.signature()}]" 
    
    def __len__(self):
        return len(self.stem)
    
    @classmethod
    def from_dict(cls, stem: dict, *args, **kwargs):
        Class = CLASSES[stem.pop("_class")]
        stem = Class(**stem)
        return cls(*args, stem=stem, **kwargs)
    
        
class Compound(Term):
    def __init__(self, stem_l: Term, stem_r: Term, *args, **kwargs):
        self.stem_l = stem_l
        self.stem_r = stem_r
        super().__init__(*args, **kwargs)
        
    def __str__(self):
        return f"{self.stem_l}|{self.stem_r}"
    
    def signature(self):
        return f"[{self.stem_l.signature()}{self.stem_r.signature()}]" 
    
    def __len__(self):
        return len(self.stem_l) + len(self.stem_r)
    
    @classmethod
    def from_dict(cls, stem_l: dict, stem_r: dict, *args, **kwargs):
        L = CLASSES[stem_l.pop("_class")]
        stem_l = L(**stem_l)
        R = CLASSES[stem_r.pop("_class")]
        stem_r = R(**stem_r)
        return cls(*args, stem_l=stem_l, stem_r=stem_r, **kwargs)


class Neoclassical(Compound):
    def signature(self):
        return f"[C{self.stem_l}.signature(){self.stem_r}.signature()]" 


class Native(Compound):
    def signature(self):
        return f"[N{self.stem_l}.signature(){self.stem_r}.signature()]" 


CLASSES = {Term.__name__: Term} | {c.__name__: c for c in Term.__subclasses__()}


class Syntagmatic:
    def __init__(self, terms: List[Term]):
        self.terms = terms
        
    def __str__(self):
        return " ".join(t.__str__() for t in self.terms)
    
    def signature(self):
        return f"[T{''.join(t.signature() for t in self.terms)}]" 
    
    def __len__(self):
        return sum(len(t) for t in self.terms)
    
    def __iter__(self):
        return iter(self.terms)
    
    def __repr__(self):
        return self.__str__()
    
    def __getitem__(self, i):
        return self.terms[i]
    
    def to_dict(self):
        return {"terms": [t.to_dict() for t in self.terms]}
        
    @classmethod
    def from_dict(cls, terms: List[dict], *args, **kwargs):
        _terms = []
        for term in terms:
            Class = CLASSES[term.pop("_class")]
            _terms.append(Class.from_dict(**term))
        return cls(_terms, *args, **kwargs)
        
    
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