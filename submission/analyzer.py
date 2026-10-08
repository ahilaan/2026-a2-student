"""Standard-library analyzer; Porter rules translated from the student's A1.

Source: own A1 submission/cpp/porter_stemmer.cpp and analyzer.cpp.
No external stemming or IR library is required.
"""
import re
from functools import lru_cache

STOPWORDS = frozenset('''a about above after again against all am an and any are as at
be because been before being below between both but by can cannot could did do does
doing down during each few for from further had has have having he her here hers
herself him himself his how i if in into is it its itself just me more most my myself
no nor not now of off on once only or other our ours ourselves out over own s same
she should so some such t than that the their theirs them themselves then there these
they this those through to too under until up very was we were what when where which
while who whom why will with would you your yours yourself yourselves don ll re ve d
m o y ain aren couldn didn doesn hadn hasn haven isn ma mightn mustn needn shan
shouldn wasn weren won wouldn'''.split())
_TOKENS = re.compile(r'[a-z0-9]+')


def _cons(word, i):
    c = word[i]
    if c in 'aeiou':
        return False
    if c == 'y':
        return i == 0 or not _cons(word, i - 1)
    return True


def _measure(word):
    return sum(not _cons(word, i - 1) and _cons(word, i) for i in range(1, len(word)))


def _vowel(word):
    return any(not _cons(word, i) for i in range(len(word)))


def _double(word):
    return len(word) >= 2 and word[-1] == word[-2] and _cons(word, len(word) - 1)


def _cvc(word):
    return (len(word) >= 3 and _cons(word, len(word) - 1) and
            not _cons(word, len(word) - 2) and _cons(word, len(word) - 3)
            and word[-1] not in 'wxy')


@lru_cache(maxsize=200000)
def porter_stem(word):
    if len(word) <= 2 or not word.isalpha():
        return word
    for suffix, replacement in [('sses', 'ss'), ('ies', 'i'), ('ss', 'ss'), ('s', '')]:
        if word.endswith(suffix):
            word = word[:-len(suffix)] + replacement
            break
    if word.endswith('eed'):
        if _measure(word[:-3]) > 0:
            word = word[:-1]
    else:
        for suffix in ('ed', 'ing'):
            if word.endswith(suffix) and _vowel(word[:-len(suffix)]):
                word = word[:-len(suffix)]
                if word.endswith(('at', 'bl', 'iz')):
                    word += 'e'
                elif _double(word) and word[-1] not in 'lsz':
                    word = word[:-1]
                elif _measure(word) == 1 and _cvc(word):
                    word += 'e'
                break
    if word.endswith('y') and _vowel(word[:-1]):
        word = word[:-1] + 'i'
    steps = [
        [('ational','ate'),('tional','tion'),('enci','ence'),('anci','ance'),
         ('izer','ize'),('abli','able'),('alli','al'),('entli','ent'),('eli','e'),
         ('ousli','ous'),('ization','ize'),('ation','ate'),('ator','ate'),
         ('alism','al'),('iveness','ive'),('fulness','ful'),('ousness','ous'),
         ('aliti','al'),('iviti','ive'),('biliti','ble')],
        [('icate','ic'),('ative',''),('alize','al'),('iciti','ic'),('ical','ic'),('ful',''),('ness','')],
        [(suffix, '') for suffix in ('al','ance','ence','er','ic','able','ible','ant',
                                     'ement','ment','ent','ion','ou','ism','ate','iti','ous','ive','ize')]
    ]
    for step, rules in enumerate(steps):
        for suffix, replacement in rules:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if suffix == 'ion' and (not stem or stem[-1] not in 'st'):
                    break
                if _measure(stem) > (1 if step == 2 else 0):
                    word = stem + replacement
                break
    if word.endswith('e'):
        stem = word[:-1]
        measure = _measure(stem)
        if measure > 1 or (measure == 1 and not _cvc(stem)):
            word = stem
    if _measure(word) > 1 and _double(word) and word.endswith('l'):
        word = word[:-1]
    return word


def analyze(text, mode='raw'):
    if mode not in ('raw', 'stop', 'porter', 'stop_porter'):
        raise ValueError('Unknown analyzer: ' + mode)
    text = text.lower()
    tokens = _TOKENS.findall(text)
    if mode in ('stop', 'stop_porter'):
        tokens = [token for token in tokens if token not in STOPWORDS]
    if mode in ('porter', 'stop_porter'):
        tokens = [porter_stem(token) for token in tokens]
    return tokens
