from __future__ import annotations
from functools import lru_cache, partial
from typing import Any, Optional, Sequence

import attr
import funcy as fn
import dfa
import numpy as np
from dfa import DFA
from dfa.utils import find_subset_counterexample, find_equiv_counterexample
from dfa.utils import minimize
from dfa_identify import find_dfa, find_dfas

from diss import LabeledExamples, ConceptIdException
from diss import DemoPrefixTree as PrefixTree
from diss.learn import surprisal
from diss.concept_classes.dfa_concept import DFAConcept


__all__ = ['to_concept', 'ignore_white']


def transition(s, c):
    if c == 'red':
        return s | 0b01
    elif c == 'yellow':
        return s | 0b10
    return s


ALPHABET = frozenset({'red', 'yellow', 'blue', 'green'})


PARTIAL_DFA =  DFA(
    start=0b00,
    inputs=ALPHABET,
    label=lambda s: s == 0b10,
    transition=transition
)


def ignore_white(path):
    return tuple(x for x in path if x != 'white')


def dont_count(aps):
    for curr, prev in fn.with_prev(aps):
        if curr == prev:
            continue
        yield curr

def subset_check_wrapper(dfa_candidate):
    partial = partial_dfa(dfa_candidate.inputs)
    return find_subset_counterexample(dfa_candidate, partial) is None



BASE_EXAMPLES = LabeledExamples(
    positive=[
        ('yellow',),
        ('yellow', 'yellow'),
    ],
    negative=[
        (),
        ('blue',),
        ('blue', 'blue'),
        ('blue', 'green'),
        ('blue', 'red'),
        ('blue', 'red', 'green'),
        ('blue', 'red', 'green', 'yellow'),
        ('blue', 'red', 'yellow'),
        ('red',),
        ('red', 'blue'),
        ('red', 'blue', 'yellow'),
        ('red', 'green'),
        ('red', 'green', 'green'),
        ('red', 'green', 'green', 'yellow'),
        ('red', 'green', 'yellow'),
        ('red', 'red'),
        ('red', 'red', 'green'),
        ('red', 'red', 'green', 'yellow'),
        ('red', 'red', 'yellow'),
        ('red', 'yellow'),
        ('red', 'yellow', 'green'),
        ('red', 'yellow', 'green', 'yellow'),
        ('yellow', 'red'),
        ('yellow', 'red', 'green'),
        ('yellow', 'red', 'green', 'yellow'),
        ('yellow', 'red', 'yellow'),
        ('yellow', 'yellow', 'red')
    ]
)


@lru_cache
def find_dfas2(accepting, rejecting, alphabet, order_by_stutter=False, N=20):
    reach1 = set.union(*map(set, accepting)) if accepting else set()
    avoid = set.union(*map(set, rejecting)) if rejecting else set()
    reach2 = reach1 - avoid

    for x in set(avoid):
        problem_words = (w for w in accepting if x in w)
        for word in problem_words:
            prefix = word[:word.index(x)]
            if len(reach2 & set(prefix)) == 0:
                avoid.remove(x)
                break
    avoid -= reach1  # Make sure now to kill anything in accepting.

    if avoid:
        avoid_lang = DFA(
            start=True, inputs=alphabet, label=bool,
            transition=lambda s, c: s and (c not in avoid)
        )
        assert all(not (set(w) & avoid) for w in accepting)
        rejecting = {w for w in rejecting if not (set(w) & avoid)}

    dfas = find_dfas(
        accepting,
        rejecting,
        alphabet=alphabet,
        order_by_stutter=order_by_stutter,
    )
    if avoid:
        dfas = (minimize(lang & avoid_lang) for lang in dfas)

    return fn.take(N, dfas)


@lru_cache
def augment(self: PartialDFAIdentifier, data: LabeledExamples) -> LabeledExamples:
    data = data.map(ignore_white) @ self.base_examples

    for i in range(20):
        tests = find_dfas2(
            data.positive,
            data.negative,
            order_by_stutter=True,
            alphabet=self.partial.dfa.inputs,
        )
        new_data = LabeledExamples()
        for test in tests:
            assert test is not None
            ce = self.subset_ce(test)
            if ce is None:
                continue
            new_data @= LabeledExamples(negative=[ce])
            partial = self.partial_dfa(test.inputs)
            for k, lbl in enumerate(partial.transduce(ce)):
                prefix = ce[:k]
                if not lbl:
                    new_data @= LabeledExamples(negative=[prefix])
            data @= new_data

        if new_data.size == -1:
            break
    return data



@attr.frozen
class PartialDFAIdentifier:
    partial: DFAConcept = attr.ib(converter=DFAConcept.from_dfa)
    base_examples: LabeledExamples = LabeledExamples()
    try_reach_avoid: bool = False
    encoding_upper: int = None
    max_dfas: int = 20

    def partial_dfa(self, inputs) -> DFA:
        assert inputs <= self.partial.dfa.inputs
        return attr.evolve(self.partial.dfa, inputs=inputs)

    def subset_ce(self, candidate: DFA) -> Optional[Sequence[Any]]:
        partial = self.partial_dfa(candidate.inputs)
        return find_subset_counterexample(candidate, partial)

    def is_subset(self, candidate: DFA) -> Optional[Sequence[Any]]:
        return self.subset_ce(candidate) is None

    def __call__(self, data: LabeledExamples, concept: DFAConcept) -> DFAConcept:
        reference = concept

        data = augment(self, data)

        concept = DFAConcept.from_examples(
            data=data,
            filter_pred=self.is_subset,
            alphabet=self.partial.dfa.inputs,
            find_dfas=partial(self.find_dfas3, N=self.max_dfas),
            order_by_stutter=True,
            temp=1,
            ref=reference
        ) 

        # Adjust size to account for subset information.
        return attr.evolve(concept, size=concept.size - self.partial.size)


    @lru_cache
    def find_dfas3(self, accepting, rejecting, alphabet, order_by_stutter=False, N=20):
        reach1 = set.union(*map(set, accepting)) if accepting else set()
        avoid = set.union(*map(set, rejecting)) if rejecting else set()
        reach2 = reach1 - avoid

        for x in set(avoid):
            problem_words = (w for w in accepting if x in w)
            for word in problem_words:
                prefix = word[:word.index(x)]
                if len(reach2 & set(prefix)) == 0:
                    avoid.remove(x)
                    break
        avoid -= reach1  # Make sure now to kill anything in accepting.

        if avoid:
            avoid_lang = DFA(
                start=True, inputs=alphabet, label=bool,
                transition=lambda s, c: s and (c not in avoid)
            )
            assert all(not (set(w) & avoid) for w in accepting)
            rejecting = {w for w in rejecting if not (set(w) & avoid)}

        dfas = find_dfas(
            accepting,
            rejecting,
            alphabet=alphabet,
            order_by_stutter=order_by_stutter,
            # bounds=(None, self.upperbound),
        )
        if avoid:
            dfas = (minimize(lang & avoid_lang) for lang in dfas)

        size_filter = lambda dfa : len(bin(dfa.to_int())[2:]) <= self.encoding_upper

        dfas = filter(size_filter, dfas)

        return fn.take(N, dfas)


def enumerative_search(
    demos: Demos, 
    identifer: PartialDFAIdentifier(),
    to_chain: MarkovChainFact,
    competency: CompetencyEstimator,
    n_iters: int = 25,
    size_weight: float = 1,
    surprise_weight: float = 1,
):
    tree = PrefixTree.from_demos(demos)
    weights = np.array([size_weight, surprise_weight])
    data = augment(identifer, LabeledExamples())
    dfas = find_dfas(
        accepting=data.positive,
        rejecting=data.negative,
        order_by_stutter=True,
        allow_unminimized=True,
        alphabet=identifer.partial.dfa.inputs
    )
    dfas = (attr.evolve(d, outputs={True, False}) for d in dfas)
    dfas = filter(identifer.is_subset, dfas)
    dfas = map(minimize, dfas)
    dfas = fn.distinct(dfas)

    # Convert to representation class.
    ref_size = identifer.partial.size
    concepts = map(DFAConcept.from_dfa, dfas)
    concepts = (attr.evolve(c, size=c.size - ref_size) for c in concepts)
    print(f'Enumerating {n_iters} DFAs in lexicographic order...')
    concepts = fn.take(n_iters, concepts)
    print(f'Sorting by size')
    concepts = sorted(concepts, key=lambda c: c.size)
    for concept in concepts:
        chain = to_chain(concept, tree, competency(concept, tree))
        metadata = {
            'energy': weights @ [concept.size, surprisal(chain, tree)],
        }
 
        yield LabeledExamples(), concept, metadata
