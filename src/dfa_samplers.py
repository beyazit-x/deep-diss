"""
This class is responsible for sampling LTL formulas typically from
given template(s).

@ propositions: The set of propositions to be used in the sampled
                formula at random.
"""

import math
import ring
import time
import pydot
import signal
import random
import pickle
import dill
import numpy as np
import networkx as nx
from copy import deepcopy
from pysat.solvers import Solver
from pythomata.impl.simple import SimpleNFA as NFA 
from scipy.special import softmax
from dfa import DFA, dfa2dict

class TimeOutException(Exception):
    pass

def alarm_handler(signum, frame):
    raise TimeOutException()

class DFASampler():
    def __init__(self, propositions):
        self.propositions = propositions

    def get_concept_class(self):
        raise NotImplemented

    def get_n_states(self):
        raise NotImplemented

    def get_n_accepting_states(self):
        raise NotImplemented

    def get_n_transitions(self):
        raise NotImplemented

    def get_n_conjunctions(self):
        raise NotImplemented

    def get_n_disjunctions(self):
        raise NotImplemented

    def sample(self):
        raise NotImplemented

    def get_n_alphabet(self):
        return len(self.propositions)

    def get_size_bound(self):
        return self._get_size_bound()

    def _get_size_bound(self):
        Q = self.get_n_states()
        F = self.get_n_accepting_states()
        E = self.get_n_alphabet()
        m = self.get_n_transitions()

        if F > Q /2: F = Q - F

        b_Q = math.ceil(math.log(Q, 2))
        b_E = math.ceil(math.log(E, 2))

        bin_size = math.ceil(3 + 2*b_Q + 2*b_E + (F + 1)*b_Q + m*(b_E + 2*b_Q) + 1)
        return len(str(2**bin_size - 1))
        # return bin_size

# Samples from one of the other samplers at random. The other samplers are sampled by their default args.
class SuperSampler(DFASampler):
    def __init__(self, propositions):
        super().__init__(propositions)
        self.reg_samplers = getRegisteredSamplers(self.propositions)

    def sample_ltl_formula(self):
        return random.choice(self.reg_samplers).sample_ltl_formula()

# This class samples formulas of form (or, op_1, op_2), where op_1 and 2 can be either specified as samplers_ids
# or by default they will be sampled at random via SuperSampler.
class OrSampler(DFASampler):
    def __init__(self, propositions, sampler_ids = ["SuperSampler"]*2):
        super().__init__(propositions)
        self.sampler_ids = sampler_ids

    def sample_ltl_formula(self):
        return ('or', getDFASampler(self.sampler_ids[0], self.propositions).sample_ltl_formula(),
                        getDFASampler(self.sampler_ids[1], self.propositions).sample_ltl_formula())

# This class generates random LTL formulas using the following template:
#   ('until',('not','a'),('and', 'b', ('until',('not','c'),'d')))
# where p1, p2, p3, and p4 are randomly sampled propositions
class DefaultSampler(DFASampler):
    def sample_ltl_formula(self):
        p = random.sample(self.propositions,4)
        return ('until',('not',p[0]),('and', p[1], ('until',('not',p[2]),p[3])))

# This class generates random conjunctions of Until-Tasks.
# Each until tasks has *n* levels, where each level consists
# of avoiding a proposition until reaching another proposition.
#   E.g.,
#      Level 1: ('until',('not','a'),'b')
#      Level 2: ('until',('not','a'),('and', 'b', ('until',('not','c'),'d')))
#      etc...
# The number of until-tasks, their levels, and their propositions are randomly sampled.
# This code is a generalization of the DefaultSampler---which is equivalent to UntilTaskSampler(propositions, 2, 2, 1, 1)
class UntilTaskSampler(DFASampler):
    def __init__(self, propositions, min_levels=1, max_levels=2, min_conjunctions=1 , max_conjunctions=2):
        super().__init__(propositions)
        self.levels       = (int(min_levels), int(max_levels))
        self.conjunctions = (int(min_conjunctions), int(max_conjunctions))
        assert 2*int(max_levels)*int(max_conjunctions) <= len(propositions), "The domain does not have enough propositions!"

        self.min_conjunctions = int(min_conjunctions)
        self.max_conjunctions = int(max_conjunctions)
        self.min_levels = int(min_levels)
        self.max_levels = int(max_levels)
        self.worst_case_dfa = self._sample(self.max_levels, self.max_levels, self.max_conjunctions, self.max_conjunctions)[0][0]
        self.n_alphabet = len(self.worst_case_dfa.inputs)
        self.n_states = 0
        self.n_accepting_states = 0
        self.n_transitions = 0
        for s in self.worst_case_dfa.states():
            self.n_states += 1
            self.n_transitions += sum(s != self.worst_case_dfa._transition(s, a) for a in self.worst_case_dfa.inputs)
            if self.worst_case_dfa._label(s):
                self.n_accepting_states += 1
        self.n_transitions = (self.n_transitions//2 + 1)*2

    def sample(self):
        return self._sample(self.min_levels, self.max_levels, self.min_conjunctions, self.max_conjunctions)

    def _sample(self, min_levels, max_levels, min_conjunctions, max_conjunctions):
        # Sampling a conjuntion of *n_conjs* (not p[0]) Until (p[1]) formulas of *n_levels* levels
        n_conjs = random.randint(min_conjunctions, max_conjunctions)
        p = random.sample(self.propositions,2*max_levels*n_conjs)
        ltl = None
        seqs = []
        b = 0
        for i in range(n_conjs):
            n_levels = random.randint(min_levels, max_levels)
            # Sampling an until task of *n_levels* levels
            until_task = ('until',('not',p[b]),p[b+1])
            seq = [(p[b], p[b+1])]
            b +=2
            for j in range(1,n_levels):
                until_task = ('until',('not',p[b]),('and', p[b+1], until_task))
                seq = [(p[b], p[b+1])] + seq
                b +=2
            # Adding the until task to the conjunction of formulas that the agent have to solve
            if ltl is None: ltl = until_task
            else:           ltl = ('and',until_task,ltl)
            seqs = [tuple(seq)] + seqs
        seqs = tuple(seqs)
        def delta(s, c):
            if s is not None:
                for i in range(len(s)):
                    if s[i] != () and c != s[i][0][0] and c == s[i][0][1]:
                        return s[:i] + (s[i][1:],) + s[i + 1:]
                    elif s[i] != () and c == s[i][0][0]:
                        return None
            return s
        return ((DFA(
            start=seqs,
            inputs=self.propositions,
            label=lambda s: s == tuple(tuple() for _ in range(n_conjs)),
            transition=delta,
        ),),)

    def get_n_alphabet(self):
        return self.n_alphabet

    def get_n_states(self):
        return self.n_states

    def get_n_accepting_states(self):
        return self.n_accepting_states

    def get_n_transitions(self):
        return self.n_transitions

    def get_n_conjunctions(self):
        return 1

    def get_n_disjunctions(self):
        return 1

class CompositionalUntilTaskSampler(DFASampler):
    def __init__(self, propositions, min_levels=1, max_levels=2, min_conjunctions=1 , max_conjunctions=2):
        super().__init__(propositions)
        self.levels       = (int(min_levels), int(max_levels))
        self.conjunctions = (int(min_conjunctions), int(max_conjunctions))
        assert 2*int(max_levels)*int(max_conjunctions) <= len(propositions), "The domain does not have enough propositions!"

        self.min_conjunctions = int(min_conjunctions)
        self.max_conjunctions = int(max_conjunctions)
        self.min_levels = int(min_levels)
        self.max_levels = int(max_levels)
        self.worst_case_dfa = self._sample(self.max_levels, self.max_levels, 1, 1)[0][0]
        self.n_alphabet = len(self.worst_case_dfa.inputs)
        self.n_states = 0
        self.n_accepting_states = 0
        self.n_transitions = 0
        for s in self.worst_case_dfa.states():
            self.n_states += 1
            self.n_transitions += sum(s != self.worst_case_dfa._transition(s, a) for a in self.worst_case_dfa.inputs)
            if self.worst_case_dfa._label(s):
                self.n_accepting_states += 1
        self.n_transitions = (self.n_transitions//2 + 1)*2

    def sample(self):
        return self._sample(self.min_levels, self.max_levels, self.min_conjunctions, self.max_conjunctions)

    def _sample(self, min_levels, max_levels, min_conjunctions, max_conjunctions):
        # Sampling a conjuntion of *n_conjs* (not p[0]) Until (p[1]) formulas of *n_levels* levels
        n_conjs = random.randint(min_conjunctions, max_conjunctions)
        p = random.sample(self.propositions,2*max_levels*n_conjs)
        ltl = None
        seqs = []
        b = 0
        for i in range(n_conjs):
            n_levels = random.randint(min_levels, max_levels)
            # Sampling an until task of *n_levels* levels
            until_task = ('until',('not',p[b]),p[b+1])
            seq = [(p[b], p[b+1])]
            b +=2
            for j in range(1,n_levels):
                until_task = ('until',('not',p[b]),('and', p[b+1], until_task))
                seq = [(p[b], p[b+1])] + seq
                b +=2
            # Adding the until task to the conjunction of formulas that the agent have to solve
            if ltl is None: ltl = until_task
            else:           ltl = ('and',until_task,ltl)
            seqs = [tuple(seq)] + seqs
        seqs = tuple(seqs)
        def delta(s, c):
            if s is not None:
                if s != () and c != s[0][0] and c == s[0][1]:
                    return s[1:]
                elif s != () and c == s[0][0]:
                    return None
            return s
        dfas = tuple(DFA(start=seq, inputs=self.propositions, label=lambda s: s == tuple(), transition=delta) for seq in seqs)
        return tuple((dfa,) for dfa in dfas)

    def get_n_alphabet(self):
        return self.n_alphabet

    def get_n_states(self):
        return self.n_states*self.max_conjunctions

    def get_n_accepting_states(self):
        return self.n_accepting_states*self.max_conjunctions

    def get_n_transitions(self):
        return self.n_transitions*self.max_conjunctions

    def get_n_conjunctions(self):
        return self.max_conjunctions

    def get_n_disjunctions(self):
        return 1

def _chain(xs, alphabet):
    def transition(s, c):
        if len(s) == 0:
            return s
        head, *tail = s
        if c in head:
            return tuple(tail)
        return s

    return DFA(
        start=tuple(xs),
        inputs=alphabet,
        outputs={False, True},
        label=lambda s: s == tuple(),
        transition=transition,
    )

def _accept_reach_avoid(sub_dfa, reach, avoid, alphabet):
    # assert not (reach & avoid)

    # state is (reach-avoid state, sub_dfa state)

    def transition(s, c):
        reach_avoid_state = s[0]
        sub_dfa_state = s[1]

        if reach_avoid_state == 0b10:
            if c in avoid:
                # reset the sub dfa and go back to the initial state
                next_sub_dfa_state = sub_dfa.start
                return (0b11, next_sub_dfa_state)
            next_sub_dfa_state = sub_dfa._transition(sub_dfa_state, c)
            return (reach_avoid_state, next_sub_dfa_state)
        elif reach_avoid_state == 0b01:
            return (reach_avoid_state, sub_dfa_state)
        elif c in reach:
            next_sub_dfa_state = sub_dfa._transition(sub_dfa_state, c)
            return (0b10, next_sub_dfa_state)
        elif c in avoid:
            return (0b01, sub_dfa_state)
        return (reach_avoid_state, sub_dfa_state)

        # if reach_avoid_state == 0b10:
        #     next_sub_dfa_state = sub_dfa._transition(sub_dfa_state, c)
        #     return (reach_avoid_state, next_sub_dfa_state)
        # elif reach_avoid_state == 0b01:
        #     return (reach_avoid_state, sub_dfa_state)
        # elif c in reach:
        #     next_sub_dfa_state = sub_dfa._transition(sub_dfa_state, c)
        #     return (0b10, next_sub_dfa_state)
        # elif c in avoid:
        #     return (0b01, sub_dfa_state)
        # return (reach_avoid_state, sub_dfa_state)

        # # if we've in the rejecting states, stay here
        # if reach_avoid_state == 0b01:
        #     return (reach_avoid_state, next_sub_dfa_state)
        # # if we hit an avoid prop, go to the rejecting state
        # elif c in avoid:
        #     return (0b01, next_sub_dfa_state)
        # # if we hit a reach prop, go to the accepting state
        # elif c in reach:
        #     return (0b10, next_sub_dfa_state)
        # # otherwise, only progress the sub_dfa
        # return (reach_avoid_state, next_sub_dfa_state)


    return DFA(
        start=(0b00, sub_dfa.start),
        inputs=alphabet,
        outputs={True, False},
        label=lambda s: s[0] == 0b10 and sub_dfa._label(s[1]),
        transition=transition,
    )

def avoidance(reach_avoids: list[tuple[str, str]], alphabet: set[str]) -> NFA:
    n = len(reach_avoids)
    states = {f'q{i}' for i in range(n+1)} | {'fail'}
    aps = set.union(*map(set, zip(*reach_avoids)))
    if alphabet is None:
        alphabet = aps
    assert aps <= alphabet
    transitions = {
        f'q{n}': {c: {f'q{n}'} for c in alphabet},
        'fail': {c: {'fail'} for c in alphabet},
    }
    for i, (reach, avoid) in enumerate(reach_avoids):
        assert reach != avoid
        state = f'q{i}'

        edges = {}
        for char in alphabet - {reach, avoid}:
            edges[char] = {state}
        edges[avoid] = {'fail'}
        edges[reach] = {f'q{i+1}', state}
        transitions[state] = edges

    return NFA(
        states=states,
        alphabet=alphabet,
        transition_function=transitions,
        initial_state='q0',
        accepting_states={f'q{n}'}
    )

def _reach_avoid(reach, avoid, alphabet):
    # assert not (reach & avoid)

    def transition(s, c):
        if s != 0:
            return s
        if c in reach:
            return 0b10
        elif c in avoid:
            return 0b01
        return s

    return DFA(
        start=0b00,
        inputs=alphabet,
        outputs={True, False},
        label=lambda s: s == 0b10,
        transition=transition,
    )

class UniversalSampler(DFASampler):
    def __init__(self, propositions, temp='0.5'):
        super().__init__(propositions)
        with open("dfas/enumerated_gridworld_dfas.pickle", 'rb') as f:
            enumerated_dfas = dill.load(f)

        temp = float(temp)
        augmented_props = list(propositions)
        augmented_props.remove('white')

        self.enumerated_dfas = [DFA.from_int(dfa_int, inputs=augmented_props) for dfa_int in enumerated_dfas]
        sizes = np.array([len(str(dfa_int)) for dfa_int in enumerated_dfas])
        # print(np.average(sizes))
        self.weights = softmax(-sizes * temp)

        # self.no_sink_dfas = []
        # self.sink_dfas = []
        # for dfaa in self.enumerated_dfas:
        #     _, accepting_nodes, nxg = self.dfa2nxg(dfaa)
        #     has_sink = False
        #     for node in nxg.nodes:
        #         if node in accepting_nodes:
        #             continue
        #         for edge in nxg.edges:
        #             if node == edge[0] and node != edge[1]: # If there is an outgoing edge to another node, then it is not an accepting state
        #                 break
        #         else:
        #             has_sink = True
        #             break
        #     if has_sink:
        #         self.sink_dfas.append(dfaa)
        #     else:
        #         self.no_sink_dfas.append(dfaa)

        # print("~~~~~~~~~~~~~~~~~OUTPUTING NOW~~~~~~~~~~~~~~")
        # print(len(self.no_sink_dfas))
        # print(len(self.sink_dfas))

    def sample_dfa_formula(self):
        # random_size = np.random.choice(self.sizes, p=self.weights)
        # return deepcopy(np.random.choice(self.enumerated_dfas_dict[random_size]))

        return deepcopy(np.random.choice(self.enumerated_dfas, p=self.weights))
    

class ListSampler(DFASampler):
    def __init__(self, propositions, filename):
        super().__init__(propositions)
        self.dfa_ints = []
        with open(filename, 'r') as f:
            for line in f:
                self.dfa_ints.append(int(line))

        # self.enumerated_dfas = [DFA.from_int(dfa_int, inputs=propositions) for dfa_int in dfa_ints]
        empty_dfa = DFA(start=False, label=lambda _: False, transition=lambda *_: False, inputs={'red', 'yellow','blue', 'green'})
        empty_size = len(str(empty_dfa.to_int()))
        self.max_size = max([len(str(d)) for d in self.dfa_ints])
        self.weights = [len(str(d)) - empty_size for d in self.dfa_ints]

    def sample_dfa_formula(self):
        # random_size = np.random.choice(self.sizes, p=self.weights)
        # return deepcopy(np.random.choice(self.enumerated_dfas_dict[random_size]))

        return DFA.from_int(np.random.choice(self.dfa_ints), inputs=self.propositions)

    def get_size_bound(self):
        return self.max_size

    def get_n_states(self):
        return 20

    def get_n_accepting_states(self):
        return 10

    def get_n_transitions(self):
        return 50

class LetterworldChainSinkSampler(DFASampler):
    def __init__(self, propositions, length, num_avoid):
        super().__init__(propositions)
        self.chain_length = int(length)
        self.num_avoid = int(num_avoid)

    def get_concept_class(self):
        return "LetterworldChainSinkSampler", self.chain_length, self.num_avoid

    def get_n_states(self):
        return self.chain_length + 2

    def get_n_accepting_states(self):
        return 1

    def get_n_transitions(self):
        return self.chain_length + self.num_avoid * (self.chain_length + 1)

    def get_n_conjunctions(self):
        return 1

    def get_n_disjunctions(self):
        return 1

    def sample(self):
        return ((self.sample_dfa_formula(),),)

    def sample_dfa_formula(self):

        prop_order = random.choices(self.propositions, k=self.chain_length)
        prop_avoid = []
        for prop in prop_order:
            prop_avoid.append(random.sample([n for n in self.propositions if n != prop], self.num_avoid))

        def transition(s, c):
            if s < self.chain_length and c == prop_order[s]:
                s = s + 1
            elif s < self.chain_length and c in prop_avoid[s]:
                s = self.chain_length + 1 # put in final state
            return s

        fixed_dfa = DFA(start=0,
                            inputs=self.propositions,
                            outputs={False, True},
                            label=lambda s: s == self.chain_length,
                            transition=transition)

        # print(len(fixed_dfa.states())) # This returns 7 for chain length 5

        return fixed_dfa


class LetterworldChainSampler(DFASampler):
    def __init__(self, propositions, length):
        super().__init__(propositions)
        self.chain_length = int(length)

    def get_concept_class(self):
        return "LetterworldChainSampler", self.chain_length

    def get_n_states(self):
        return self.chain_length + 1

    def get_n_accepting_states(self):
        return 1

    def get_n_transitions(self):
        return self.chain_length

    def get_n_conjunctions(self):
        return 1

    def get_n_disjunctions(self):
        return 1

    def sample(self):
        return ((self.sample_dfa_formula(),),)

    def sample_dfa_formula(self):

        prop_order = random.choices(self.propositions, k=self.chain_length)

        def transition(s, c):
            if s < self.chain_length and c == prop_order[s]:
                s = s + 1
            return s

        fixed_dfa = DFA(start=0,
                            inputs=self.propositions,
                            outputs={False, True},
                            label=lambda s: s == self.chain_length,
                            transition=transition)

        # print(len(fixed_dfa.states())) # This returns 6 for chain length 5

        return fixed_dfa

class FixedLetterworldChainSampler(DFASampler):
    def __init__(self, propositions):
        super().__init__(propositions)
        self.chain_length = 10

    def transition(self, s, c):
        if c == self.propositions[s] and s < self.chain_length:
            s = s + 1
        return s


    def sample_dfa_formula(self):
        fixed_dfa = DFA(start=0,
                            inputs=self.propositions,
                            outputs={False, True},
                            label=lambda s: s == self.chain_length,
                            transition=self.transition)

        return fixed_dfa

class FixedLetterworldSampler(DFASampler):
    def __init__(self, propositions):
        super().__init__(propositions)

    def transition(self, s, c):
        red, blue, yellow, green = self.propositions[0:4]
        if s == 0:
            if c == red:
                s = 1 # fail
            elif c == blue:
                s = 2 # got wet
            elif c == yellow:
                s = 3 # success
        elif s == 2:
            if c == red or c == yellow:
                s = 1 # fail
            elif c == green:
                s = 0 # back to start
        elif s == 3:
            if c == blue:
                s = 2 # got wet
            if c == red:
                s = 1 # fail

        return s


    def sample_dfa_formula(self):
        fixed_dfa = DFA(start=0,
                            inputs=self.propositions,
                            outputs={False, True},
                            label=lambda s: s == 3,
                            transition=self.transition)

        return fixed_dfa

class FixedGridworldSampler(DFASampler):
    def __init__(self, propositions):
        super().__init__(propositions)

    def transition(self, s, c):
        if s == 0:
            if c == 'red':
                s = 1 # fail
            elif c == 'blue':
                s = 2 # got wet
            elif c == 'yellow':
                s = 3 # success
        elif s == 2:
            if c == 'red' or c == 'yellow':
                s = 1 # fail
            elif c == 'green':
                s = 0 # back to start
        elif s == 3:
            if c == 'blue':
                s = 2 # got wet
            if c == 'red':
                s = 1 # fail

        return s


    def sample_dfa_formula(self):
        fixed_dfa = DFA(start=0,
                            inputs={'blue', 'green', 'red', 'yellow', 'white'},
                            outputs={False, True},
                            label=lambda s: s == 3,
                            transition=self.transition)

        return fixed_dfa

class EventuallySampler(DFASampler):
    def __init__(self, propositions, min_levels = 1, max_levels=4, min_conjunctions=1, max_conjunctions=3):
        super().__init__(propositions)
        assert(len(propositions) >= 3)
        self.min_conjunctions = int(min_conjunctions)
        self.max_conjunctions = int(max_conjunctions)
        self.min_levels = int(min_levels)
        self.max_levels = int(max_levels)
        self.worst_case_dfa = self._sample(self.max_levels, self.max_levels, self.max_conjunctions, self.max_conjunctions, 1.0)[0][0]
        self.n_alphabet = len(self.worst_case_dfa.inputs)
        self.n_states = 0
        self.n_accepting_states = 0
        self.n_transitions = 0
        for s in self.worst_case_dfa.states():
            self.n_states += 1
            self.n_transitions += sum(s != self.worst_case_dfa._transition(s, a) for a in self.worst_case_dfa.inputs)
            if self.worst_case_dfa._label(s):
                self.n_accepting_states += 1
        self.n_transitions = (self.n_transitions//2 + 1)*2

    def get_n_alphabet(self):
        return self.n_alphabet

    def get_n_states(self):
        return self.n_states

    def get_n_accepting_states(self):
        return self.n_accepting_states

    def get_n_transitions(self):
        return self.n_transitions

    def get_n_conjunctions(self):
        return 1

    def get_n_disjunctions(self):
        return 1

    def get_size_bound(self):
        return self._get_size_bound()

    def _get_size_bound(self):
        Q = self.get_n_states()
        F = self.get_n_accepting_states()
        E = self.get_n_alphabet()
        m = self.get_n_transitions()

        if F > Q /2: F = Q - F

        b_Q = math.ceil(math.log(Q, 2))
        b_E = math.ceil(math.log(E, 2))

        bin_size = math.ceil(3 + 2*b_Q + 2*b_E + (F + 1)*b_Q + m*(b_E + 2*b_Q) + 1)
        return len(str(2**bin_size - 1))
        # return bin_size

    def sample(self):
        return self._sample(self.min_levels, self.max_levels, self.min_conjunctions, self.max_conjunctions, 0.25)

    def _sample(self, min_levels, max_levels, min_conjunctions, max_conjunctions, p=0.25):
        conjs = random.randint(min_conjunctions, max_conjunctions)
        seqs = tuple(self._sample_sequence(min_levels, max_levels, p) for _ in range(conjs))
        def delta(s, c):
            for i in range(len(s)):
                if s[i] != () and c in s[i][0]:
                    return s[:i] + (s[i][1:],) + s[i + 1:]
            return s
        return ((DFA(
            start=seqs,
            inputs=self.propositions,
            label=lambda s: s == tuple(tuple() for _ in range(conjs)),
            transition=delta,
        ),),)

    def sample_sequence(self):
        return self._sample_sequence(self.min_levels, self.max_levels, 0.25)

    def _sample_sequence(self, min_levels, max_levels, p=0.25):
        length = random.randint(min_levels, max_levels)
        seq = []

        last = []
        while len(seq) < length:
            # Randomly replace some propositions with a disjunction to make more complex formulas
            population = [s for s in self.propositions if s not in last]

            if random.random() < p:
                c = random.sample(population, 2)
            else:
                c = random.sample(population, 1)

            seq.append(tuple(c))
            last = c

        return tuple(seq)

class CompositionalEventuallySampler(EventuallySampler):
    def __init__(self, propositions, min_levels = 1, max_levels=4, min_conjunctions=1, max_conjunctions=3):
        super().__init__(propositions, min_levels, max_levels, 1, 1)
        assert(len(propositions) >= 3)
        self.min_conjunctions = int(min_conjunctions)
        self.max_conjunctions = int(max_conjunctions)
        self.min_levels = int(min_levels)
        self.max_levels = int(max_levels)

    def sample(self):
        conjs = random.randint(self.min_conjunctions, self.max_conjunctions)
        seqs = tuple(self.sample_sequence() for _ in range(conjs))
        dfas = tuple(DFA(start=seq, inputs=self.propositions, label=lambda s: s == tuple(), transition=lambda s, c: s[1:] if s != () and c in s[0] else s) for seq in seqs)
        return tuple((dfa,) for dfa in dfas)

    def get_size_bound(self):
        size_per_dfa = self._get_size_bound()
        return size_per_dfa * self.max_conjunctions

    def get_n_conjunctions(self):
        return self.max_conjunctions

    def get_n_disjunctions(self):
        return 1

class AdversarialEnvSampler(DFASampler):
    def sample(self):
        p = random.randint(0,1)
        if p == 0:
            def delta(s, c):
                if s == 0 and c == 'a':
                    return 1
                elif s == 1 and c == 'b':
                    return 2
                return s
            return ((DFA(
                start=0,
                inputs=self.propositions,
                label=lambda s: s == 2,
                transition=delta,
            ),),)
        else:
            def delta(s, c):
                if s == 0 and c == 'a':
                    return 1
                elif s == 1 and c == 'c':
                    return 2
                return s
            return ((DFA(
                start=0,
                inputs=self.propositions,
                label=lambda s: s == 2,
                transition=delta,
            ),),)

    def get_n_alphabet(self):
        return len(self.propositions)

    def get_n_states(self):
        return 3

    def get_n_accepting_states(self):
        return 1

    def get_n_transitions(self):
        return 2

    def get_n_conjunctions(self):
        return 1

    def get_n_disjunctions(self):
        return 1

def getRegisteredSamplers(propositions):
    return [SequenceSampler(propositions),
            UntilTaskSampler(propositions),
            DefaultSampler(propositions),
            EventuallySampler(propositions),
            CompositionalEventuallySampler(propositions)]

# The DFASampler factory method that instantiates the proper sampler
# based on the @sampler_id.
def getDFASampler(sampler_id, propositions):
    tokens = ["Default"]
    if (sampler_id != None):
        tokens = sampler_id.split("_")

    # Don't change the order of ifs here otherwise the OR sampler will fail
    if (tokens[0] == "OrSampler"):
        return OrSampler(propositions)
    elif ("_OR_" in sampler_id): # e.g., Sequence_2_4_OR_UntilTask_3_3_1_1
        sampler_ids = sampler_id.split("_OR_")
        return OrSampler(propositions, sampler_ids)
    elif (tokens[0] == "Sequence"):
        return SequenceSampler(propositions, tokens[1], tokens[2])
    elif (tokens[0] == "Until"):
        return UntilTaskSampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4])
    elif (tokens[0] == "CompositionalUntil"):
        return CompositionalUntilTaskSampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4])
    elif (tokens[0] == "SuperSampler"):
        return SuperSampler(propositions)
    elif (tokens[0] == "Adversarial"):
        return AdversarialEnvSampler(propositions)
    elif (tokens[0] == "Eventually"):
        return EventuallySampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4])
    elif (tokens[0] == "CompositionalEventually"):
        return CompositionalEventuallySampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4])
    elif (tokens[0] == "Universal"):
        return UniversalSampler(propositions, tokens[1])
    elif (tokens[0] == "List"):
        return ListSampler(propositions, tokens[1])
    elif (tokens[0] == "FixedGridworld"):
        return FixedGridworldSampler(propositions)
    elif (tokens[0] == "FixedLetterworld"):
        return FixedLetterworldSampler(propositions)
    elif (tokens[0] == "FixedLetterworldChain"):
        return FixedLetterworldChainSampler(propositions)
    elif (tokens[0] == "LetterworldChain"):
        return LetterworldChainSampler(propositions, tokens[1])
    elif (tokens[0] == "LetterworldChainSink"):
        return LetterworldChainSinkSampler(propositions, tokens[1], tokens[2])
    else: # "Default"
        return DefaultSampler(propositions)

def draw(G, path):
    from networkx.drawing.nx_agraph import to_agraph
    A = to_agraph(G) 
    A.layout('dot')                                                                 
    A.draw(path)

if __name__ == '__main__':
    import sys
    props = ["yellow", "green", "blue", "red"]
    sampler_id = sys.argv[1]
    sampler = getDFASampler(sampler_id, props)
    draw_path = "sample_dfa.png"
    dfa = sampler.sample_dfa_formula()
    print(dfa)
    # draw(dfa, draw_path)

