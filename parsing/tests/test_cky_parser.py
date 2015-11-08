# https://docs.python.org/3/library/unittest.html
from unittest import TestCase
from math import log2

from nltk.tree import Tree
from nltk.grammar import PCFG

from parsing.cky_parser import CKYParser


class TestCKYParser(TestCase):

    def test_parse(self):
        grammar = PCFG.fromstring(
            """
                S -> NP VP              [1.0]
                NP -> Det Noun          [0.6]
                NP -> Noun Adj          [0.4]
                VP -> Verb NP           [1.0]
                Det -> 'el'             [1.0]
                Noun -> 'gato'          [0.9]
                Noun -> 'pescado'       [0.1]
                Verb -> 'come'          [1.0]
                Adj -> 'crudo'          [1.0]
            """)

        parser = CKYParser(grammar)

        lp, t = parser.parse('el gato come pescado crudo'.split())

        # check chart
        pi = {
            (1, 1): {'Det': log2(1.0)},
            (2, 2): {'Noun': log2(0.9)},
            (3, 3): {'Verb': log2(1.0)},
            (4, 4): {'Noun': log2(0.1)},
            (5, 5): {'Adj': log2(1.0)},

            (1, 2): {'NP': log2(0.6 * 1.0 * 0.9)},
            (2, 3): {},
            (3, 4): {},
            (4, 5): {'NP': log2(0.4 * 0.1 * 1.0)},

            (1, 3): {},
            (2, 4): {},
            (3, 5): {'VP': log2(1.0) + log2(1.0) + log2(0.4 * 0.1 * 1.0)},

            (1, 4): {},
            (2, 5): {},

            (1, 5): {'S':
                     log2(1.0) +  # rule S -> NP VP
                     log2(0.6 * 1.0 * 0.9) +  # left part
                     log2(1.0) + log2(1.0) + log2(0.4 * 0.1 * 1.0)},  # right part
        }
        self.assertEqualPi(parser._pi, pi)

        # check partial results
        # I had comment empty dicts, this checks are no needed in my cky parser.
        bp = {
            (1, 1): {'Det': Tree.fromstring("(Det el)")},
            (2, 2): {'Noun': Tree.fromstring("(Noun gato)")},
            (3, 3): {'Verb': Tree.fromstring("(Verb come)")},
            (4, 4): {'Noun': Tree.fromstring("(Noun pescado)")},
            (5, 5): {'Adj': Tree.fromstring("(Adj crudo)")},

            (1, 2): {'NP': Tree.fromstring("(NP (Det el) (Noun gato))")},
            # (2, 3): {},
            # (3, 4): {},
            (4, 5): {'NP': Tree.fromstring("(NP (Noun pescado) (Adj crudo))")},

            # (1, 3): {},
            # (2, 4): {},
            (3, 5): {'VP': Tree.fromstring(
                "(VP (Verb come) (NP (Noun pescado) (Adj crudo)))")},

            # (1, 4): {},
            # (2, 5): {},

            (1, 5): {'S': Tree.fromstring(
                """(S
                    (NP (Det el) (Noun gato))
                    (VP (Verb come) (NP (Noun pescado) (Adj crudo)))
                   )
                """)},
        }
        self.assertEqual(parser._bp, bp)

        # check tree
        t2 = Tree.fromstring(
            """
                (S
                    (NP (Det el) (Noun gato))
                    (VP (Verb come) (NP (Noun pescado) (Adj crudo)))
                )
            """)
        self.assertEqual(t, t2)

        # check log probability
        lp2 = log2(1.0 * 0.6 * 1.0 * 0.9 * 1.0 * 1.0 * 0.4 * 0.1 * 1.0)
        self.assertAlmostEqual(lp, lp2)

    def assertEqualPi(self, pi1, pi2):
        self.assertEqual(set(pi1.keys()), set(pi2.keys()))

        for k in pi1.keys():
            d1, d2 = pi1[k], pi2[k]
            self.assertEqual(d1.keys(), d2.keys(), k)
            for k2 in d1.keys():
                prob1 = d1[k2]
                prob2 = d2[k2]
                self.assertAlmostEqual(prob1, prob2)

    def test_ambiguity(self):
        grammar = PCFG.fromstring(
            """
                S -> NP VP [1.0]
                VP -> Vt NP [0.8]
                VP -> VP PP [0.2]
                NP -> DT NN [0.2]
                NP -> NP PP [0.8]
                PP -> IN NP [1.0]
                Vi -> 'sleeps' [1.0]
                Vt -> 'saw' [1.0]
                NN -> 'man' [0.1]
                NN -> 'woman' [0.1]
                NN -> 'telescope' [0.3]
                NN -> 'dog' [0.5]
                DT -> 'the' [1.0]
                IN -> 'with' [0.6]
                IN -> 'in' [0.4]
            """)

        """
        I'm looking my parser to say that the man saw a dog that was using a telescope,
        think in ... Snoopy
        """
        parser = CKYParser(grammar)

        lp, t = parser.parse('the man saw the dog with the telescope'.split())

        # check chart
        pi = {
            (1, 1): {'DT': log2(1.0)},
            (2, 2): {'NN': log2(0.1)},
            (3, 3): {'Vt': log2(1.0)},
            (4, 4): {'DT': log2(1.0)},
            (5, 5): {'NN': log2(0.5)},
            (6, 6): {'IN': log2(0.6)},
            (7, 7): {'DT': log2(1.0)},
            (8, 8): {'NN': log2(0.3)},

            (1, 2): {'NP': log2(0.2 * 1.0 * 0.1)},
            (2, 3): {},
            (3, 4): {},
            (4, 5): {'NP': log2(0.2 * 1.0 * 0.5)},
            (5, 6): {},
            (6, 7): {},
            (7, 8): {'NP': log2(0.2 * 1.0 * 0.3)},

            (1, 3): {},
            (2, 4): {},
            (3, 5): {'VP':
                     log2(0.8) +  # Rule VP -> Vt NP
                     log2(1.0) + # Left Part
                     log2(0.2 * 1.0 * 0.5) # Right Part
                     },
            (4, 6): {},
            (5, 7): {},
            (6, 8): {'PP':
                     log2(1.0) +  # Rule PP -> IN NP
                     log2(0.6) +  # Left part
                     log2(0.2 * 1.0 * 0.3)  # Right part
                     },

            (1, 4): {},
            (2, 5): {},
            (3, 6): {},
            (4, 7): {},
            (5, 8): {},

            (1, 5): {'S':
                     log2(1.0) + # Rule S -> NP VP
                     log2(0.2 * 1.0 * 0.1) + # Left Part
                     log2(0.8 * 1.0) + log2(0.2 * 1.0 * 0.5) # Right Part
                     },
            (2, 6): {},
            (3, 7): {},
            (4, 8): {'NP':
                     log2(0.8) +  # Rule NP -> NP PP}
                     log2(0.2 * 1.0 * 0.5) +  # Left part
                     (log2(1.0) + log2(0.6) + log2(0.2 * 1.0 * 0.3))  # Right part
                     },

            (1, 6): {},
            (2, 7): {},
            (3, 8): {'VP':
                     log2(0.8) +  # Rule VP -> Vt NP
                     log2(1.0) +  # Left Part
                     (log2(0.8) + log2(0.2 * 1.0 * 0.5) +
                      (log2(1.0) + log2(0.6) + log2(0.2 * 1.0 * 0.3)))  # Right part
                     },

            (1, 7): {},
            (2, 8): {},

            (1, 8): {'S':
                     log2(1.0) +  # Rule S -> NP VP
                     log2(0.2 * 1.0 * 0.1) +  # Left Part
                     (log2(0.8) + log2(1.0) +
                      (log2(0.8) + log2(0.2 * 1.0 * 0.5) +
                       (log2(1.0) + log2(0.6) + log2(0.2 * 1.0 * 0.3))))  # Right part
                     }
        }
        self.assertEqualPi(parser._pi, pi)

        # check partial results
        # I had comment empty dicts, those checks are no needed in my cky parser.
        bp = {
              (1, 1): {'DT': Tree('DT', ['the'])},
              (1, 2): {'NP': Tree('NP', [Tree('DT', ['the']), Tree('NN', ['man'])])},
              (1, 5): {'S': Tree('S',
                                 [
                                  Tree('NP',[Tree('DT', ['the']), Tree('NN', ['man'])]),
                                  Tree('VP',
                                       [
                                        Tree('Vt', ['saw']),
                                        Tree('NP', [Tree('DT', ['the']), Tree('NN', ['dog'])])
                                       ]
                                      )
                                 ]
                                )
                       },
              (1, 8): {'S': Tree('S',
                                 [
                                  Tree('NP', [Tree('DT', ['the']), Tree('NN', ['man'])]),
                                  Tree('VP',
                                       [
                                        Tree('Vt', ['saw']),
                                        Tree('NP',
                                             [
                                              Tree('NP', [Tree('DT', ['the']), Tree('NN', ['dog'])]),
                                              Tree('PP', [Tree('IN', ['with']), Tree('NP', [Tree('DT', ['the']), Tree('NN', ['telescope'])])])
                                             ]
                                            )
                                       ]
                                      )
                                  ]
                                )
                      },
              (2, 2): {'NN': Tree('NN', ['man'])},
              (3, 3): {'Vt': Tree('Vt', ['saw'])},
              (3, 5): {'VP': Tree('VP',
                                  [
                                   Tree('Vt', ['saw']),
                                   Tree('NP', [Tree('DT', ['the']), Tree('NN', ['dog'])])
                                  ]
                                 )
                      },
              (3, 8): {'VP': Tree('VP',
                                  [
                                   Tree('Vt', ['saw']),
                                   Tree('NP',
                                        [
                                         Tree('NP', [Tree('DT', ['the']), Tree('NN', ['dog'])]),
                                         Tree('PP',
                                              [
                                               Tree('IN', ['with']),
                                               Tree('NP', [Tree('DT', ['the']), Tree('NN', ['telescope'])])
                                              ]
                                             )
                                         ]
                                        )
                                   ]
                                  )
                      },
              (4, 4): {'DT': Tree('DT', ['the'])},
              (4, 5): {'NP': Tree('NP', [Tree('DT', ['the']), Tree('NN', ['dog'])])},
              (4, 8): {'NP': Tree('NP',
                                  [
                                   Tree('NP', [Tree('DT', ['the']), Tree('NN', ['dog'])]),
                                   Tree('PP',
                                        [
                                         Tree('IN', ['with']),
                                         Tree('NP', [Tree('DT', ['the']), Tree('NN', ['telescope'])])
                                        ]
                                       )
                                  ]
                                 )
                       },
              (5, 5): {'NN': Tree('NN', ['dog'])},
              (6, 6): {'IN': Tree('IN', ['with'])},
              (6, 8): {'PP': Tree('PP',
                                  [
                                   Tree('IN', ['with']),
                                   Tree('NP', [Tree('DT', ['the']), Tree('NN', ['telescope'])])
                                  ]
                                 )
                      },
              (7, 7): {'DT': Tree('DT', ['the'])},
              (7, 8): {'NP': Tree('NP', [Tree('DT', ['the']), Tree('NN', ['telescope'])])},
              (8, 8): {'NN': Tree('NN', ['telescope'])}
              }

        self.assertEqual(parser._bp, bp)


        # check tree
        t2 = Tree('S',
                  [
                   Tree('NP', [Tree('DT', ['the']), Tree('NN', ['man'])]),
                   Tree('VP',
                        [
                         Tree('Vt', ['saw']),
                         Tree('NP',
                              [
                               Tree('NP', [Tree('DT', ['the']), Tree('NN', ['dog'])]),
                               Tree('PP',
                                    [
                                     Tree('IN', ['with']),
                                     Tree('NP',
                                          [
                                           Tree('DT', ['the']),
                                           Tree('NN', ['telescope'])
                                          ]
                                         )
                                    ]
                                   )
                              ]
                             )
                        ]
                       )
                  ]
                 )

        self.assertEqual(t, t2)

        # check log probability
        lp2 = log2(1.0 * 0.2 * 1.0 * 0.1 * 0.8 * 1.0 * 0.8 * 0.2 * 1.0 * 0.5 * 1.0 * 0.6 * 0.2 * 1.0 * 0.3)
        self.assertAlmostEqual(lp, lp2)
