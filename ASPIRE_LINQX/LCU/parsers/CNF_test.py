import unittest
from CNF import DisjunctionStatement, ConjunctionStatement
import operator

#integers
a = 5
b = 8
c = 2
d = 5
e = 'hi'
f = 'bye'
g = 'hi'

class TestDisjunction(unittest.TestCase):
    def test_gt1(self):
        statement = DisjunctionStatement(a,b,operator.gt)
        print(str(statement))
        self.assertEqual(statement.eval(), False)
    
    def test_gt2(self):
        print("")
        statement = DisjunctionStatement(b,a,operator.gt)
        print(str(statement))
        self.assertEqual(statement.eval(),True)

        
class TestConjunction(unittest.TestCase):
    def test_conjunction1(self):
        statement1 = DisjunctionStatement(a,b,operator.gt)
        statement2 = DisjunctionStatement(b,a,operator.gt)
        statement3 = ConjunctionStatement([statement1,statement2])
        print(str(statement3))
        self.assertEqual(statement3.eval(),True)


class TestCNF(unittest.TestCase):
    pass

if __name__ == '__main__':
    unittest.main()