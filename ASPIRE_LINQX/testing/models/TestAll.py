import unittest
import TestParameter
# import TestVParameter
import TestCommand
# import TestVCommand
import TestContainer
import TestWorkflow
# import TestVWorkflow
import TestOptimizer
# import TestScheduler

def test_all():
    all_suites = [
        unittest.TestLoader().loadTestsFromModule(TestParameter),
        # unittest.TestLoader().loadTestsFromModule(TestVParameter),
        unittest.TestLoader().loadTestsFromModule(TestCommand),
        unittest.TestLoader().loadTestsFromModule(TestWorkflow),
        unittest.TestLoader().loadTestsFromModule(TestOptimizer),
        # unittest.TestLoader().loadTestsFromModule(TestVCommand),
        # unittest.TestLoader().loadTestsFromModule(TestVWorkflow),
        # unittest.TestLoader().loadTestsFromModule(TestScheduler),
        unittest.TestLoader().loadTestsFromModule(TestContainer)
    ]
    return unittest.TestSuite(all_suites)

if __name__ == "__main__":
    TestSuite = test_all()
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(TestSuite)