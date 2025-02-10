import operator

# Class for a disjunction statement of a CNF statment  
# CNF = ((disjuction statement) or ()) and (() or ())
class DisjunctionStatement():

    # Supported statement operators
    OPERATORS = (
        operator.lt,
        operator.le,
        operator.eq,
        operator.ne,
        operator.ge,
        operator.gt,
    )

    # Initalize the statement object
    # system statement: value of the offer from the system
    # request statement: value of the offer from the request
    # operation: operation to be applied
    # not_flag: allows for negation
    def __init__(self,system_statement,request_statement,operation,not_flag=True):
        self.system_statement = system_statement
        self.request_statement = request_statement 
        if operation not in DisjunctionStatement.OPERATORS:
            raise Exception
        self.operation = operation
        self.not_flag = not_flag
    
    # Evaluate the statement to True or False
    def eval(self) -> bool:
        if self.not_flag == True:
            return self.operation(self.system_statement,self.request_statement)
        else:
            return not self.operation(self.system_statement,self.request_statement)

    # Print the statement in string format
    def __str__(self) -> str:
        return "({} {} {})".format(self.system_statement,self.operation,self.request_statement)

# Class for a disjunction statement of a CNF statment  
# Defined as a list of disjunction statements
# CNF = (conjunction statement) and (conjunction statement)
class ConjunctionStatement():

    # Initalize the conjunction statement
    # disjunctive_statements: a list of disjunctive statements
    def __init__(self, disjunctive_statements=[]):
        if not isinstance(disjunctive_statements, list):
            raise Exception
        if not all(isinstance(elem,DisjunctionStatement) for elem in disjunctive_statements):
            raise Exception
        self.disjunctive_statements = disjunctive_statements
    
    # Adds a new disjunctive statement to the list
    # statement: the disjunctive statetement to be added
    def append(self, statement):
        if not isinstance(statement, DisjunctionStatement):
            raise Exception
        self.disjunctive_statements.append(statement)
    
    # Evaluates the conjunctive statement to True or False
    def eval(self) -> bool:
        if not all(isinstance(elem,DisjunctionStatement) for elem in self.disjunctive_statements):
            raise Exception
        boolean_list = [elem.eval() for elem in self.disjunctive_statements]
        return any(elem for elem in boolean_list)

    # Prints out the conjunctive statement as a string
    def __str__(self) -> str:
        return "({})".format(" or ".join(str(elem) for elem in self.disjunctive_statements))

# Class for a CNF statement, a list of conjunctive statements
class CNF():

    # Initalizes the CNF statement to a list of conjunctive statements
    # conjunctive_statements: the list of conjunctive statements
    def __init__(self, conjunctive_statements=[]):
        if not isinstance(conjunctive_statements, list):
            raise Exception
        if not all(isinstance(elem,ConjunctionStatement) for elem in conjunctive_statements):
            raise Exception
        self.conjunctive_statements = conjunctive_statements

    # Adds a new conjunctive statement to the CNF 
    # statement: the statement to be added
    def append(self, statement):
        if not isinstance(statement, ConjunctionStatement):
            raise Exception
        self.conjunctive_statements.append(statement)

    # Evaluates the CNF
    def eval(self) -> bool:
        if not all(isinstance(elem,ConjunctionStatement) for elem in self.conjunctive_statements):
            raise Exception
        boolean_list = [elem.eval() for elem in self.conjunctive_statements]
        return all(elem for elem in boolean_list)

    # Prints the CNF as a string
    def __str__(self) -> str:
        return "({})".format(" and ".join(self.conjunctive_statements))