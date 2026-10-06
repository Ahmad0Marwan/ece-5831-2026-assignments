import numpy as np


class LogicGate:
    def __init__(self):
        self.w1 = None
        self.w2 = None
        self.th = None
        self.out = None
        self.x1 = self.x2 = None

    def print_output(self, gate):
        if gate == "AND":
            print(f"Output of AND logic is: {self.out}, with x1 = {self.x1}, x2 = {self.x2}")
        elif gate == "NAND":
            print(f"Output of NAND logic is: {self.out}, with x1 = {self.x1}, x2 = {self.x2}")
        elif gate == "OR":
            print(f"Output of OR logic is: {self.out}, with x1 = {self.x1}, x2 = {self.x2}")
        elif gate == "NOR":
            print(f"Output of NOR logic is: {self.out}, with x1 = {self.x1}, x2 = {self.x2}")
        elif gate == "XOR":
            print(f"Output of XOR logic is: {self.out}, with x1 = {self.x1}, x2 = {self.x2}")

    def and_gate(self, x1, x2):
        self.w1 = 0.5
        self.w2 = 0.5
        self.th = 0.99

        self.x1 = x1
        self.x2 = x2

        x = np.array([x1, x2])
        w = np.array([self.w1, self.w2])

        if np.dot(x, w) > self.th:
            self.out = 1
            return 1
        else:
            self.out = 0
            return 0

    def nand_gate(self, x1, x2):
        self.w1 = -1
        self.w2 = -1
        self.th = -1.5

        self.x1 = x1
        self.x2 = x2

        x = np.array([x1, x2])
        w = np.array([self.w1, self.w2])

      
        if np.dot(x, w) > self.th:
            self.out = 1
            return 1
        else:
            self.out = 0
            return 0

    def or_gate(self, x1, x2):
        self.w1 = 0.5
        self.w2 = 0.5
        self.th = 0.0

        self.x1 = x1
        self.x2 = x2

        x = np.array([x1, x2])
        w = np.array([self.w1, self.w2])

      
        if np.dot(x, w) > self.th:
            self.out = 1
            return 1
        else:
            self.out = 0
            return 0


    def nor_gate(self, x1, x2):
        self.w1 = -1
        self.w2 = -1
        self.th = -0.5

        self.x1 = x1
        self.x2 = x2

        x = np.array([x1, x2])
        w = np.array([self.w1, self.w2])

     
        if np.dot(x, w) > self.th:
            self.out = 1
            return 1
        else:
            self.out = 0
            return 0

    def xor_gate(self, x1, x2):
        c = self.or_gate(x1, x2)
        d = self.nand_gate(x1, x2)
        e = self.and_gate(c, d)

        self.x1 = x1
        self.x2 = x2

        if e == 1:
            self.out = 1
            return 1
        else:
            self.out = 0
            return 0


if __name__ == "__main__":
    logic_gate = LogicGate()

    logic_gate.and_gate(1, 1)
    logic_gate.print_output("AND")

    logic_gate.nand_gate(1, 1)
    logic_gate.print_output("NAND")

    logic_gate.or_gate(1, 0)
    logic_gate.print_output("OR")

    logic_gate.nor_gate(0, 0)
    logic_gate.print_output("NOR")

    logic_gate.xor_gate(0, 1)
    logic_gate.print_output("XOR")
