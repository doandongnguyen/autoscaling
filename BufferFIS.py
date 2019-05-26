from fuzzy import FIS


# Defined membership functions for Buffer Environments
class Buffer_Fis():
    def __init__(self):
        # Workloads
        l = FIS.InputStateVariable(
            FIS.Trapeziums(0, 0, 50, 100),
            FIS.Trapeziums(50, 100, 150, 200),
            FIS.Trapeziums(150, 200, 250, 300),
            FIS.Trapeziums(250, 300, 350, 400),
            FIS.Trapeziums(350, 400, 450, 450),
        )
        # Buffers
        b = FIS.InputStateVariable(
            FIS.Trapeziums(0, 0, 20, 40),
            FIS.Trapeziums(20, 50, 50, 80),
            FIS.Trapeziums(60, 80, 100, 100))
        self.rules = FIS.Rules(l, b)
        self.fis = FIS.FIS(Rules=self.rules)

    def get_truth_values(self, state):
        return self.fis.truth_values(state)
