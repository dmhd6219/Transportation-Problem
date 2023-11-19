from pprint import pprint

import numpy as np


class TransportationProblem:
    n: int
    m: int
    supply: np.ndarray[int]
    demand: np.ndarray[int]
    costs: np.ndarray[np.ndarray[int]]

    def __init__(self):
        self.parse_input()
        self.check_input()

    def parse_input(self) -> None:
        self.n = int(input("Enter number of source points : \n"))
        self.m = int(input("Enter number of demand points : \n"))

        self.supply = self.string2array(input("Enter source points : \n"))
        if len(self.supply) != self.n:
            raise ValueError(f"Expected {self.n} arguments, got {len(self.supply)}")

        print("Input costs matrix :")
        self.costs = np.zeros((self.n, self.m))
        input_index = 0
        input_string = input()
        while input_string != "":
            self.costs[input_index] = self.string2array(input_string)
            if len(self.costs[input_index]) != self.m:
                raise ValueError(f"Expected {self.m} arguments, got {len(self.costs[input_index])}")
            input_index += 1
            input_string = input()

        if input_index != self.n:
            raise ValueError(f"Expected {self.n} arguments, got {input_index}")

        self.demand = self.string2array(input("Enter demand points : \n"))
        if len(self.demand) != self.m:
            raise ValueError(f"Expected {self.m} arguments, got {len(self.demand)}")

    def check_input(self) -> None:
        pass

    @staticmethod
    def string2array(string: str) -> np.ndarray[int]:
        return np.array([int(x) for x in string.split()])

    def solve_with_north_west_corner(self) -> np.ndarray[np.ndarray[int]]:
        start_row = 0
        start_column = 0

        costs = np.zeros_like(self.costs)
        supply = self.supply.copy()
        demand = self.demand.copy()

        while start_row != self.n and start_column != self.m:
            costs[start_row, start_column] = min(supply[start_row], demand[start_column])
            supply[start_row] -= costs[start_row, start_column]
            demand[start_column] -= costs[start_row, start_column]
            if supply[start_row] == 0:
                start_row += 1
            elif demand[start_column] == 0:
                start_column += 1

        return costs

    def solve_with_vogel_approximation(self) -> np.ndarray[np.ndarray[int]]:
        pass

    def solve_with_russel_approximation(self):
        pass


def main() -> None:
    solver = TransportationProblem()

    print("North-West Corner method returned :")
    pprint(solver.solve_with_north_west_corner())

    print("Vogel's Approximation method returned :")
    pprint(solver.solve_with_vogel_approximation())

    print("Russel's Approximation method returned :")
    pprint(solver.solve_with_russel_approximation())


if __name__ == "__main__":
    main()
