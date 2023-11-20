from pprint import pprint

import numpy as np

MAX_INT = np.iinfo(np.intc).max


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
        with open("./test.txt") as test:
            self.supply = self.string2array(test.readline().strip("\n"))
            self.demand = self.string2array(test.readline().strip("\n"))

            self.n = len(self.supply)
            self.m = len(self.demand)

            self.costs = np.zeros((self.n, self.m))

            for i in range(self.n):
                input_string = test.readline().strip("\n")
                self.costs[i] = self.string2array(input_string)

    def check_input(self) -> None:
        pass

    @staticmethod
    def string2array(string: str) -> np.ndarray[int]:
        return np.array([int(x) for x in string.split()])

    @staticmethod
    def allocate_at_min_cost(
            start_row: int,
            start_column: int,
            costs: np.ndarray[np.ndarray[int]],
            supply: np.ndarray[int],
            demand: np.ndarray[int],
    ) -> None:
        costs[start_row, start_column] = min(supply[start_row], demand[start_column])
        supply[start_row] -= costs[start_row, start_column]
        demand[start_column] -= costs[start_row, start_column]

    def solve_with_north_west_corner(self) -> np.ndarray[np.ndarray[int]]:
        start_row = 0
        start_column = 0

        costs = np.zeros_like(self.costs)
        supply = self.supply.copy()
        demand = self.demand.copy()

        while start_row != self.n and start_column != self.m:
            self.allocate_at_min_cost(start_row, start_column, costs, supply, demand)

            start_row += 1 if supply[start_row] == 0 else 0
            start_column += 1 if demand[start_column] == 0 else 0

        return costs

    def find_diff(self, costs: np.ndarray[np.ndarray[int]]):

        row_diff = np.array([])
        col_diff = np.array([])

        for i in range(self.n):
            arr = costs[i][:]
            arr = np.sort(arr)
            row_diff = np.append(row_diff, arr[1] - arr[0])
        col = 0

        while col < self.m:
            arr = np.array([])
            for i in range(self.n):
                arr = np.append(arr, costs[i][col])
            arr = np.sort(arr)
            col += 1
            col_diff = np.append(col_diff, arr[1] - arr[0])
        return row_diff, col_diff

    def solve_with_vogel_approximation(self) -> np.ndarray[np.ndarray[int]]:
        ans = np.zeros_like(self.costs)
        costs = self.costs.copy()
        supply = self.supply.copy()
        demand = self.demand.copy()

        while np.max(supply) != 0 or np.max(demand) != 0:
            row, col = self.find_diff(costs)
            row_max = np.max(row)
            row_col = np.max(col)

            if row_max >= row_col:
                for row_index, row_value in enumerate(row):
                    if row_value == row_max:
                        row_min = np.min(costs[row_index])

                        for col_index, col_value in enumerate(costs[row_index]):
                            if col_value == row_min:
                                min_value = min(supply[row_index], demand[col_index])

                                ans[row_index][col_index] = min_value

                                supply[row_index] -= min_value
                                demand[col_index] -= min_value
                                if demand[col_index] == 0:
                                    for r in range(self.n):
                                        costs[r][col_index] = MAX_INT
                                else:
                                    costs[row_index] = [MAX_INT for _ in range(self.m)]
                                break
                        break
            else:
                for row_index, row_value in enumerate(col):
                    if row_value == row_col:
                        row_min = MAX_INT
                        for j in range(self.n):
                            row_min = min(row_min, costs[j][row_index])

                        for col_index in range(self.n):
                            col_value = costs[col_index][row_index]
                            if col_value == row_min:
                                min_value = min(supply[col_index], demand[row_index])
                                ans[col_index][row_index] = min_value
                                supply[col_index] -= min_value
                                demand[row_index] -= min_value
                                if demand[row_index] == 0:
                                    for r in range(self.n):
                                        costs[r][row_index] = MAX_INT
                                else:
                                    costs[col_index] = [MAX_INT for _ in range(self.m)]
                                break
                        break
        return ans

    @staticmethod
    def update_max_values(
            n: int,
            m: int,
            u: np.ndarray[int],
            v: np.ndarray[int],
            costs: np.ndarray[np.ndarray[int]],
            supply: np.ndarray[int],
            demand: np.ndarray[int],
    ) -> None:
        for i in range(n):
            u[i] = max(costs[i, :]) if supply[i] > 0 else u[i]
        for j in range(m):
            v[j] = max(costs[:, j]) if demand[j] > 0 else v[j]

    @staticmethod
    def find_max_position(
            u: np.ndarray[int],
            v: np.ndarray[int],
            costs: np.ndarray[np.ndarray[int]],
            supply: np.ndarray[int],
            demand: np.ndarray[int],
    ) -> tuple[int, int]:
        max_value = -MAX_INT
        max_pos = -1, -1
        for i in range(len(u)):
            for j in range(len(v)):
                if supply[i] > 0 and demand[j] > 0:
                    russell_value = u[i] + v[j] - costs[i, j]
                    if russell_value > max_value:
                        max_value = russell_value
                        max_pos = i, j
        return max_pos

    @staticmethod
    def allocate_at_max_position(
            ans: np.ndarray[np.ndarray[int]],
            max_pos: tuple[int, int],
            supply: np.ndarray[int],
            demand: np.ndarray[int],
    ) -> None:
        allocation = min(supply[max_pos[0]], demand[max_pos[1]])
        ans[max_pos[0], max_pos[1]] = allocation
        supply[max_pos[0]] -= allocation
        demand[max_pos[1]] -= allocation

    def solve_with_russel_approximation(self):
        ans = np.zeros_like(self.costs)

        u = np.full(self.n, -MAX_INT)
        v = np.full(self.m, -MAX_INT)

        supply = self.supply.copy()
        demand = self.demand.copy()
        costs = self.costs.copy()

        while supply.sum() > 0 and demand.sum() > 0:
            self.update_max_values(self.n, self.m, u, v, costs, supply, demand)
            max_pos = self.find_max_position(u, v, costs, supply, demand)
            self.allocate_at_max_position(ans, max_pos, supply, demand)

        return ans


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
