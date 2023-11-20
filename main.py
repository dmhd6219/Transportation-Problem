#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pprint import pprint

import numpy as np

MAX_INT = np.iinfo(np.intc).max


class TransportationProblem:
    """
        Represents a Transportation Problem solver using various approximation methods.

        Attributes:
        - n (int): Number of suppliers.
        - m (int): Number of consumers.
        - supply (np.ndarray[int]): Array representing the supply from each supplier.
        - demand (np.ndarray[int]): Array representing the demand from each consumer.
        - costs (np.ndarray[np.ndarray[int]]): 2D array representing the transportation costs.
    """

    n: int
    m: int
    supply: np.ndarray[int]
    demand: np.ndarray[int]
    costs: np.ndarray[np.ndarray[int]]

    def __init__(self):
        """
            Initializes a TransportationProblem instance by parsing input and checking validity.
        """
        self.__parse_input()
        self.__check_input()

        print(f"Supply vector :")
        pprint(self.supply)
        print()

        print(f"Demand vector :")
        pprint(self.demand)
        print()

        print(f"Costs matrix :")
        pprint(self.costs)
        print()
        print()

    def __parse_input(self) -> None:
        """
            Parses input from a file and initializes problem parameters.
        """
        with open("./test.txt") as test:
            self.supply = self.__string2array(test.readline().strip("\n"))
            self.demand = self.__string2array(test.readline().strip("\n"))

            self.n = len(self.supply)
            self.m = len(self.demand)

            self.costs = np.zeros((self.n, self.m))

            for i in range(self.n):
                input_string = test.readline().strip("\n")
                try :
                    self.costs[i] = self.__string2array(input_string)
                except ValueError:
                    print("The method is not applicable!")
                    exit(1)

    def __check_input(self) -> None:
        """
            Checks the validity of the parsed input.
        """
        if self.n != self.supply.size or self.m != self.demand.size:
            print("The method is not applicable!")
            exit(1)

        if sum(self.supply) != sum(self.demand):
            print("The problem is not balanced!")
            exit(1)

    @staticmethod
    def __string2array(string: str) -> np.ndarray[int]:
        """
            Converts a space-separated string to a NumPy integer array.

            Args:
            - string (str): Space-separated string.

            Returns:
            - np.ndarray[int]: Integer array.
        """
        return np.array([int(x) for x in string.split()])

    @staticmethod
    def __allocate_at_min_cost(
            start_row: int,
            start_column: int,
            costs: np.ndarray[np.ndarray[int]],
            supply: np.ndarray[int],
            demand: np.ndarray[int],
    ) -> None:
        """
            Allocates transportation at minimum cost in the given row and column.

            Args:
            - start_row (int): Starting row index.
            - start_column (int): Starting column index.
            - costs (np.ndarray[np.ndarray[int]]): Transportation costs matrix.
            - supply (np.ndarray[int]): Array representing the supply from each supplier.
            - demand (np.ndarray[int]): Array representing the demand from each consumer.
        """

        costs[start_row, start_column] = min(supply[start_row], demand[start_column])
        supply[start_row] -= costs[start_row, start_column]
        demand[start_column] -= costs[start_row, start_column]

    def solve_with_north_west_corner(self) -> np.ndarray[np.ndarray[int]]:
        """
            Solves the Transportation Problem using the North-West Corner method and returns the solution.

            Returns:
            - np.ndarray[np.ndarray[int]]: Solution matrix.
        """
        start_row = 0
        start_column = 0

        costs = np.zeros_like(self.costs)
        supply = self.supply.copy()
        demand = self.demand.copy()

        while start_row != self.n and start_column != self.m:
            self.__allocate_at_min_cost(start_row, start_column, costs, supply, demand)

            start_row += 1 if supply[start_row] == 0 else 0
            start_column += 1 if demand[start_column] == 0 else 0

        return costs

    def __find_diff(self, costs: np.ndarray[np.ndarray[int]]):
        """
            Finds the difference between the two smallest and the two smallest values in each row and column.

            Args:
            - costs (np.ndarray[np.ndarray[int]]): Transportation costs matrix.

            Returns:
            - Tuple[np.ndarray[int], np.ndarray[int]]: Tuple containing row differences and column differences.
        """

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
        """
            Solves the Transportation Problem using Vogel's Approximation method and returns the solution.

            Returns:
            - np.ndarray[np.ndarray[int]]: Solution matrix.
        """
        ans = np.zeros_like(self.costs)
        costs = self.costs.copy()
        supply = self.supply.copy()
        demand = self.demand.copy()

        while np.max(supply) != 0 or np.max(demand) != 0:
            row, col = self.__find_diff(costs)
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
    def __update_max_values(
            n: int,
            m: int,
            u: np.ndarray[int],
            v: np.ndarray[int],
            costs: np.ndarray[np.ndarray[int]],
            supply: np.ndarray[int],
            demand: np.ndarray[int],
    ) -> None:
        """
            Updates the maximum values for each row and column in the given arrays.

            Args:
            - n (int): Number of suppliers.
            - m (int): Number of consumers.
            - u (np.ndarray[int]): Array representing the dual variable for each supplier.
            - v (np.ndarray[int]): Array representing the dual variable for each consumer.
            - costs (np.ndarray[np.ndarray[int]]): Transportation costs matrix.
            - supply (np.ndarray[int]): Array representing the supply from each supplier.
            - demand (np.ndarray[int]): Array representing the demand from each consumer.
        """

        for i in range(n):
            u[i] = max(costs[i, :]) if supply[i] > 0 else u[i]
        for j in range(m):
            v[j] = max(costs[:, j]) if demand[j] > 0 else v[j]

    @staticmethod
    def __find_max_position(
            u: np.ndarray[int],
            v: np.ndarray[int],
            costs: np.ndarray[np.ndarray[int]],
            supply: np.ndarray[int],
            demand: np.ndarray[int],
    ) -> tuple[int, int]:
        """
            Finds the position with the maximum Russell value in the given arrays.

            Args:
            - u (np.ndarray[int]): Array representing the dual variable for each supplier.
            - v (np.ndarray[int]): Array representing the dual variable for each consumer.
            - costs (np.ndarray[np.ndarray[int]]): Transportation costs matrix.
            - supply (np.ndarray[int]): Array representing the supply from each supplier.
            - demand (np.ndarray[int]): Array representing the demand from each consumer.

            Returns:
            - Tuple[int, int]: Tuple containing the row and column indices of the maximum position.
        """
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
    def __allocate_at_max_position(
            ans: np.ndarray[np.ndarray[int]],
            max_pos: tuple[int, int],
            supply: np.ndarray[int],
            demand: np.ndarray[int],
    ) -> None:
        """
            Allocates transportation at the position with the maximum Russell value.

            Args:
            - ans (np.ndarray[np.ndarray[int]]): Solution matrix.
            - max_pos (Tuple[int, int]): Tuple containing the row and column indices of the maximum position.
            - supply (np.ndarray[int]): Array representing the supply from each supplier.
            - demand (np.ndarray[int]): Array representing the demand from each consumer.
        """
        allocation = min(supply[max_pos[0]], demand[max_pos[1]])
        ans[max_pos[0], max_pos[1]] = allocation
        supply[max_pos[0]] -= allocation
        demand[max_pos[1]] -= allocation

    def solve_with_russel_approximation(self):
        """
            Solves the Transportation Problem using Russell's Approximation method and returns the solution.

            Returns:
            - np.ndarray[np.ndarray[int]]: Solution matrix.
        """
        ans = np.zeros_like(self.costs)

        u = np.full(self.n, -MAX_INT)
        v = np.full(self.m, -MAX_INT)

        supply = self.supply.copy()
        demand = self.demand.copy()
        costs = self.costs.copy()

        while supply.sum() > 0 and demand.sum() > 0:
            self.__update_max_values(self.n, self.m, u, v, costs, supply, demand)
            max_pos = self.__find_max_position(u, v, costs, supply, demand)
            self.__allocate_at_max_position(ans, max_pos, supply, demand)

        return ans


def main() -> None:
    """
        Entry point of the script; initializes the solver and prints results.
    """
    solver = TransportationProblem()

    print("North-West Corner method returned :")
    pprint(solver.solve_with_north_west_corner())
    print()

    print("Vogel's Approximation method returned :")
    pprint(solver.solve_with_vogel_approximation())
    print()

    print("Russel's Approximation method returned :")
    pprint(solver.solve_with_russel_approximation())
    print()


if __name__ == "__main__":
    main()
