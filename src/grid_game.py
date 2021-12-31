#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 10:00:47 2021

@author: alexxcollins

The first task tests your Python skills and capacity to plan a statistical 
analysis. You need to develop a simple game consisting of a rectangular grid 
(of size height x width) where each cell has a random value between 0 and n. 
An agent starts at the upper-left corner of the grid and must reach the 
lower-right corner of the grid as fast as possible. Accordingly, the task 
consists on finding the shortest path.

There are two game modes:
• The time spent on a cell is the number on this cell
• The time spent on a cell is the absolute of the difference between the 
previous cell the
agent was on and the current cell it is on
"""

#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

#%% Class to implement the game


class GridGame:
    def __init__(self, size=(3, 3), max_value=9, mode="absolute"):
        """
        Sets up the grid object to play the game, as well as sets the rules
        mode

        Parameters
        ----------
        size : tuple of integers, optional
            denotes size of grid. The default is (3,3).
        mode : string, optional
            either 'absolute' or 'relative'. The default is 'absolute'.
        """

        self.size = size
        self.max_value = max_value
        self.grid = np.random.randint(max_value + 1, size=size)
        # create empty array which will store the time needed for
        # shortest currently calculated paths from each coordinate
        self.fastest_path_time = np.ones(shape=size) * np.inf
        # create an empty array to hold the actual paths for shortest route
        self.fastest_path = np.empty(shape=size, dtype=object)
        self.mode = mode
        # boolean to state whether a search for a better path has made any
        # changes to the current fastest paths. Starts at True
        self.paths_updated = True
        # to count how many times code revisits each square to revise best
        # strategy.
        self.iteration_counter = 0

    def visualise_grid(self, path=[]):
        """
        Create a matplotlib chart visualisation with gridlines

        There is very likely a much less convoluted way to do this!
        """
        color_matrix = np.zeros(shape=self.size)
        for posn in path:
            color_matrix[posn] = 0.5
        font_size = min(15, 160 / max(self.size[0], self.size[1]))

        fig, ax = plt.subplots(facecolor="lightgray")

        plt.imshow(color_matrix, cmap="Greens", norm=mpl.colors.Normalize(vmax=1.0))
        ax.set_xticks(np.arange(0, self.size[1]))
        ax.set_yticks(np.arange(0, self.size[0]))
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        ax.tick_params(labelsize=font_size)

        # set up minor ticks and gridlines
        ax.set_xticks(np.arange(-0.5, self.size[1]), minor=True)
        ax.set_yticks(np.arange(-0.5, self.size[0]), minor=True)
        ax.tick_params(
            which="minor", left=False, bottom=False, labeltop=False, labelleft=False
        )
        ax.grid(True, which="minor", color="k")

        # # write numbers on the grid
        for i in np.arange(self.size[0]):
            for j in np.arange(self.size[1]):
                ax.text(
                    j,
                    i,
                    str(self.grid[i, j]),
                    size=font_size,
                    horizontalalignment="center",
                    verticalalignment="center",
                    # transform=ax.transData
                )
        plt.show()

    def move_time(self, from_posn, to_posn):
        """
        calculate the time needed to move from one square to the next. No
        restrictions on how player can move.

        Parameters
        ----------
        from_posn : tuple in form (i,j) where i, j are integers
            describes the position the player is moving from
        to_posn : tuple in form (i,j) where i, j are integers
            describes the position the player is moving to

        Returns
        -------
        time taken to move from one square to another square

        """
        if self.mode == "relative":
            time = abs(self.grid[to_posn] - self.grid[from_posn])
        elif self.mode == "absolute":
            time = self.grid[to_posn]
        else:
            raise ValueError("mode must be either 'relative' or 'absolute'")

        return time

    def moves(self, from_posn, any_direction=True):
        """
        Create list of allowable moves from one square to the next. Includes
        toggle to allow moves only down and right (i.e. towards lower right) or
        moves in any direction

        Parameters
        ----------
        from_posn : tuple of integers
            represents the square player is moving from
        any_direction : boolean, optional
            True to allow any move, False to restrict to moving down or right
        Returns
        -------
        list of destination squares with 1 to 4 elements.

        """
        moves = []
        # if grid is size (3,3), say, then x coordinates are [0, 1, 2], and so
        # only coords [0, 1] can move move right. So test is < size[0] - 1.
        if from_posn[0] < self.size[0] - 1:
            moves += [(from_posn[0] + 1, from_posn[1])]
        if from_posn[1] < self.size[1] - 1:
            moves += [(from_posn[0], from_posn[1] + 1)]
        if any_direction:
            if from_posn[0] > 0:
                moves += [(from_posn[0] - 1, from_posn[1])]
            if from_posn[1] > 0:
                moves += [(from_posn[0], from_posn[1] - 1)]
        return moves

    def iterate_path(self, any_direction=True):
        """
        Implements recursive search. If final point is at (n,n) then first
        work out fastest paths and their times from (n-1,n) and (n,n-1).

        Then work out fastest paths and times from (n-2,n), (n-1,n-1), (n,n-2)
        etc. Record time taken for fastest path from each square on grid and
        the sequence of moves.

        Intention is that 'any_direction' is set to False for first iteration
        to restrict moves only to the right or down. After that moves can be
        up or left if that finds a faster route.

        A grid of size (a x b) needs a + b - 2 iterations to visit every
        diagonal
        """
        old_grid = self.fastest_path_time.copy()
        for a in range(1, self.size[0] + self.size[1] - 1):
            # define diagonal squares adjacent to the last diagonal where we
            # have calculated paths. This list could include coordinates off-
            # grid - e.g. a 4x8 grid could have posn of (-1,8) below. These
            # possibilities will be dealt with afterwards
            start_posns = [
                (self.size[0] - 1 - a + q, self.size[1] - 1 - q) for q in range(a + 1)
            ]

            for posn in start_posns:
                # check posn is actually in the grid, and only run if it is
                if posn[0] in range(self.size[0]) and posn[1] in range(self.size[1]):
                    moves = self.moves(posn, any_direction=any_direction)

                    for i, move in enumerate(moves):
                        if (
                            self.move_time(posn, move) + self.fastest_path_time[move]
                            < self.fastest_path_time[posn]
                        ):

                            self.fastest_path_time[posn] = (
                                self.move_time(posn, move)
                                + self.fastest_path_time[move]
                            )
                            self.fastest_path[posn] = self.fastest_path[move] + [posn]

        if np.array_equal(old_grid, self.fastest_path_time):
            self.paths_updated = False
        self.iteration_counter += 1

        return None

    def solve_game(self, show_iterations=False):
        """
        Run the iterations until game is solved

        Returns
        -------
        None.

        """
        # need to assign values to bottom right of grid
        self.fastest_path_time[-1, -1] = 0
        self.fastest_path[-1, -1] = [(self.size[0] - 1, self.size[1] - 1)]
        self.iter_counter = 1
        self.iterate_path(any_direction=False)
        best_path = self.fastest_path[0, 0].copy()
        print("\nFastest path takes {:d}".format(int(self.fastest_path_time[0, 0])))
        if show_iterations:
            self.visualise_grid(path=best_path)
        while self.paths_updated:
            self.iter_counter += 1
            self.iterate_path()
            if show_iterations:
                if np.array_equal(best_path, self.fastest_path[0, 0]) is False:
                    best_path = self.fastest_path[0, 0].copy()
                    print(
                        "Wahoo! Fastest path now takes {:d}".format(
                            int(self.fastest_path_time[0, 0])
                        )
                    )
                    self.visualise_grid(path=best_path)
        print("\nCode required {:d} iterations".format(self.iter_counter))

    def solve_game_timed(self):
        """
        Run the iterations until game is solved. No extra logic or
        visualisation so that algorithm itself can be timed.

        Returns
        -------
        None.

        """
        self.iter_counter = 1
        self.iterate_path(any_direction=False)

        while self.paths_updated:
            self.iter_counter += 1
            self.iterate_path()

        return None

    def create_unvisited_list(self):
        """
        Create list of unvisited nodes for Djikstra algorithm.
        Set attribute value.

        Returns
        -------
        None.

        """
        unvisited = list()
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                unvisited.append((i, j))
        # don't think I need to remove the source at the start
        # unvisited.remove(self.source)
        self.unvisited_list = unvisited

    def djikstra_shortest_path(self):
        """
        Find shortest path. Necessary for each step of Djikstra's algorithm.

        Returns
        -------
        None.

        """
        S = [self.destination]
        u = S[-1]
        while u != self.source:
            S.append(self.prev_node[u])
            u = S[-1]
        self.djikstra_path = S

    def djikstra(self, source=(0, 0), destination=None):
        """
        implement Djikstra's algorithm

        The priority queue is maintained within the algorithm. In future iterations
        of code I would create a class for the priority queue.
        The dict self.unvisited_time holdes the dictionary of unvsitied nodes
        with key as node id and value as shortest path to that node. Values
        are initialised at np.inf. When a node has a viable path to it, the value
        will be an integer representing shortest path to that node.

        Returns
        -------
        None.

        """
        # the starting cell. It obvsiously takes zero time to "travel to" the
        # starting point:
        self.source = source
        if destination is None:
            self.destination = (self.size[0] - 1, self.size[1] - 1)
        else:
            self.destination = destination
        self.create_unvisited_list()  # method sets self.unvisited_list attribute.
        # Inialises list with all cells in grid

        self.visited_list = []
        # for priority queue, use dict with keys as indices and
        # values as current shortest paths
        self.unvisited_time = dict(
            zip(self.unvisited_list, [np.inf] * len(self.unvisited_list))
        )
        self.unvisited_time[self.source] = 0  # time to reach starting cell

        # use dict.pop((x,y)) to remove nodes that have been visited and return
        # the value of the shortest path to be added to the visited dict
        self.visited_time = {}

        self.prev_node = dict(
            zip(self.unvisited_list, [np.nan] * len(self.unvisited_list))
        )

        # use min(dict) to return the key corresponding to minimum path
        # this will be the node to start exploring now
        # this is part of the priority queue pocess: return location with min
        # path length
        while len(self.unvisited_time) > 0:
            # u is the unvisited node with smallest time
            # following function finds min of all dict keys (i.e. shortest paths)
            u = min(self.unvisited_time, key=self.unvisited_time.get)
            # take node out of unvisited "list" (dict) and put into visited list
            self.visited_time[u] = self.unvisited_time.pop(u)
            if u == self.destination:
                break

            neigbours = self.moves(from_posn=u)
            # neigbours just gives all possible moves - up, down, left, right
            # moves may be off grid or to visited nodes. So use Try to discount
            # these possibilities
            for n in neigbours:
                ### if x in dict.keys()
                # this code works but I have since learned that I should avoid
                # "try" and just use "if x in dict.keys()" as above.
                if n in self.unvisited_time.keys():
                    # try:
                    # next line is just to get code to fail if n is not a
                    # key of self.unvisited_time
                    _ = self.unvisited_time[n]
                    alt = self.visited_time[u] + self.move_time(u, n)
                    if alt < self.unvisited_time[n]:
                        self.unvisited_time[n] = alt
                        self.prev_node[n] = u
                # except Exception:
                # continue
        self.djikstra_shortest_path()


#%% test
def test(size=(10, 10), max_value=9, mode="absolute", show_its=True):
    game = GridGame(size=size, max_value=max_value, mode=mode)
    game.visualise_grid()
    game.solve_game(show_iterations=show_its)
    game.djikstra()
    game.visualise_grid(game.djikstra_path)
    return game


def break_the_algo():
    # this reads an unusual path
    from pathlib import Path
    import pandas as pd

    folder = Path("../data")
    file = "test_grid.csv"
    df = pd.read_csv(folder / file, header=None)
    game = GridGame(size=(6, 5))
    game.grid = df.values
    game.solve_game(show_iterations=True)


# this doesn't work!
import timeit


def timed_test(size=(10, 10), max_value=9, mode="absolute", number=20):
    game = GridGame(size=size, max_value=max_value, mode=mode)
    game.solve_game_timed()
    return None


# uncomment line below to time the algorithm (without the visualisations)
# print(timeit.timeit('timed_test(size=(25,25), max_value = 9, mode="absolute")',number=100, globals=globals()))


# this doesn't work!
import timeit


def timed_test(size=(10, 10), max_value=9, mode="absolute", number=20):
    game = GridGame(size=size, max_value=max_value, mode=mode)
    game.solve_game_timed()
    return None


# uncomment line below to time the algorithm (without the visualisations)
# print(timeit.timeit('timed_test(size=(25,25), max_value = 9, mode="absolute")',number=100, globals=globals()))
