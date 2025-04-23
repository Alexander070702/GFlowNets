"""
An environment inspired by the game of Tetris.
"""

import itertools
import re
import warnings
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from matplotlib.axes import Axes
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import set_device, tint

PIECES = {
    "I": [1, [[1], [1], [1], [1]]],
    "J": [2, [[0, 2], [0, 2], [2, 2]]],
    "L": [3, [[3, 0], [3, 0], [3, 3]]],
    "O": [4, [[4, 4], [4, 4]]],
    "S": [5, [[0, 5, 5], [5, 5, 0]]],
    "T": [6, [[6, 6, 6], [0, 6, 0]]],
    "Z": [7, [[7, 7, 0], [0, 7, 7]]],
}

PIECES_COLORS = {
    0: [255, 255, 255],
    1: [19, 232, 232],
    2: [30, 30, 201],
    3: [240, 110, 2],
    4: [236, 236, 14],
    5: [0, 128, 0],
    6: [125, 5, 126],
    7: [236, 14, 14],
}


class Tetris(GFlowNetEnv):
    """
    Tetris environment: an environment inspired by the game of tetris. It's not
    supposed to be a game, but rather a toy environment with an intuitive state and
    action space.

    The state space is 2D board, with all the combinations of pieces on it. Pieces that
    are added to the board are identified by a number that starts from
    piece_idx * max_pieces_per_type, and is incremented by 1 with each new piece from
    the same type. This number fills in the cells of the board where the piece is
    located. This enables telling apart pieces of the same type.

    The action space is the choice of piece, its rotation and horizontal location
    where to drop the piece. The action space may be constrained according to needs.

    Attributes
    ----------
    width : int
        Width of the board.

    height : int
        Height of the board.

    pieces : list
        Pieces to use, identified by [I, J, L, O, S, T, Z]

    rotations : list
        Valid rotations, from [0, 90, 180, 270]
    """

    def __init__(
        self,
        width: int = 10,
        height: int = 20,
        pieces: List = ["I", "J", "L", "O", "S", "T", "Z"],
        rotations: List = [0, 90, 180, 270],
        allow_redundant_rotations: bool = False,
        allow_eos_before_full: bool = False,
        **kwargs,
    ):
        assert all([p in ["I", "J", "L", "O", "S", "T", "Z"] for p in pieces])
        assert all([r in [0, 90, 180, 270] for r in rotations])
        self.device = set_device(kwargs["device"])
        self.int = torch.int16
        self.width = width
        self.height = height
        self.pieces = pieces
        self.rotations = rotations
        self.allow_redundant_rotations = allow_redundant_rotations
        self.allow_eos_before_full = allow_eos_before_full
        self.max_pieces_per_type = 100
        # Helper functions and dicts
        self.piece2idx = lambda letter: PIECES[letter][0]
        self.idx2piece = {v[0]: k for k, v in PIECES.items()}
        self.piece2mat = lambda letter: tint(
            PIECES[letter][1], int_type=self.int, device=self.device
        )
        self.rot2idx = {0: 0, 90: 1, 180: 2, 270: 3}
        # Check width and height compatibility
        heights, widths = [], []
        for piece in self.pieces:
            for rotation in self.rotations:
                piece_mat = torch.rot90(self.piece2mat(piece), k=self.rot2idx[rotation])
                hp, wp = piece_mat.shape
                heights.append(hp)
                widths.append(wp)
        assert all([self.height >= h for h in widths])
        assert all([self.width >= w for w in widths])
        # Source state: empty board
        self.source = torch.zeros(
            (self.height, self.width), dtype=self.int, device=self.device
        )
        # End-of-sequence action: all -1
        self.eos = (-1, -1, -1)

        # Precompute all possible rotations of each piece and the corresponding binary
        # mask
        self.piece_rotation_mat = {}
        self.piece_rotation_mask_mat = {}
        for p in pieces:
            self.piece_rotation_mat[p] = {}
            self.piece_rotation_mask_mat[p] = {}
            for r in rotations:
                self.piece_rotation_mat[p][r] = torch.rot90(
                    self.piece2mat(p), k=self.rot2idx[r]
                )
                self.piece_rotation_mask_mat[p][r] = self.piece_rotation_mat[p][r] != 0

        # Base class init
        super().__init__(**kwargs)

    def get_action_space(self):
        """
        Constructs list with all possible actions, including eos. An action is
        represented by a tuple of length 3 (piece, rotation, col). The piece is
        represented by its index, the rotation by the integer rotation in degrees
        and the location by horizontal cell in the board of the left-most part of the
        piece.
        """
        actions = []
        pieces_mat = []
        for piece in self.pieces:
            for rotation in self.rotations:
                piece_mat = torch.rot90(self.piece2mat(piece), k=self.rot2idx[rotation])
                if self.allow_redundant_rotations or not any(
                    [torch.equal(p, piece_mat) for p in pieces_mat]
                ):
                    pieces_mat.append(piece_mat)
                else:
                    continue
                for col in range(self.width):
                    if col + piece_mat.shape[1] <= self.width:
                        actions.append((self.piece2idx(piece), rotation, col))
        actions.append(self.eos)
        return actions

    def _drop_piece_on_board(
        self, action, state: Optional[TensorType["height", "width"]] = None
    ):
        """
        Drops a piece defined by the argument action onto the board. It returns an
        updated board (copied) and a boolean variable, which is True if the piece can
        be dropped onto the current and False otherwise.
        """
        if state is None:
            state = self.state.clone().detach()
        board = state.clone().detach()

        piece_idx, rotation, col = action
        piece_mat = self.piece_rotation_mat[self.idx2piece[piece_idx]][rotation]
        piece_mat_mask = self.piece_rotation_mask_mat[self.idx2piece[piece_idx]][
            rotation
        ]
        hp, wp = piece_mat.shape

        # Check if piece goes overboard horizontally
        if col + wp > self.width:
            return board, False

        # Find the highest row occupied by any other piece in the columns where we wish
        # to add the new piece
        occupied_rows = board[:, col : col + wp].sum(1).nonzero()
        if len(occupied_rows) == 0:
            # All rows are unoccupied. Set highest occupied row as the row "below" the
            # board.
            highest_occupied_row = self.height
        else:
            highest_occupied_row = occupied_rows[0, 0]

        # Iteratively attempt to place piece on the board starting from the top.
        # As soon as we reach a row where we can't place the piece because it
        # creates a collision, we can stop the search (since we can't put a piece under
        # this obstacle anyway).
        starting_row = highest_occupied_row - hp
        lowest_valid_row = None
        for row in range(starting_row, self.height - hp + 1):
            if row == -hp:
                # Placing the piece here would make it land fully outside the board.
                # This means that there is no place on the board for the piece
                break

            elif row < 0:
                # It is not possible to place the piece at this row because the piece
                # would not completely be in the board. However, it is still possible
                # to check for obstacles because any obstacle will prevent placing the
                # piece at any position below
                board_section = board[: row + hp, col : col + wp]
                piece_mask_section = piece_mat_mask[-(row + hp) :]
                if (board_section * (piece_mask_section != 0)).any():
                    # An obstacle has been found.
                    break

            else:
                # The piece can be placed here if all board cells under piece are empty
                board_section = board[row : row + hp, col : col + wp]
                if (board_section * piece_mat_mask).any():
                    # The piece cannot be placed here and cannot be placed any lower
                    # because of an obstacle.
                    break
                else:
                    # The piece can be placed here.
                    lowest_valid_row = row

        # Place the piece if possible
        if lowest_valid_row is None:
            # The piece cannot be placed. Return the board as-is.
            return board, False
        else:
            # Get and set index of new piece
            piece_idx = self._get_max_piece_idx(board, piece_idx, incr=1)
            piece_mat[piece_mat_mask] = piece_idx

            # Place the piece on the board.
            row = lowest_valid_row
            board[row : row + hp, col : col + wp] += piece_mat
            return board, True

    def get_mask_invalid_actions_forward(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
    ) -> List:
        """
        Returns a list of length the action space with values:
            - True if the forward action is invalid from the current state.
            - False otherwise.
        """
        if state is None:
            state = self.state.clone().detach()
        if done is None:
            done = self.done
        if done:
            return [True for _ in range(self.policy_output_dim)]
        mask = [False for _ in range(self.policy_output_dim)]
        for idx, action in enumerate(self.action_space[:-1]):
            _, valid = self._drop_piece_on_board(action, state)
            if not valid:
                mask[idx] = True
        if not self.allow_eos_before_full and not all(mask[:-1]):
            mask[-1] = True
        return mask

    def states2proxy(
        self,
        states: Union[
            List[TensorType["height", "width"]], TensorType["height", "width", "batch"]
        ],
    ) -> TensorType["height", "width", "batch"]:
        """
        Prepares a batch of states in "environment format" for a proxy: : simply
        converts non-zero (non-empty) cells into 1s.

        Args
        ----
        states : list of 2D tensors or 3D tensor
            A batch of states in environment format, either as a list of states or as a
            single tensor.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        states = tint(states, device=self.device, int_type=self.int)
        states[states != 0] = 1
        return states

    def states2policy(
        self,
        states: Union[
            List[TensorType["height", "width"]], TensorType["height", "width", "batch"]
        ],
    ) -> TensorType["height", "width", "batch"]:
        """
        Prepares a batch of states in "environment format" for the policy model.

        See states2proxy().

        Args
        ----
        states : list of 2D tensors or 3D tensor
            A batch of states in environment format, either as a list of states or as a
            single tensor.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        states = tint(states, device=self.device, int_type=self.int)
        return self.states2proxy(states).flatten(start_dim=1).to(self.float)

    def state2readable(self, state: Optional[TensorType["height", "width"]] = None):
        """
        Converts a state (board) into a human-friendly string.
        """
        state = self._get_state(state)
        if isinstance(state, tuple):
            readable = str(np.stack(state))
        else:
            readable = str(state.cpu().numpy())
        readable = readable.replace("[[", "[").replace("]]", "]").replace("\n ", "\n")
        return readable

    def readable2state(self, readable, alphabet={}):
        """
        Converts a human-readable string representing a state into a state as a list of
        positions.
        """
        pattern = re.compile(r"\s+")
        state = []
        rows = readable.split("\n")
        for row in rows:
            # Preprocess
            row = re.sub(pattern, " ", row)
            row = row.replace(" ]", "]")
            row = row.replace("[ ", "[")
            # Process
            state.append(
                tint(
                    [int(el) for el in row.strip("[]").split(" ")],
                    int_type=self.int,
                    device=self.device,
                )
            )
        return torch.stack(state)

    def get_parents(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
        action: Optional[Tuple] = None,
    ) -> Tuple[List, List]:
        """
        Determines all parents and actions that lead to state.

        See: _is_parent_action()

        Args
        ----
        state : list
            Representation of a state, as a list of length length where each element is
            the position at each dimension.

        done : bool
            Whether the trajectory is done. If None, done is taken from instance.

        action : None
            Ignored

        Returns
        -------
        parents : list
            List of parents in state format

        actions : list
            List of actions that lead to state for each parent in parents
        """

        if state is None:
            state = self.state.clone().detach()
        if done is None:
            done = self.done
        if done:
            return [state], [self.eos]
        else:
            parents = []
            actions = []
            indices = state.unique()
            for idx in indices[indices > 0]:
                if self._piece_can_be_lifted(state, idx):
                    piece_idx, rotation, col = self._get_idx_rotation_col(state, idx)
                    parent = state.clone().detach()
                    parent[parent == idx] = 0
                    action = (piece_idx, rotation, col)
                    parents.append(parent)
                    actions.append(action)
        return parents, actions

    def step(
        self, action: Tuple[int], skip_mask_check: bool = False
    ) -> Tuple[List[int], Tuple[int], bool]:
        """
        Executes step given an action.

        Args
        ----
        action : tuple
            Action to be executed. An action is a tuple int values indicating the
            dimensions to increment by 1.

        skip_mask_check : bool
            If True, skip computing forward mask of invalid actions to check if the
            action is valid.

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : tuple
            Action executed

        valid : bool
            False, if the action is not allowed for the current state.
        """
        # Generic pre-step checks
        do_step, self.state, action = self._pre_step(
            action, skip_mask_check or self.skip_mask_check
        )
        if not do_step:
            return self.state, action, False
        # If action is eos
        if action == self.eos:
            self.done = True
            self.n_actions += 1
            return self.state, self.eos, True
        # If action is not eos, then perform action
        else:
            state_next, valid = self._drop_piece_on_board(action)
            if valid:
                self.state = state_next
                self.n_actions += 1
            return self.state, action, valid

    def _get_max_trajectory_length(self) -> int:
        """
        Returns the maximum trajectory length of the environment, including the EOS
        action.
        """
        return (self.width * self.height) // 4 + 1

    def set_state(
        self, state: TensorType["height", "width"], done: Optional[bool] = False
    ):
        """
        Sets the state and done. If done is True but incompatible with state (done is
        True, allow_eos_before_full is False and state is not full), then force done
        False and print warning. Also, make sure state is tensor.
        """
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.int16)
        if done is True and not self.allow_eos_before_full:
            mask = self.get_mask_invalid_actions_forward(state, done=False)
            if not all(mask[:-1]):
                done = False
                warnings.warn(
                    f"Attempted to set state\n\n{self.state2readable(state)}\n\n"
                    "with done = True, which is not compatible with "
                    "allow_eos_before_full = False. Forcing done = False."
                )
        return super().set_state(state, done)

    def _piece_can_be_lifted(self, board, piece_idx):
        """
        Returns True if the piece with index piece_idx could be lifted, that is all
        cells of the board above the piece are zeros. False otherwise.
        """
        board_aux = board.clone().detach()
        if piece_idx < self.max_pieces_per_type:
            piece_idx = self._get_max_piece_idx(board_aux, piece_idx, incr=0)
        rows, cols = torch.where(board_aux == piece_idx)
        board_top = torch.cat([board[:r, c] for r, c in zip(rows, cols)])
        board_top[board_top == piece_idx] = 0
        return not any(board_top)

    def _get_idx_rotation_col(self, board, piece_idx):
        piece_idx_base = int(piece_idx / self.max_pieces_per_type)
        board_aux = board.clone().detach()
        piece_mat = self.piece2mat(self.idx2piece[piece_idx_base])
        rows, cols = torch.where(board_aux == piece_idx)
        row = min(rows).item()
        col = min(cols).item()
        hp = max(rows).item() - row + 1
        wp = max(cols).item() - col + 1
        board_section = board_aux[row : row + hp, col : col + wp]
        board_section[board_section != piece_idx] = 0
        board_section[board_section == piece_idx] = piece_idx_base
        for rotation in self.rotations:
            piece_mat_rot = torch.rot90(piece_mat, k=self.rot2idx[rotation])
            if piece_mat_rot.shape == board_section.shape and torch.equal(
                torch.rot90(piece_mat, k=self.rot2idx[rotation]), board_section
            ):
                return piece_idx_base, rotation, col
        raise ValueError(
            f"No valid rotation found for piece {piece_idx} in board {board}"
        )

    def _get_max_piece_idx(
        self, board: TensorType["height", "width"], piece_idx: int, incr: int = 0
    ):
        """
        Gets the index of a new piece with base index piece_idx, based on the board.

        board : tensor
            The current board matrix.

        piece_idx : int
            Piece index, in base format [1, 2, ...]

        incr : int
            Increment of the returned index with respect to the max.
        """

        min_idx = piece_idx * self.max_pieces_per_type
        max_idx = min_idx + self.max_pieces_per_type
        max_relevant_piece_idx = (board * (board < max_idx)).max()

        if max_relevant_piece_idx >= min_idx:
            return max_relevant_piece_idx + incr
        else:
            return min_idx

    def plot_samples_topk(
        self,
        samples: List,
        rewards: TensorType["batch_size"],
        k_top: int = 10,
        n_rows: int = 2,
        dpi: int = 150,
        **kwargs,
    ):
        """
        Plot tetris boards of top K samples.

        Parameters
        ----------
        samples : list
            List of terminating states sampled from the policy.
        rewards : list
            Rewards of the samples.
        k_top : int
            The number of samples that will be included in the plot. The k_top samples
            with the highest reward are selected.
        n_rows : int
            Number of rows in the plot. The number of columns will be calculated
            according the n_rows and k_top.
        dpi : int
            DPI (dots per inch) of the figure, to determine the resolution.
        """
        # Init figure
        n_cols = np.ceil(k_top / n_rows).astype(int)
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, dpi=dpi)
        # Select top-k samples and plot them
        rewards_topk, indices_topk = torch.sort(rewards, descending=True)[:k_top]
        indices_topk = indices_topk.tolist()
        for idx, ax in zip(indices_topk, axes.flatten()):
            self._plot_board(samples[idx], ax)
        fig.tight_layout()
        return fig

    @staticmethod
    def _plot_board(board, ax: Axes, cellsize: int = 20, linewidth: int = 2):
        """
        Plots a single Tetris board (a state).

        Parameters
        ----------
        board : tensor
            State to plot.
        ax : matplotlib Axes object
            A matplotlib Axes object on which the board will be plotted.
        cellsize : int
           The size (length) of each board cell, in pixels.
        linewidth : int
            The width of the separation between cells, in pixels.
        """
        board = board.clone().numpy()
        height = board.shape[0] * cellsize
        width = board.shape[1] * cellsize
        board_img = 128 * np.ones(
            (height + linewidth, width + linewidth, 3), dtype=np.uint8
        )
        for row in range(board.shape[0]):
            for col in range(board.shape[1]):
                row_init = row * cellsize + linewidth
                row_end = row_init + cellsize - linewidth
                col_init = col * cellsize + linewidth
                col_end = col_init + cellsize - linewidth
                color_key = int(board[row, col] / 100)
                board_img[row_init:row_end, col_init:col_end, :] = PIECES_COLORS[
                    color_key
                ]
        ax.imshow(board_img)
        ax.set_axis_off()
        
        
        
BASE.PY:
    """
Base class of GFlowNet environments
"""

import numbers
import random
import uuid
from abc import abstractmethod
from copy import deepcopy
from textwrap import dedent
from typing import List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from torch.distributions import Categorical
from torchtyping import TensorType

from gflownet.utils.common import copy, set_device, set_float_precision, tbool, tfloat

CMAP = mpl.colormaps["cividis"]
"""
Plotting colour map (cividis).
"""


class GFlowNetEnv:
    """
    Base class of GFlowNet environments
    """

    def __init__(
        self,
        device: str = "cpu",
        float_precision: int = 32,
        env_id: Union[int, str] = "env",
        fixed_distr_params: Optional[dict] = None,
        random_distr_params: Optional[dict] = None,
        skip_mask_check: bool = False,
        conditional: bool = False,
        continuous: bool = False,
        **kwargs,
    ):
        # Flag whether env is conditional
        self.conditional = conditional
        # Flag whether env is continuous
        self.continuous = continuous
        # Call reset() to set initial state, done, n_actions
        self.reset()
        # Device
        self.device = set_device(device)
        # Float precision
        self.float = set_float_precision(float_precision)
        # Flag to skip checking if action is valid (computing mask) before step
        self.skip_mask_check = skip_mask_check
        # Log SoftMax function
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        # Action space
        self.action_space = self.get_action_space()
        self.action_space_torch = torch.tensor(
            self.action_space, device=self.device, dtype=self.float
        )
        # Mask dimensionality
        self._mask_dim = self._compute_mask_dim()
        # Max trajectory length
        self._max_traj_length = self._get_max_trajectory_length()
        # Policy outputs
        self.fixed_distr_params = fixed_distr_params
        self.random_distr_params = random_distr_params
        self.fixed_policy_output = self.get_policy_output(self.fixed_distr_params)
        self.random_policy_output = self.get_policy_output(self.random_distr_params)
        self.policy_output_dim = len(self.fixed_policy_output)
        self.policy_input_dim = len(self.state2policy())

    @abstractmethod
    def get_action_space(self):
        """
        Constructs list with all possible actions (excluding end of sequence)
        """
        pass

    @property
    def action_space_dim(self) -> int:
        """
        Returns the dimensionality of the action space (number of actions).

        Returns
        -------
        The number of actions in the action space.
        """
        return len(self.action_space)

    @property
    def mask_dim(self):
        """
        Returns the dimensionality of the masks.

        Returns
        -------
        The dimensionality of the masks.
        """
        return self._mask_dim

    def _compute_mask_dim(self) -> int:
        """
        Calculates the mask dimensionality.

        By default, the mask dimensionality is equal to the dimensionality of the
        action space.

        This method should be overriden in environments where this may not be the case,
        for example continuous environments (ContinuousCube) and meta-environments such
        as Stack and Set.

        Returns
        -------
        int
            The number of elements in the masks.
        """
        return self.action_space_dim

    def _get_max_trajectory_length(self) -> int:
        """
        Returns the maximum trajectory length of the environment, including the EOS
        action.

        While it is not required to override this method because it does return a
        default value of 100, it is recommended to override it to return the correct
        value or an upper bound as tight as possible to  the maximum.

        The maximum trajectory length does not play a critical role but it is used for
        testing purposes. For example, it is used by get_random_states(), and poor
        estimation of the trajectory length could result in stark inefficiency.
        """
        return 100

    @property
    def max_traj_length(self) -> int:
        """
        Returns the maximum trajectory length of the environment, including the EOS
        action.

        Returns
        -------
        The maximum number of steps in a trajectory of the environment.
        """
        return self._max_traj_length

    def action2representative(self, action: Tuple) -> int:
        """
        For continuous or hybrid environments, converts a continuous action into its
        representative in the action space. Discrete actions remain identical, thus
        fully discrete environments do not need to re-implement this method.
        Continuous environments should re-implement this method in order to replace
        continuous actions by their representatives in the action space.
        """
        return action

    def action2index(self, action: Tuple) -> int:
        """
        Returns the index in the action space of the action passed as an argument, or
        its representative if it is a continuous action.

        See: self.action2representative()
        """
        return self.action_space.index(self.action2representative(action))

    def actions2indices(
        self, actions: TensorType["batch_size", "action_dim"]
    ) -> TensorType["batch_size"]:
        """
        Returns the corresponding indices in the action space of the actions in a batch.
        """
        # Expand the action_space tensor: [batch_size, d_actions_space, action_dim]
        action_space = torch.unsqueeze(self.action_space_torch, 0).expand(
            actions.shape[0], -1, -1
        )
        # Expand the actions tensor: [batch_size, d_actions_space, action_dim]
        actions = torch.unsqueeze(actions, 1).expand(-1, self.action_space_dim, -1)
        # Take the indices at the d_actions_space dimension where all the elements in
        # the action_dim dimension are True
        return torch.where(torch.all(actions == action_space, dim=2))[1]

    def _get_state(self, state: Union[List, TensorType["state_dims"]]):
        """
        A helper method for other methods to determine whether state should be taken
        from the arguments or from the instance (self.state): if is None, it is taken
        from the instance.

        Args
        ----
        state : list or tensor or None
            None, or a state in GFlowNet format.

        Returns
        -------
        state : list or tensor
            The argument state, or self.state if state is None.
        """
        if state is None:
            state = copy(self.state)
        return state

    def _get_done(self, done: bool):
        """
        A helper method for other methods to determine whether done should be taken
        from the arguments or from the instance (self.done): if it is None, it is taken
        from the instance.

        Args
        ----
        done : bool or None
            None, or whether the environment is done.

        Returns
        -------
        done: bool
            The argument done, or self.done if done is None.
        """
        if done is None:
            done = self.done
        return done

    def is_source(
        self, state: Optional[Union[List, TensorType["state_dims"]]] = None
    ) -> bool:
        """
        Returns True if the environment's state or the state passed as parameter (if
        not None) is the source state of the environment.

        Parameters
        ----------
        state : list or tensor or None
            None, or a state in environment format.

        Returns
        -------
        bool
            Whether the state is the source state of the environment
        """
        state = self._get_state(state)
        return self.equal(state, self.source)

    def get_mask_invalid_actions_forward(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
    ) -> List:
        """
        Returns a list of length the action space with values:
            - True if the forward action is invalid from the current state.
            - False otherwise.
        For continuous or hybrid environments, this mask corresponds to the discrete
        part of the action space.
        """
        return [False for _ in range(self.action_space_dim)]

    def get_mask_invalid_actions_backward(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
        parents_a: Optional[List] = None,
    ) -> List:
        """
        Returns a list of length the action space with values:
            - True if the backward action is invalid from the current state.
            - False otherwise.
        For continuous or hybrid environments, this mask corresponds to the discrete
        part of the action space.

        The base implementation below should be common to all discrete spaces as it
        relies on get_parents, which is environment-specific and must be implemented.
        Continuous environments will probably need to implement its specific version of
        this method.
        """
        state = self._get_state(state)
        done = self._get_done(done)
        if parents_a is None:
            _, parents_a = self.get_parents(state, done)
        mask = [True for _ in range(self.action_space_dim)]
        for pa in parents_a:
            mask[self.action_space.index(pa)] = False
        return mask

    def get_mask(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
        backward: Optional[bool] = False,
    ) -> List:
        """
        Returns a mask of invalid actions given a state and a done value. Depending on
        backward, either the forward or the backward mask is returned, by calling the
        corresponding method.
        """
        if backward:
            return self.get_mask_invalid_actions_backward(state, done)
        else:
            return self.get_mask_invalid_actions_forward(state, done)

    def get_valid_actions(
        self,
        mask: Optional[bool] = None,
        state: Optional[List] = None,
        done: Optional[bool] = None,
        backward: Optional[bool] = False,
    ) -> List[Tuple]:
        """
        Returns the list of non-invalid (valid, for short) according to the mask of
        invalid actions.

        More documentation about the meaning and use of invalid actions can be found in
        gflownet/envs/README.md.
        """
        if mask is None:
            mask = self.get_mask(state, done, backward)
        return [action for action, m in zip(self.action_space, mask) if not m]

    def get_parents(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
        action: Optional[Tuple] = None,
    ) -> Tuple[List, List]:
        """
        Determines all parents and actions that lead to state.

        In continuous environments, get_parents() should return only the parent from
        which action leads to state.

        Args
        ----
        state : list
            Representation of a state

        done : bool
            Whether the trajectory is done. If None, done is taken from instance.

        action : tuple
            Last action performed

        Returns
        -------
        parents : list
            List of parents in state format

        actions : list
            List of actions that lead to state for each parent in parents
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [state], [(self.eos,)]
        parents = []
        actions = []
        return parents, actions

    # TODO: consider returning only do_step
    def _pre_step(
        self, action: Tuple[int], backward: bool = False, skip_mask_check: bool = False
    ) -> Tuple[bool, List[int], Tuple[int]]:
        """
        Performs generic checks shared by the step() and step_backward() (backward must
        be True) methods of all environments.

        Args
        ----
        action : tuple
            Action from the action space.

        skip_mask_check : bool
            If True, skip computing forward mask of invalid actions to check if the
            action is valid.

        Returns
        -------
        do_step : bool
            If True, step() should continue further, False otherwise.

        self.state : list
            The sequence after executing the action

        action : int
            Action index
        """
        # If action not found in action space raise an error
        if action not in self.action_space:
            raise ValueError(
                f"Tried to execute action {action} not present in action space."
            )
        # If backward and state is source, step should not proceed.
        if backward is True:
            if self.equal(self.state, self.source) and action != self.eos:
                return False, self.state, action
        # If forward and env is done, step should not proceed.
        else:
            if self.done:
                return False, self.state, action
        # If action is in invalid mask (not in valid actions), step should not proceed.
        if not (self.skip_mask_check or skip_mask_check):
            if action not in self.get_valid_actions(backward=backward):
                return False, self.state, action
        return True, self.state, action

    @abstractmethod
    def step(
        self, action: Tuple[int], skip_mask_check: bool = False
    ) -> Tuple[List[int], Tuple[int], bool]:
        """
        Executes step given an action.

        Args
        ----
        action : tuple
            Action from the action space.

        skip_mask_check : bool
            If True, skip computing forward mask of invalid actions to check if the
            action is valid.

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : int
            Action index

        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """
        _, self.state, action = self._pre_step(action, skip_mask_check)
        return None, None, None

    def step_backwards(
        self, action: Tuple[int], skip_mask_check: bool = False
    ) -> Tuple[List[int], Tuple[int], bool]:
        """
        Executes a backward step given an action. This generic implementation should
        work for all discrete environments, as it relies on get_parents(). Continuous
        environments should re-implement a custom step_backwards(). Despite being valid
        for any discrete environment, the call to get_parents() may be expensive. Thus,
        it may be advantageous to re-implement step_backwards() in a more efficient
        way as well for discrete environments. Especially, because this generic
        implementation will make two calls to get_parents - once here and one in
        _pre_step() through the call to get_mask_invalid_actions_backward() if
        skip_mask_check is True.

        Args
        ----
        action : tuple
            Action from the action space.

        skip_mask_check : bool
            If True, skip computing forward mask of invalid actions to check if the
            action is valid.

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : int
            Action index

        valid : bool
            False, if the action is not allowed for the current state.
        """
        do_step, self.state, action = self._pre_step(action, True, skip_mask_check)
        if not do_step:
            return self.state, action, False
        parents, parents_a = self.get_parents()
        state_next = parents[parents_a.index(action)]
        self.set_state(state_next, done=False)
        self.n_actions += 1
        return self.state, action, True

    # TODO: do not apply temperature here but before calling this method.
    # TODO: rethink whether sampling_method should be here.
    def sample_actions_batch(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        mask: Optional[TensorType["n_states", "policy_output_dim"]] = None,
        states_from: Optional[List] = None,
        is_backward: Optional[bool] = False,
        sampling_method: Optional[str] = "policy",
        temperature_logits: Optional[float] = 1.0,
        max_sampling_attempts: Optional[int] = 10,
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        """
        Samples a batch of actions from a batch of policy outputs.

        This implementation is generally valid for all discrete environments but
        continuous or mixed environments need to reimplement this method.

        The method is valid for both forward and backward actions in the case of
        discrete environments. Some continuous environments may also be agnostic to the
        difference between forward and backward actions since the necessary information
        can be contained in the mask. However, some continuous environments do need to
        know whether the actions are forward of backward, which is why this can be
        specified by the argument is_backward.

        Most environments do not need to know the states from which the actions are to
        be sampled since the necessary information is in both the policy outputs and
        the mask. However, some continuous environments do need to know the originating
        states in order to construct the actions, which is why one of the arguments is
        states_from.

        Args
        ----
        policy_outputs : tensor
            The output of the GFlowNet policy model.

        mask : tensor
            The mask of invalid actions. For continuous or mixed environments, the mask
            may be tensor with an arbitrary length contaning information about special
            states, as defined elsewhere in the environment.

        states_from : tensor
            The states originating the actions, in GFlowNet format. Ignored in discrete
            environments and only required in certain continuous environments.

        is_backward : bool
            True if the actions are backward, False if the actions are forward
            (default). Ignored in discrete environments and only required in certain
            continuous environments.

        max_sampling_attempts : int
            Maximum of number of attempts to sample actions that are not invalid
            according to the mask before throwing an error, in order to ensure that
            non-invalid actions are returned without getting stuck.
        """
        device = policy_outputs.device
        ns_range = torch.arange(policy_outputs.shape[0], device=device)
        if sampling_method == "uniform":
            logits = torch.ones(policy_outputs.shape, dtype=self.float, device=device)
        elif sampling_method == "policy":
            logits = policy_outputs.clone().detach()
            logits /= temperature_logits
        else:
            raise NotImplementedError(
                f"Sampling method {sampling_method} is invalid. "
                "Options are: policy, uniform."
            )

        if mask is not None:
            assert not torch.all(mask, dim=1).any(), dedent(
                """
            All actions in the mask are invalid for some states in the batch.
            """
            )
            logits[mask] = -torch.inf
        else:
            mask = torch.zeros(policy_outputs.shape, dtype=torch.bool, device=device)
        # Make sure that a valid action is sampled, otherwise throw an error.
        for _ in range(max_sampling_attempts):
            action_indices = Categorical(logits=logits).sample()
            if not torch.any(mask[ns_range, action_indices]):
                break
        else:
            raise ValueError(
                dedent(
                    f"""
            No valid action could be sampled after {max_sampling_attempts} attempts.
            """
                )
            )
        logprobs = self.logsoftmax(logits)[ns_range, action_indices]
        # Build actions
        actions = [self.action_space[idx] for idx in action_indices]
        return actions, logprobs

    def get_logprobs(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        actions: TensorType["n_states", "actions_dim"],
        mask: TensorType["batch_size", "policy_output_dim"] = None,
        states_from: Optional[List] = None,
        is_backward: bool = False,
    ) -> TensorType["batch_size"]:
        """
        Computes log probabilities of actions given policy outputs and actions. This
        implementation is generally valid for all discrete environments but continuous
        environments will likely have to implement its own.

        Args
        ----
        policy_outputs : tensor
            The output of the GFlowNet policy model.

        mask : tensor
            The mask of invalid actions. For continuous or mixed environments, the mask
            may be tensor with an arbitrary length contaning information about special
            states, as defined elsewhere in the environment.

        actions : tensor
            The actions from each state in the batch for which to compute the log
            probability.

        states_from : tensor
            The states originating the actions, in GFlowNet format. Ignored in discrete
            environments and only required in certain continuous environments.

        is_backward : bool
            True if the actions are backward, False if the actions are forward
            (default). Ignored in discrete environments and only required in certain
            continuous environments.
        """
        device = policy_outputs.device
        ns_range = torch.arange(policy_outputs.shape[0]).to(device)
        logits = policy_outputs.clone()
        if mask is not None:
            logits[mask] = -torch.inf
        action_indices = (
            torch.tensor(
                [self.action_space.index(tuple(action.tolist())) for action in actions]
            )
            .to(int)
            .to(device)
        )
        logprobs = self.logsoftmax(logits)[ns_range, action_indices]
        return logprobs

    # TODO: add seed
    def step_random(self, backward: bool = False):
        """
        Samples a random action and executes the step.

        Parameters
        ----------
        backward : bool
            If True, the step is performed backwards. False by default.

        Returns
        -------
        state : list
            The state after executing the action.

        action : int
            Action, randomly sampled.

        valid : bool
            False, if the action is not allowed for the current state.
        """
        if backward:
            mask_invalid = torch.unsqueeze(
                tbool(self.get_mask_invalid_actions_backward(), device=self.device), 0
            )
        else:
            mask_invalid = torch.unsqueeze(
                tbool(self.get_mask_invalid_actions_forward(), device=self.device), 0
            )
        random_policy = torch.unsqueeze(
            tfloat(
                self.random_policy_output, float_type=self.float, device=self.device
            ),
            0,
        )
        actions, _ = self.sample_actions_batch(
            random_policy,
            mask_invalid,
            [self.state],
            backward,
        )
        action = actions[0]
        if backward:
            return self.step_backwards(action)
        return self.step(action)

    def trajectory_random(self, backward: bool = False):
        """
        Samples and applies a random trajectory on the environment, by sampling random
        actions until an EOS action is sampled.

        Parameters
        ----------
        backward : bool
            If True, the trajectory is sampled backwards. False by default.

        Returns
        -------
        state : list
            The final state.

        action: list
            The list of actions (tuples) in the trajectory.
        """
        actions = []
        while True:
            _, action, valid = self.step_random(backward)
            if valid:
                actions.append(action)
            if backward and self.is_source():
                break
            elif self.done:
                break
            else:
                continue
        return self.state, actions

    def get_random_terminating_states(
        self, n_states: int, unique: bool = True, max_attempts: int = 100000
    ) -> List:
        """
        Samples n terminating states by using the random policy of the environment
        (calling self.trajectory_random()).

        Note that this method is general for all environments but it may be suboptimal
        in terms of efficiency. In particular, 1) it samples full trajectories in order
        to get terminating states, 2) if unique is True, it needs to compare each newly
        sampled state with all the previously sampled states. If
        get_uniform_terminating_states is available, it may be preferred, or for some
        environments, a custom get_random_terminating_states may be straightforward to
        implement in a much more efficient way.

        Args
        ----
        n_states : int
            The number of terminating states to sample.

        unique : bool
            Whether samples should be unique. True by default.

        max_attempts : int
            The maximum number of attempts, to prevent the method from getting stuck
            trying to obtain n_states different samples if unique is True. 100000 by
            default, therefore if more than 100000 are requested, max_attempts should
            be increased accordingly.

        Returns
        -------
        states : list
            A list of randomly sampled terminating states.
        """
        if unique is False:
            max_attempts = n_states + 1
        states = []
        count = 0
        while len(states) < n_states and count < max_attempts:
            add = True
            self.reset()
            state, _ = self.trajectory_random()
            if unique is True:
                if any([self.equal(state, s) for s in states]):
                    add = False
            if add is True:
                states.append(state)
            count += 1
        return states

    def get_random_states(
        self,
        n_states: int,
        unique: bool = True,
        exclude_source: bool = False,
        max_attempts: int = 1000,
    ) -> List:
        """
        Samples n states (not necessarily terminating) by using the random policy of
        the environment (calling self.step_random()).

        It relies on self.max_traj_length in order to uniformly sample the number
        of steps, in order to obtain states with varying trajectory lengths.

        The method iteratively samples first a trajectory length and attempts to
        perform as many steps. If the trajectory ends before the requested number of
        steps is reached, then it is discarded and a new one is attempted.

        This may introduced a bias towards states that can be reached with a few steps.

        Note that this method is general for all environments but it may be suboptimal
        in terms of efficiency. In particular, 1) it samples trajectories step by step
        in order to get random states, 2) if unique is True, it needs to compare each
        newly sampled state with all the previously sampled states, 3) states are
        copied before adding them to the list, 4) only the last state of a trajectory
        is added to the list in order to have diversity of trajectories.

        Parameters
        ----------
        n_states : int
            The number of terminating states to sample.
        unique : bool
            Whether samples should be unique. True by default.
        max_attempts : int
            The maximum number of attempts, to prevent the method from getting stuck
            trying to obtain n_states different samples if unique is True. 100000 by
            default, therefore if more than 100000 are requested, max_attempts should
            be increased accordingly.
        exclude_source : bool
            If True, exclude the source state from the list of states.

        Returns
        -------
        states : list
            A list of randomly sampled states.

        Raises
        ------
        ValueError
            If max_attempts is smaller than n_states
        RuntimeError
            If the maximum number of attempts is reached before obtaining the requested
            number of unique states.
        """
        max_traj_length = self.max_traj_length
        if max_attempts < n_states:
            raise ValueError(
                f"max_attempts (received {max_attempts})  must larger than or "
                f"equal to n_states (received {n_states})."
            )
        states = []
        n_attempts = 0
        # Iterate until the requested number of states is obtained
        while len(states) < n_states:
            n_attempts += 1
            # Sample a trajectory length for this state
            traj_length = random.randint(1, max_traj_length)
            self.reset()
            is_valid = True
            for _ in range(traj_length):
                # If the trajectory has reached done before the number of requested
                # steps, discard it and start a new one.
                if self.done:
                    is_valid = False
                    break
                # Perform a random step
                self.step_random()

            # If exclude_source is True and the state is the source, mark the
            # trajectory as invalid.
            if is_valid and exclude_source and self.is_source(self.state):
                is_valid = False
            # If unique is True and the state is in the list, mark the trajetory as
            # invalid
            if is_valid and unique and any([self.equal(self.state, s) for s in states]):
                is_valid = False
            # If the trajectory is valid, add the state to the list
            if is_valid:
                states.append(copy(self.state))

            # Check if the number of attempts has reached the maximum
            if n_attempts >= max_attempts:
                raise RuntimeError(
                    f"Reached the maximum number of attempts ({max_attempts}) to "
                    f"sample {n_states} states but only {len(states)} could "
                    "be obtained. It is possible that the state space is too small "
                    f"to contain {n_states} states. Otherwise, consider "
                    "increasing the number of attempts"
                )

        return states

    def get_policy_output(
        self, params: Optional[dict] = None
    ) -> TensorType["policy_output_dim"]:
        """
        Defines the structure of the output of the policy model, from which an
        action is to be determined or sampled, by returning a vector with a fixed
        random policy. As a baseline, the policy is uniform over the dimensionality of
        the action space.

        Continuous environments will generally have to overwrite this method.
        """
        return torch.ones(self.action_space_dim, dtype=self.float, device=self.device)

    def states2proxy(
        self, states: Union[List[List], TensorType["batch", "state_dim"]]
    ) -> TensorType["batch", "state_proxy_dim"]:
        """
        Prepares a batch of states in "environment format" for the proxy. By default,
        the batch of states is converted into a tensor with float dtype and returned as
        is.

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            single tensor.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        return tfloat(states, device=self.device, float_type=self.float)

    def state2proxy(
        self, state: Union[List, TensorType["state_dim"]] = None
    ) -> TensorType["state_proxy_dim"]:
        """
        Prepares a single state in "GFlowNet format" for the proxy. By default, simply
        states2proxy is called and the output will be a "batch" with a single state in
        the proxy format.

        Args
        ----
        state : list
            A state
        """
        state = self._get_state(state)
        return self.states2proxy([state])

    def states2policy(
        self, states: Union[List, TensorType["batch", "state_dim"]]
    ) -> TensorType["batch", "policy_input_dim"]:
        """
        Prepares a batch of states in "environment format" for the policy model: By
        default, the batch of states is converted into a tensor with float dtype and
        returned as is.

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            single tensor.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        return tfloat(states, device=self.device, float_type=self.float)

    def state2policy(
        self, state: Union[List, TensorType["state_dim"]] = None
    ) -> TensorType["policy_input_dim"]:
        """
        Prepares a state in "GFlowNet format" for the policy model. By default,
        states2policy is called, which by default will return the state as is.

        Args
        ----
        state : list
            A state
        """
        state = self._get_state(state)
        return torch.squeeze(self.states2policy([state]), dim=0)

    def state2readable(self, state=None):
        """
        Converts a state into human-readable representation.
        """
        if state is None:
            state = self.state
        return str(state)

    def readable2state(self, readable):
        """
        Converts a human-readable representation of a state into the standard format.
        """
        return readable

    def traj2readable(self, traj=None):
        """
        Converts a trajectory into a human-readable string.
        """
        return str(traj).replace("(", "[").replace(")", "]").replace(",", "")

    def reset(self, env_id: Union[int, str] = None):
        """
        Resets the environment.

        Args
        ----
        env_id: int or str
            Unique (ideally) identifier of the environment instance, used to identify
            the trajectory generated with this environment. If None, uuid.uuid4() is
            used.

        Returns
        -------
        self
        """
        self.state = copy(self.source)
        self.n_actions = 0
        self.done = False
        if env_id is None:
            self.id = str(uuid.uuid4())
        else:
            self.id = env_id
        return self

    def set_id(self, env_id: Union[int, str]):
        """
        Sets the id given as argument and returns the environment.

        Args
        ----
        env_id: int or str
            Unique (ideally) identifier of the environment instance, used to identify
            the trajectory generated with this environment.

        Returns
        -------
        self
        """
        self.id = env_id
        return self

    def set_state(self, state: List, done: Optional[bool] = False):
        """
        Sets the state and done of an environment. Environments that cannot be "done"
        at all states (intermediate states are not fully constructed objects) should
        overwrite this method and check for validity.
        """
        self.state = copy(state)
        self.done = done
        return self

    def copy(self):
        # return self.__class__(**self.__dict__)
        return deepcopy(self)

    @staticmethod
    def equal(state_x, state_y):
        if isinstance(state_x, numbers.Number) or isinstance(state_x, str):
            return state_x == state_y
        if type(state_x) != type(state_y):
            return False
        if torch.is_tensor(state_x) and torch.is_tensor(state_y):
            # Check for nans because (torch.nan == torch.nan) == False
            x_nan = torch.isnan(state_x)
            if torch.any(x_nan):
                y_nan = torch.isnan(state_y)
                if not torch.equal(x_nan, y_nan):
                    return False
                return torch.equal(state_x[~x_nan], state_y[~y_nan])
            return torch.equal(state_x, state_y)
        if isinstance(state_x, dict) and isinstance(state_y, dict):
            if len(state_x) != len(state_y):
                return False
            return all(
                [
                    key_x == key_y and GFlowNetEnv.equal(value_x, value_y)
                    for (key_x, value_x), (key_y, value_y) in zip(
                        sorted(state_x.items()), sorted(state_y.items())
                    )
                ]
            )
        if (isinstance(state_x, list) and isinstance(state_y, list)) or (
            isinstance(state_x, tuple) and isinstance(state_y, tuple)
        ):
            if len(state_x) != len(state_y):
                return False
            if len(state_x) == 0:
                return True
            if isinstance(state_x[0], numbers.Number) or isinstance(state_x[0], str):
                value_type = type(state_x[0])
                if all([isinstance(sx, value_type) for sx in state_x]) and all(
                    [isinstance(sy, value_type) for sy in state_y]
                ):
                    return state_x == state_y
        return all([GFlowNetEnv.equal(sx, sy) for sx, sy in zip(state_x, state_y)])

    @staticmethod
    def isclose(state_x, state_y, atol=1e-8):
        if isinstance(state_x, numbers.Number) or isinstance(state_x, str):
            return np.isclose(state_x, state_y, atol=atol)
        if type(state_x) != type(state_y):
            return False
        if torch.is_tensor(state_x) and torch.is_tensor(state_y):
            # Check for nans because (torch.nan == torch.nan) == False
            x_nan = torch.isnan(state_x)
            if torch.any(x_nan):
                y_nan = torch.isnan(state_y)
                if not torch.equal(x_nan, y_nan):
                    return False
                return torch.all(
                    torch.isclose(state_x[~x_nan], state_y[~y_nan], atol=atol)
                )
            return torch.equal(state_x, state_y)
        if isinstance(state_x, dict) and isinstance(state_y, dict):
            if len(state_x) != len(state_y):
                return False
            return all(
                [
                    key_x == key_y and GFlowNetEnv.isclose(value_x, value_y)
                    for (key_x, value_x), (key_y, value_y) in zip(
                        sorted(state_x.items()), sorted(state_y.items())
                    )
                ]
            )
        if (isinstance(state_x, list) and isinstance(state_y, list)) or (
            isinstance(state_x, tuple) and isinstance(state_y, tuple)
        ):
            if len(state_x) != len(state_y):
                return False
            if len(state_x) == 0:
                return True
            if isinstance(state_x[0], numbers.Number) or isinstance(state_x[0], str):
                value_type = type(state_x[0])
                if all([isinstance(sx, value_type) for sx in state_x]) and all(
                    [isinstance(sy, value_type) for sy in state_y]
                ):
                    return np.all(np.isclose(state_x, state_y, atol=atol))
        return all([GFlowNetEnv.isclose(sx, sy) for sx, sy in zip(state_x, state_y)])

    def get_trajectories(
        self, traj_list, traj_actions_list, current_traj, current_actions
    ):
        """
        Determines all trajectories leading to each state in traj_list, recursively.

        Args
        ----
        traj_list : list
            List of trajectories (lists)

        traj_actions_list : list
            List of actions within each trajectory

        current_traj : list
            Current trajectory

        current_actions : list
            Actions of current trajectory

        Returns
        -------
        traj_list : list
            List of trajectories (lists)

        traj_actions_list : list
            List of actions within each trajectory
        """
        parents, parents_actions = self.get_parents(current_traj[-1], False)
        if parents == []:
            traj_list.append(current_traj)
            traj_actions_list.append(current_actions)
            return traj_list, traj_actions_list
        for idx, (p, a) in enumerate(zip(parents, parents_actions)):
            traj_list, traj_actions_list = self.get_trajectories(
                traj_list, traj_actions_list, current_traj + [p], current_actions + [a]
            )
        return traj_list, traj_actions_list

    @torch.no_grad()
    def compute_train_energy_proxy_and_rewards(self):
        """
        Gather batched proxy data:

        * The ground-truth energy of the train set
        * The predicted proxy energy over the train set
        * The reward version of those energies (with env.proxy2reward)

        Returns
        -------
        gt_energy : torch.Tensor
            The ground-truth energies in the proxy's train set

        proxy_energy : torch.Tensor
            The proxy's predicted energies over its train set

        gt_reward : torch.Tensor
            The reward version of the ground-truth energies

        proxy_reward : torch.Tensor
            The reward version of the proxy's predicted energies
        """
        gt_energy, proxy_energy = self.proxy.infer_on_train_set()
        gt_reward = self.proxy2reward(gt_energy)
        proxy_reward = self.proxy2reward(proxy_energy)

        return gt_energy, proxy_energy, gt_reward, proxy_reward

    def mask_conditioning(
        self, mask: Union[List[bool], TensorType["mask_dim"]], env_cond, backward: bool
    ):
        """
        Conditions the input mask based on the restrictions imposed by a conditioning
        environment, env_cond.

        It is assumed that the state space of the conditioning environment is a subset
        of the state space of the original environment (self). The conditioning
        mechanism goes as follows: given a state, its corresponding mask and a
        conditioning environment, the mask of invalid actions is updated such that all
        actions that would be invalid in the conditioning environment are made invalid,
        even though they may not be invalid in the original environment.
        """
        # Set state in conditional environment
        env_cond.reset()
        env_cond.set_state(self.state, self.done)
        # If the environment is continuous, then we simply return the mask of the
        # conditioning environment. It is thus assumed that the dimensionality and
        # interpretation is the same.
        if self.continuous:
            return env_cond.get_mask(backward=backward)
        # Get valid actions common to both the original and the conditioning env
        actions_valid_orig = self.get_valid_actions(mask)
        actions_valid_cond = env_cond.get_valid_actions(backward=backward)
        actions_valid = set(actions_valid_orig).intersection(set(actions_valid_cond))
        # Construct new mask by setting to False (valid or not invalid) the actions
        # that are valid to both environments
        mask = [True] * self.mask_dim
        for action in actions_valid:
            mask[self.action_space.index(action)] = False
        return mask

    @torch.no_grad()
    def top_k_metrics_and_plots(
        self,
        states,
        top_k,
        name,
        energy=None,
        reward=None,
        step=None,
        **kwargs,
    ):
        """
        Compute top_k metrics and plots for the given states.

        In particular, if no states, energy, or reward are passed, then the name
        *must* be "train", and the energy and reward will be computed from the
        proxy using ``env.compute_train_energy_proxy_and_rewards()``. In this case,
        ``top_k_metrics_and_plots`` will be called a second time to compute the
        metrics and plots of the proxy distribution in addition to the ground-truth
        distribution.
        Train mode should only be called once at the begining of training as
        distributions do not change over time.

        If ``states`` are passed, then the energy and reward will be computed from the
        proxy for those states. They are typically sampled from the current GFN.

        Otherwise, energy and reward should be passed directly.

        *Plots and metrics*:
        - mean+std of energy and reward
        - mean+std of top_k energy and reward
        - histogram of energy and reward
        - histogram of top_k energy and reward


        Args
        ----
        states: list
            List of states to compute metrics and plots for.

        top_k: int
            Number of top k states to compute metrics and plots for.
            "top" means lowest energy/highest reward.

        name: str
            Name of the distribution to compute metrics and plots for.
            Typically "gflownet", "random" or "train". Will be used in
            metrics names like ``f"Mean {name} energy"``.

        energy: torch.Tensor, optional
            Batch of pre-computed energies

        reward: torch.Tensor, optional
            Batch of pre-computed rewards

        step: int, optional
            Step number to use for the plot title.

        Returns
        -------
        metrics: dict
            Dictionary of metrics: str->float

        figs: list
            List of matplotlib figures

        figs_names: list
            List of figure names for ``figs``
        """

        if states is None and energy is None and reward is None:
            assert name == "train"
            (
                energy,
                proxy,
                energy_reward,
                proxy_reward,
            ) = self.compute_train_energy_proxy_and_rewards()
            name = "train ground truth"
            reward = energy_reward
        elif energy is None and reward is None:
            # TODO: fix this
            x = torch.stack([self.state2proxy(s) for s in states])
            energy = self.proxy(x.to(self.device)).cpu()
            reward = self.proxy2reward(energy)

        assert energy is not None and reward is not None

        # select top k best energies and rewards
        top_k_e = torch.topk(energy, top_k, largest=False, dim=0).values.numpy()
        top_k_r = torch.topk(reward, top_k, largest=True, dim=0).values.numpy()

        # find best energy and reward
        best_e = torch.min(energy).item()
        best_r = torch.max(reward).item()

        # to numpy to plot
        energy = energy.numpy()
        reward = reward.numpy()

        # compute stats
        mean_e = np.mean(energy)
        mean_r = np.mean(reward)

        std_e = np.std(energy)
        std_r = np.std(reward)

        mean_top_k_e = np.mean(top_k_e)
        mean_top_k_r = np.mean(top_k_r)

        std_top_k_e = np.std(top_k_e)
        std_top_k_r = np.std(top_k_r)

        # automatic color scale
        # currently: cividis colour map
        colors = ["full", "top_k"]
        normalizer = mpl.colors.Normalize(vmin=0, vmax=len(colors) - 0.5)
        colors = {k: CMAP(normalizer(i)) for i, k in enumerate(colors[::-1])}

        # two sublopts: left is energy, right is reward
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # energy full distribution and stats lines
        ax[0].hist(
            energy,
            bins=100,
            alpha=0.35,
            label=f"All = {len(energy)}",
            color=colors["full"],
            density=True,
        )
        ax[0].axvline(
            mean_e,
            color=colors["full"],
            linestyle=(0, (5, 10)),
            label=f"Mean = {mean_e:.3f}",
        )
        ax[0].axvline(
            mean_e + std_e,
            color=colors["full"],
            linestyle=(0, (1, 10)),
            label=f"Std = {std_e:.3f}",
        )
        ax[0].axvline(
            mean_e - std_e,
            color=colors["full"],
            linestyle=(0, (1, 10)),
        )

        # energy top k distribution and stats lines
        ax[0].hist(
            top_k_e,
            bins=100,
            alpha=0.7,
            label=f"Top k = {top_k}",
            color=colors["top_k"],
            density=True,
        )
        ax[0].axvline(
            mean_top_k_e,
            color=colors["top_k"],
            linestyle=(0, (5, 10)),
            label=f"Mean = {mean_top_k_e:.3f}",
        )
        ax[0].axvline(
            mean_top_k_e + std_top_k_e,
            color=colors["top_k"],
            linestyle=(0, (1, 10)),
            label=f"Std = {std_top_k_e:.3f}",
        )
        ax[0].axvline(
            mean_top_k_e - std_top_k_e,
            color=colors["top_k"],
            linestyle=(0, (1, 10)),
        )
        # energy title & legend
        ax[0].set_title(
            f"Energy distribution for {top_k} vs {len(energy)}"
            + f" samples\nBest: {best_e:.3f}",
            y=0,
            pad=-20,
            verticalalignment="top",
            size=12,
        )
        ax[0].legend()

        # reward full distribution and stats lines
        ax[1].hist(
            reward,
            bins=100,
            alpha=0.35,
            label=f"All = {len(reward)}",
            color=colors["full"],
            density=True,
        )
        ax[1].axvline(
            mean_r,
            color=colors["full"],
            linestyle=(0, (5, 10)),
            label=f"Mean = {mean_r:.3f}",
        )
        ax[1].axvline(
            mean_r + std_r,
            color=colors["full"],
            linestyle=(0, (1, 10)),
            label=f"Std = {std_r:.3f}",
        )
        ax[1].axvline(
            mean_r - std_r,
            color=colors["full"],
            linestyle=(0, (1, 10)),
        )

        # reward top k distribution and stats lines
        ax[1].hist(
            top_k_r,
            bins=100,
            alpha=0.7,
            label=f"Top k = {top_k}",
            color=colors["top_k"],
            density=True,
        )
        ax[1].axvline(
            mean_top_k_r,
            color=colors["top_k"],
            linestyle=(0, (5, 10)),
            label=f"Mean = {mean_top_k_r:.3f}",
        )
        ax[1].axvline(
            mean_top_k_r + std_top_k_r,
            color=colors["top_k"],
            linestyle=(0, (1, 10)),
            label=f"Std = {std_top_k_r:.3f}",
        )
        ax[1].axvline(
            mean_top_k_r - std_top_k_r,
            color=colors["top_k"],
            linestyle=(0, (1, 10)),
        )

        # reward title & legend
        ax[1].set_title(
            f"Reward distribution for {top_k} vs {len(reward)}"
            + f" samples\nBest: {best_r:.3f}",
            y=0,
            pad=-20,
            verticalalignment="top",
            size=12,
        )
        ax[1].legend()

        # Finalize figure
        title = f"{name.capitalize()} energy and reward distributions"
        if step is not None:
            title += f" (step {step})"
        fig.suptitle(title, y=0.95)
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])

        # store metrics
        metrics = {
            f"Mean {name} energy": mean_e,
            f"Std {name} energy": std_e,
            f"Mean {name} reward": mean_r,
            f"Std {name} reward": std_r,
            f"Mean {name} top k energy": mean_top_k_e,
            f"Std {name} top k energy": std_top_k_e,
            f"Mean {name} top k reward": mean_top_k_r,
            f"Std {name} top k reward": std_top_k_r,
            f"Best (min) {name} energy": best_e,
            f"Best (max) {name} reward": best_r,
        }
        figs = [fig]
        fig_names = [title]

        if name.lower() == "train ground truth":
            # train stats mode: the ground truth data has meen plotted
            # and computed, let's do it again for the proxy data.
            # This can be used to visualize potential distribution mismatch
            # between the proxy and the ground truth data.
            proxy_metrics, proxy_figs, proxy_fig_names = self.top_k_metrics_and_plots(
                None,
                top_k,
                "train proxy",
                energy=proxy,
                reward=proxy_reward,
                step=None,
                **kwargs,
            )
            # aggregate metrics and figures
            metrics.update(proxy_metrics)
            figs += proxy_figs
            fig_names += proxy_fig_names

        return metrics, figs, fig_names

    def plot_reward_distribution(
        self, states=None, scores=None, ax=None, title=None, proxy=None, **kwargs
    ):
        if ax is None:
            fig, ax = plt.subplots()
            standalone = True
        else:
            standalone = False
        if title == None:
            title = "Scores of Sampled States"
        if proxy is None:
            proxy = self.proxy
        if scores is None:
            if isinstance(states[0], torch.Tensor):
                states = torch.vstack(states).to(self.device, self.float)
            if isinstance(states, torch.Tensor) == False:
                states = torch.tensor(states, device=self.device, dtype=self.float)
            states_proxy = self.states2proxy(states)
            scores = self.proxy(states_proxy)
        if isinstance(scores, TensorType):
            scores = scores.cpu().detach().numpy()
        ax.hist(scores)
        ax.set_title(title)
        ax.set_ylabel("Number of Samples")
        ax.set_xlabel("Energy")
        plt.show()
        if standalone == True:
            plt.tight_layout()
            plt.close()
        return ax

    def test(
        self,
        samples: Union[
            TensorType["n_trajectories", "..."], npt.NDArray[np.float32], List
        ],
    ) -> dict:
        """
        Placeholder for a custom test function that can be defined for a specific
        environment. Can be overwritten if special evaluation procedure is needed
        for a given environment.

        Args
        ----
        samples
            A collection of sampled terminating states.

        Returns
        -------
        metrics
            A dictionary with metrics and their calculated values.
        """
        return {}
    
    
    Common.py:
        import os
import random
from copy import deepcopy
from functools import partial
from os.path import expandvars
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf
from torchtyping import TensorType

from gflownet.utils.policy import parse_policy_config


def set_device(device: Union[str, torch.device]):
    """
    Get `torch` device from device.

    Examples
    --------
    >>> set_device("cuda")
    device(type='cuda')

    >>> set_device("cpu")
    device(type='cpu')

    >>> set_device(torch.device("cuda"))
    device(type='cuda')

    Parameters
    ----------
    device : Union[str, torch.device]
        Device.

    Returns
    -------
    torch.device
        `torch` device.
    """
    if isinstance(device, torch.device):
        return device
    if device.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def set_float_precision(precision: Union[int, torch.dtype]):
    """
    Get `torch` float type from precision.

    Examples
    --------
    >>> set_float_precision(32)
    torch.float32

    >>> set_float_precision(torch.float32)
    torch.float32

    Parameters
    ----------
    precision : Union[int, torch.dtype]
        Precision.

    Returns
    -------
    torch.dtype
        `torch` float type.

    Raises
    ------
    ValueError
        If precision is not one of [16, 32, 64].
    """
    if isinstance(precision, torch.dtype):
        return precision
    if precision == 16:
        return torch.float16
    elif precision == 32:
        return torch.float32
    elif precision == 64:
        return torch.float64
    else:
        raise ValueError("Precision must be one of [16, 32, 64]")


def set_int_precision(precision: Union[int, torch.dtype]):
    """
    Get `torch` integer type from `int` precision.

    Examples
    --------
    >>> set_int_precision(32)
    torch.int32

    >>> set_int_precision(torch.int32)
    torch.int32

    Parameters
    ----------
    precision : Union[int, torch.dtype]
        Integer precision.

    Returns
    -------
    torch.dtype
        `torch` integer type.

    Raises
    ------
    ValueError
        If precision is not one of [16, 32, 64].
    """
    if isinstance(precision, torch.dtype):
        return precision
    if precision == 16:
        return torch.int16
    elif precision == 32:
        return torch.int32
    elif precision == 64:
        return torch.int64
    else:
        raise ValueError("Precision must be one of [16, 32, 64]")


def torch2np(x):
    """
    Convert a torch tensor to a numpy array.

    Parameters
    ----------
    x : Union[torch.Tensor, np.ndarray, list]
        Data to be converted.

    Returns
    -------
    np.ndarray
        Converted data.
    """
    if hasattr(x, "is_cuda") and x.is_cuda:
        x = x.detach().cpu()
    return np.array(x)


def download_file_if_not_exists(path: str, url: str):
    """
    Download a file from google drive if path doestn't exist.
    url should be in the format: https://drive.google.com/uc?id=FILE_ID
    """
    import gdown

    path = Path(path)
    if not path.is_absolute():
        # to avoid storing downloaded files with the logs, prefix is set to the original working dir
        prefix = get_original_cwd()
        path = Path(prefix) / path
    if not path.exists():
        path.absolute().parent.mkdir(parents=True, exist_ok=True)
        gdown.download(url, str(path.absolute()), quiet=False)
    return path


def resolve_path(path: str) -> Path:
    """
    Resolve a path by expanding environment variables, user home directory, and making
    it absolute.

    Examples
    --------
    >>> resolve_path("~/scratch/$SLURM_JOB_ID/data")
    Path("/home/user/scratch/12345/data")

    Parameters
    ----------
    path : Union[str, Path]
        Path to be resolved.

    Returns
    -------
    Path
        Resolved path.
    """
    return Path(expandvars(str(path))).expanduser().resolve()


def find_latest_checkpoint(ckpt_dir):
    """
    Find the latest checkpoint in the input directory.

    If the directory contains a checkpoint file with the name "final", that checkpoint
    is returned. Otherwise, the latest checkpoint is returned based on the iteration
    number set in the file names.

    Parameters
    ----------
    ckpt_dir : Union[str, Path]
        Directory in which to search for the checkpoints.

    Returns
    -------
    Path
        Path to the latest checkpoint.

    Raises
    ------
    ValueError
        If no checkpoint files are found in the input directory.
    """
    ckpt_dir = Path(ckpt_dir)
    final = [f for f in ckpt_dir.glob(f"*final*")]
    if len(final) > 0:
        return final[0]
    ckpts = [f for f in ckpt_dir.glob(f"iter_*")]
    if not ckpts:
        raise ValueError(
            f"No checkpoints found in {ckpt_dir} with pattern iter_* or *final*"
        )
    return sorted(ckpts, key=lambda f: float(f.stem.split("iter_")[1]))[-1]


def read_hydra_config(rundir=None, config_name="config"):
    if rundir is None:
        rundir = Path(config_name)
        hydra_dir = rundir.parent
        config_name = rundir.name
    else:
        hydra_dir = rundir / ".hydra"

    with initialize_config_dir(
        version_base=None, config_dir=str(hydra_dir), job_name="xxx"
    ):
        return compose(config_name=config_name)


def gflownet_from_config(config, env=None):
    """
    Create GFlowNet from a Hydra OmegaConf config.

    Parameters
    ----------
    config : DictConfig
        Config.

    env : GFlowNetEnv
        Optional environment instance to be used in the initialization.

    Returns
    -------
    GFN
        GFlowNet.
    """
    # Logger
    logger = instantiate(config.logger, config, _recursive_=False)

    # The proxy is required by the GFlowNetAgent for computing rewards
    proxy = instantiate(
        config.proxy,
        device=config.device,
        float_precision=config.float_precision,
    )

    # Using Hydra's partial instantiation, see:
    # https://hydra.cc/docs/advanced/instantiate_objects/overview/#partial-instantiation
    # If env is passed as an argument, we create an env maker with a partial
    # instantiation from the copy method of the environment (this is used in unit
    # tests, for example). Otherwise, we create the env maker with partial
    # instantiation from the config.
    if env is not None:
        env_maker = partial(env.copy)
    else:
        env_maker = instantiate(
            config.env,
            device=config.device,
            float_precision=config.float_precision,
            _partial_=True,
        )
        env = env_maker()

    # TOREVISE: set up proxy so when buffer calls it (when it creates train / test
    # dataset) it has the correct infro from env
    # proxy.setup(env)
    buffer = instantiate(
        config.buffer,
        env=env,
        proxy=proxy,
        datadir=logger.datadir,
    )

    # The evaluator is used to compute metrics and plots
    evaluator = instantiate(config.evaluator)

    # The policy is used to model the probability of a forward/backward action
    forward_config = parse_policy_config(config, kind="forward")
    backward_config = parse_policy_config(config, kind="backward")

    forward_policy = instantiate(
        forward_config,
        env=env,
        device=config.device,
        float_precision=config.float_precision,
    )
    backward_policy = instantiate(
        backward_config,
        env=env,
        device=config.device,
        float_precision=config.float_precision,
        base=forward_policy,
    )

    # State flow
    if config.gflownet.state_flow is not None:
        state_flow = instantiate(
            config.gflownet.state_flow,
            env=env,
            device=config.device,
            float_precision=config.float_precision,
            base=forward_policy,
        )
    else:
        state_flow = None

    # GFlowNet Agent
    gflownet = instantiate(
        config.gflownet,
        device=config.device,
        float_precision=config.float_precision,
        env_maker=env_maker,
        proxy=proxy,
        forward_policy=forward_policy,
        backward_policy=backward_policy,
        state_flow=state_flow,
        buffer=buffer,
        logger=logger,
        evaluator=evaluator,
    )

    return gflownet


def load_gflownet_from_rundir(
    rundir,
    no_wandb=True,
    print_config=False,
    device=None,
    load_last_checkpoint=True,
    is_resumed: bool = False,
):
    """
    Load GFlowNet from a run path (directory with a `.hydra` directory inside).

    Parameters
    ----------
    rundir : Union[str, Path]
        Path to the run directory. Must contain a `.hydra` directory.
    no_wandb : bool, optional
        Whether to disable wandb in the GFN init, by default True.
    print_config : bool, optional
        Whether to print the loaded config, by default False.
    device : str, optional
        Device to which the models should be moved. If None (default), take the device
        from the loaded config.
    load_last_checkpoint : bool, optional
        Whether to load the final models, by default True.
    is_resumed : bool, optional
        Whether the GFlowNet is loaded to resume training.

    Returns
    -------
    Tuple[GFN, DictConfig]
        Loaded GFlowNet and the loaded config.

    Raises
    ------
    ValueError
        If no checkpoints are found in the directory.
    """
    rundir = resolve_path(rundir)

    # Read experiment config
    config = OmegaConf.load(Path(rundir) / ".hydra" / "config.yaml")
    # Resolve variables
    config = OmegaConf.to_container(config, resolve=True)
    # Re-create OmegaCong DictConfig
    config = OmegaConf.create(config)

    if print_config:
        print(OmegaConf.to_yaml(config))

    # Device
    if device is None:
        device = config.device

    if no_wandb:
        # Disable wandb
        config.logger.do.online = False

    # -----------------------------------------
    # -----  Load last model checkpoints  -----
    # -----------------------------------------

    if load_last_checkpoint:
        checkpoint_latest = find_latest_checkpoint(rundir / config.logger.logdir.ckpts)
        checkpoint = torch.load(checkpoint_latest, map_location=set_device(device))

        # Set run id in logger to enable WandB resume
        config.logger.run_id = checkpoint["run_id"]

        # Set up Buffer configuration to load data sets and buffers from run
        if checkpoint["buffer"]["train"]:
            config.buffer.train = {
                "type": "pkl",
                "path": checkpoint["buffer"]["train"],
            }
        if checkpoint["buffer"]["test"]:
            config.buffer.test = {
                "type": "pkl",
                "path": checkpoint["buffer"]["test"],
            }
        if checkpoint["buffer"]["replay"]:
            config.buffer.replay_buffer = checkpoint["buffer"]["replay"]
        # load them here

        if is_resumed:
            config.logger.logdir.root = rundir
            config.logger.is_resumed = True

    # Initialize a GFlowNet agent from the configuration
    gflownet = gflownet_from_config(config)

    # Load checkpoint into the GFlowNet agent
    if load_last_checkpoint:
        gflownet.load_checkpoint(checkpoint)

    return gflownet, config


def batch_with_rest(start, stop, step, tensor=False):
    """
    Yields batches of indices from start to stop with step size. The last batch may be
    smaller than step.

    Parameters
    ----------
    start : int
        Start index
    stop : int
        End index (exclusive)
    step : int
        Step size
    tensor : bool, optional
        Whether to return a `torch` tensor of indices instead of a `numpy` array, by
        default False.

    Yields
    ------
    Union[np.ndarray, torch.Tensor]
        Batch of indices
    """
    for i in range(start, stop, step):
        if tensor:
            yield torch.arange(i, min(i + step, stop))
        else:
            yield np.arange(i, min(i + step, stop))


def tfloat(x, device, float_type):
    """
    Convert input to a float tensor. If the input is a list of tensors, the tensors
    are stacked along the first dimension.

    The resulting tensor is moved to the specified device.

    Parameters
    ----------
    x : Union[List[torch.Tensor], torch.Tensor, List[Union[int, float]], Union[int,
    float]]
        Input to be converted to a float tensor.
    device : torch.device
        Device to which the tensor should be moved.
    float_type : torch.dtype
        Float type to which the tensor should be converted.

    Returns
    -------
    Union[torch.Tensor, List[torch.Tensor]]
        Float tensor.
    """
    if isinstance(x, list) and torch.is_tensor(x[0]):
        return torch.stack(x).to(device=device, dtype=float_type)
    if torch.is_tensor(x):
        return x.to(device=device, dtype=float_type)
    else:
        return torch.tensor(x, dtype=float_type, device=device)


def tlong(x, device):
    """
    Convert input to a long tensor. If the input is a list of tensors, the tensors
    are stacked along the first dimension.

    The resulting tensor is moved to the specified device.

    Parameters
    ----------
    x : Union[List[torch.Tensor], torch.Tensor, List[Union[int, float]], Union[int,
    float]]
        Input to be converted to a long tensor.
    device : torch.device
        Device to which the tensor should be moved.

    Returns
    -------
    Union[torch.Tensor, List[torch.Tensor]]
        Long tensor.
    """
    if isinstance(x, list) and torch.is_tensor(x[0]):
        return torch.stack(x).to(device=device, dtype=torch.long)
    if torch.is_tensor(x):
        return x.to(device=device, dtype=torch.long)
    else:
        return torch.tensor(x, dtype=torch.long, device=device)


def tint(x, device, int_type):
    """
    Convert input to an integer tensor. If the input is a list of tensors, the tensors
    are stacked along the first dimension.

    The resulting tensor is moved to the specified device.

    Parameters
    ----------
    x : Union[List[torch.Tensor], torch.Tensor, List[Union[int, float]], Union[int,
    float]]
        Input to be converted to an integer tensor.
    device : torch.device
        Device to which the tensor should be moved.
    int_type : torch.dtype
        Integer type to which the tensor should be converted.

    Returns
    -------
    Union[torch.Tensor, List[torch.Tensor]]
        Integer tensor.
    """
    if isinstance(x, list) and torch.is_tensor(x[0]):
        return torch.stack(x).to(device=device, dtype=int_type)
    if torch.is_tensor(x):
        return x.to(device=device, dtype=int_type)
    else:
        return torch.tensor(x, dtype=int_type, device=device)


def tbool(x, device):
    """
    Convert input to a boolean tensor. If the input is a list of tensors, the tensors
    are stacked along the first dimension.

    The resulting tensor is moved to the specified device.

    Parameters
    ----------
    x : Union[List[torch.Tensor], torch.Tensor, List[Union[int, float]], Union[int,
    float]]
        Input to be converted to a boolean tensor.
    device : torch.device
        Device to which the tensor should be moved.

    Returns
    -------
    Union[torch.Tensor, List[torch.Tensor]]
        Boolean tensor.
    """
    if isinstance(x, list) and torch.is_tensor(x[0]):
        return torch.stack(x).to(device=device, dtype=torch.bool)
    if torch.is_tensor(x):
        return x.to(device=device, dtype=torch.bool)
    else:
        return torch.tensor(x, dtype=torch.bool, device=device)


def concat_items(list_of_items, indices=None):
    """
    Concatenates a list of items into a single tensor or array.

    Parameters
    ----------
    list_of_items :
        List of items to be concatenated, i.e. list of arrays or list of tensors.
    indices : Union[List[np.ndarray], List[torch.Tensor]], optional
        Indices to select in the resulting concatenated tensor or array, by default
        None.

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        Concatenated tensor or array, with optional selection of indices.

    Raises
    ------
    NotImplementedError
        If the input type is not supported, i.e., not a list of arrays or a list of
        tensors.
    """
    if isinstance(list_of_items[0], np.ndarray):
        result = np.concatenate(list_of_items)
        if indices is not None:
            if torch.is_tensor(indices[0]):
                indices = indices.cpu().numpy()
            result = result[indices]
    elif torch.is_tensor(list_of_items[0]):
        result = torch.cat(list_of_items)
        if indices is not None:
            result = result[indices]
    else:
        raise NotImplementedError(
            "cannot concatenate {}".format(type(list_of_items[0]))
        )

    return result


def extend(
    orig: Union[List, TensorType["..."]], new: Union[List, TensorType["..."]]
) -> Union[List, TensorType["..."]]:
    """
    Extends the original list or tensor with the new list or tensor.

    Returns
    -------
    Union[List, TensorType["..."]]
        Extended list or tensor.

    Raises
    ------
    NotImplementedError
        If the input type is not supported, i.e., not a list or a tensor.
    """
    assert isinstance(orig, type(new))
    if isinstance(orig, list):
        orig.extend(new)
    elif torch.tensor(orig):
        orig = torch.cat([orig, new])
    else:
        raise NotImplementedError(
            "Extension only supported for lists and torch tensors"
        )
    return orig


def copy(x: Union[List, TensorType["..."]]):
    """
    Makes copy of the input tensor or list.

    A tensor is cloned and detached from the computational graph.

    Parameters
    ----------
    x : Union[List, TensorType["..."]]
        Input tensor or list to be copied.

    Returns
    -------
    Union[List, TensorType["..."]]
        Copy of the input tensor or list.
    """
    if torch.is_tensor(x):
        return x.clone().detach()
    else:
        return deepcopy(x)


def bootstrap_samples(tensor, num_samples):
    """
    Bootstraps tensor along the last dimention
    returns tensor of the shape [initial_shape, num_samples]
    """
    dim_size = tensor.size(-1)
    bs_indices = torch.randint(
        0, dim_size, size=(num_samples * dim_size,), device=tensor.device
    )
    bs_samples = torch.index_select(tensor, -1, index=bs_indices)
    bs_samples = bs_samples.view(
        tensor.size()[:-1] + (num_samples, dim_size)
    ).transpose(-1, -2)
    return bs_samples


def example_documented_function(arg1, arg2):
    r"""Summary line: this function is not used anywhere, it's just an example.

    Extended description of function from the docstrings tutorial :ref:`write
    docstrings-extended`.

    Refer to

    * functions with :py:func:`gflownet.utils.common.set_device`
    * classes with :py:class:`gflownet.gflownet.GFlowNetAgent`
    * methods with :py:meth:`gflownet.envs.base.GFlowNetEnv.get_action_space`
    * constants with :py:const:`gflownet.envs.base.CMAP`

    Prepenend with ``~`` to refer to the name of the object only instead of the full
    path -> :py:func:`~gflownet.utils.common.set_device` will display as ``set_device``
    instead of the full path.

    Great maths:

    .. math::

        \int_0^1 x^2 dx = \frac{1}{3}

    .. important::

        A docstring with **math** MUST be a raw Python string (a string prepended with
        an ``r``: ``r"raw"``) to avoid backslashes being treated as escape characters.

        Alternatively, you can use double backslashes.

    .. warning::

        Display a warning. See :ref:`learn by example`. (<-- this is a cross reference,
        learn about it `here
        <https://www.sphinx-doc.org/en/master/usage/referencing.html#ref-rolel>`_)


    Examples
    --------
    >>> function(1, 'a')
    True
    >>> function(1, 2)
    True

    >>> function(1, 1)
    Traceback (most recent call last):
        ...

    Notes
    -----
    This block uses ``$ ... $`` for inline maths -> $e^{\frac{x}{2}}$.

    Or ``$$ ... $$`` for block math instead of the ``.. math:`` directive above.

    $$\int_0^1 x^2 dx = \frac{1}{3}$$


    Parameters
    ----------
    arg1 : int
        Description of arg1
    arg2 : str
        Description of arg2

    Returns
    -------
    bool
        Description of return value
    """
    if arg1 == arg2:
        raise ValueError("arg1 must not be equal to arg2")
    return True