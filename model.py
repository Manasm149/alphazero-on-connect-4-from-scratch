"""
AlphaZero on Connect-4 from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_empty_board
import numpy as np

def make_empty_board():
    return np.zeros((6, 7), dtype=int)

# Step 2 - column_top_row
def column_top_row(board, column):
    for row in range(5, -1, -1):
        if board[row, column] == 0:
            return row

    return -1

# Step 3 - drop_piece
def drop_piece(board, column, player):
    row = column_top_row(board, column)

    if row == -1:
        raise ValueError("Column is full")

    new_board = board.copy()
    new_board[row, column] = player

    return new_board

# Step 4 - column_full
def column_full(board, column):
    return column_top_row(board, column) == -1

# Step 5 - valid_moves
def valid_moves(board):
    return [column for column in range(7)
            if not column_full(board, column)]

# Step 6 - four_in_a_row_horizontal
def four_in_a_row_horizontal(board):
    for row in range(6):
        for col in range(4):
            if (board[row, col] != 0 and
                board[row, col] == board[row, col + 1] and
                board[row, col] == board[row, col + 2] and
                board[row, col] == board[row, col + 3]):
                return int(board[row, col])

    return 0

# Step 7 - four_in_a_row_vertical
def four_in_a_row_vertical(board):
    for col in range(7):
        for row in range(3):
            if (board[row, col] != 0 and
                board[row, col] == board[row + 1, col] and
                board[row, col] == board[row + 2, col] and
                board[row, col] == board[row + 3, col]):
                return int(board[row, col])

    return 0

# Step 8 - four_in_a_row_diagonal_down_right
def four_in_a_row_diagonal_down_right(board):
    for row in range(3):
        for col in range(4):
            if (board[row, col] != 0 and
                board[row, col] == board[row + 1, col + 1] and
                board[row, col] == board[row + 2, col + 2] and
                board[row, col] == board[row + 3, col + 3]):
                
                return int(board[row, col])

    return 0

# Step 9 - four_in_a_row_diagonal_up_right
def four_in_a_row_diagonal_up_right(board):
    for row in range(3, 6):
        for col in range(4):
            if (board[row, col] != 0 and
                board[row, col] == board[row - 1, col + 1] and
                board[row, col] == board[row - 2, col + 2] and
                board[row, col] == board[row - 3, col + 3]):

                return int(board[row, col])

    return 0

# Step 10 - check_winner
def check_winner(board):
    winner = four_in_a_row_horizontal(board)
    if winner != 0:
        return winner

    winner = four_in_a_row_vertical(board)
    if winner != 0:
        return winner

    winner = four_in_a_row_diagonal_down_right(board)
    if winner != 0:
        return winner

    winner = four_in_a_row_diagonal_up_right(board)
    if winner != 0:
        return winner

    return 0

# Step 11 - board_is_full
def column_full(board, column):
    return board[0, column] != 0


def valid_moves(board):
    return [column for column in range(7)
            if not column_full(board, column)]


def board_is_full(board):
    return len(valid_moves(board)) == 0

# Step 12 - is_terminal
def is_terminal(board):
    winner = check_winner(board)

    if winner != 0:
        return (True, winner)

    if board_is_full(board):
        return (True, 0)

    return (False, 0)

# Step 13 - other_player
def other_player(player):
    if player == 1:
        return 2
    return 1

# Step 14 - step_env
def step_env(board, column, player):
    # Drop the piece; drop_piece returns a new board
    new_board = drop_piece(board, column, player)

    # Check whether the game is over
    done, winner = is_terminal(new_board)

    # The next player is always the opponent
    next_player = other_player(player)

    return new_board, done, winner, next_player

# Step 15 - encode_board
import numpy as np

def encode_board(board, current_player):
    opponent = other_player(current_player)

    encoded = np.zeros((2, 6, 7), dtype=np.float32)

    encoded[0] = (board == current_player)
    encoded[1] = (board == opponent)

    return encoded

# Step 16 - board_to_torch_tensor
import torch

def board_to_torch_tensor(board, current_player):
    encoded = encode_board(board, current_player)

    tensor = torch.tensor(encoded, dtype=torch.float32)

    return tensor.unsqueeze(0)

# Step 17 - init_conv_backbone
import torch.nn as nn

def init_conv_backbone(in_channels=2, hidden_channels=16):
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )

# Step 18 - init_policy_head
import torch
import torch.nn as nn


def init_policy_head(hidden_channels=16, num_columns=7):
    """Return an nn.Module mapping (B, hidden_channels, 6, 7) -> (B, num_columns) logits."""
    return nn.Sequential(
        nn.AdaptiveAvgPool2d((1, num_columns)),
        nn.Conv2d(hidden_channels, 1, kernel_size=1),
        nn.Flatten(start_dim=1)
    )

# Step 19 - init_value_head
import torch
import torch.nn as nn


def init_value_head(hidden_channels=16):
    """Return an nn.Module mapping (B, hidden_channels, 6, 7) -> (B, 1) in (-1, 1)."""
    return nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(start_dim=1),
        nn.Linear(hidden_channels, 1),
        nn.Tanh()
    )

# Step 20 - build_policy_value_net
import torch
import torch.nn as nn


def build_policy_value_net(in_channels=2, hidden_channels=16, num_columns=7):
    """Compose backbone + policy head + value head into one nn.Module."""

    class PolicyValueNet(nn.Module):
        def __init__(self):
            super().__init__()

            self.backbone = init_conv_backbone(
                in_channels,
                hidden_channels
            )

            self.policy_head = init_policy_head(
                hidden_channels,
                num_columns
            )

            self.value_head = init_value_head(
                hidden_channels
            )

        def forward(self, x):
            features = self.backbone(x)

            policy_logits = self.policy_head(features)
            value = self.value_head(features)

            return policy_logits, value

    return PolicyValueNet()

# Step 21 - policy_value_forward
import torch
import torch.nn as nn


def policy_value_forward(net, encoded_board):
    """Run encoded_board (B,2,6,7) through net and return (logits, value)."""
    logits, value = net(encoded_board)

    return logits, value

# Step 22 - action_mask
import numpy as np

def action_mask(board):
    mask = np.zeros(7, dtype=bool)

    for column in valid_moves(board):
        mask[column] = True

    return mask

# Step 23 - masked_policy_logits
import torch
import numpy as np

def masked_policy_logits(logits, mask):
    mask = torch.as_tensor(
        mask,
        dtype=torch.bool,
        device=logits.device
    )

    masked_logits = logits.clone()
    masked_logits[..., ~mask] = float('-inf')

    return masked_logits

# Step 24 - masked_log_softmax
import torch

def masked_log_softmax(logits, mask):
    masked_logits = masked_policy_logits(logits, mask)

    return torch.log_softmax(masked_logits, dim=-1)

# Step 25 - sample_action_from_policy
import torch


def sample_action_from_policy(logits, mask, temperature=1.0):
    """Sample a legal column from a tempered masked categorical policy."""

    # Apply temperature
    scaled_logits = logits / temperature

    # Mask illegal columns
    masked_logits = masked_policy_logits(scaled_logits, mask)

    # Convert logits to probabilities
    probabilities = torch.softmax(masked_logits, dim=-1)

    # Sample one action
    action = torch.multinomial(probabilities, num_samples=1)

    return action.item()

# Step 26 - greedy_action_from_policy
import torch

def greedy_action_from_policy(logits, mask):
    masked_logits = masked_policy_logits(logits, mask)
    return int(torch.argmax(masked_logits, dim=-1).item())
def sample_action_from_policy(logits, mask, temperature=1.0):
    """Sample a legal column from a tempered masked categorical policy."""

    # Apply temperature
    scaled_logits = logits / temperature

    # Mask illegal columns
    masked_logits = masked_policy_logits(scaled_logits, mask)

    # Convert logits to probabilities
    probabilities = torch.softmax(masked_logits, dim=-1)

    # Sample one action
    action = torch.multinomial(probabilities, num_samples=1)

    return action.item()

# Step 27 - make_mcts_node
def make_mcts_node(prior=0.0, parent=None):
    return {
        'prior': prior,
        'visit_count': 0,
        'value_sum': 0.0,
        'children': {},
        'parent': parent
    }

# Step 28 - node_q_value
def node_q_value(node):
    if node['visit_count'] == 0:
        return 0.0

    return node['value_sum'] / node['visit_count']

# Step 29 - ucb_score
import math

def ucb_score(parent, child, c_puct=1.5):
    q = node_q_value(child)

    exploration = (
        c_puct
        * child['prior']
        * math.sqrt(parent['visit_count'])
        / (1 + child['visit_count'])
    )

    return float(q + exploration)

# Step 30 - select_best_child
def select_best_child(node, legal_actions, c_puct=1.5):
    best_action = None
    best_child = None
    best_score = float('-inf')

    for action in legal_actions:
        child = node['children'][action]

        score = ucb_score(node, child, c_puct)

        if score > best_score:
            best_score = score
            best_action = action
            best_child = child

    return best_action, best_child

# Step 31 - select_leaf
def select_leaf(root, c_puct=1.5):
    node = root

    while node.get('is_expanded', False):
        legal_actions = list(node['children'].keys())
        _, node = select_best_child(node, legal_actions, c_puct)

    return node

# Step 32 - evaluate_with_network
import numpy as np
import torch

def evaluate_with_network(net, state, to_play):
    # Encode board and add batch dimension
    encoded = board_to_torch_tensor(state, to_play)

    # Run network
    logits, value = policy_value_forward(net, encoded)

    # Legal move mask
    mask = action_mask(state)

    # Convert logits to log-probabilities, respecting legal moves
    log_probs = masked_log_softmax(logits, mask)

    # Convert log-probabilities to probabilities
    priors = torch.exp(log_probs)

    # Remove batch dimension and convert to numpy
    priors = priors.squeeze(0).detach().cpu().numpy()

    # Extract scalar value
    value = float(value.squeeze().detach().cpu().item())

    return priors, value

# Step 33 - expand_node
def expand_node(node, priors):
    board = node['board']
    player = node['to_play']

    for action in valid_moves(board):
        new_board = drop_piece(board, action, player)
        next_player = other_player(player)

        child = make_mcts_node(
            prior=float(priors[action]),
            parent=node
        )

        child['board'] = new_board
        child['to_play'] = next_player

        node['children'][action] = child

# Step 34 - backup_value
def backup_value(leaf, value):
    node = leaf

    while node is not None:
        node['visits'] += 1
        node['value_sum'] += value

        value = -value
        node = node['parent']
def make_mcts_node(prior=0.0, parent=None):
    return {
        'prior': prior,
        'visits': 0,
        'value_sum': 0.0,
        'children': {},
        'parent': parent
    }

# Step 35 - run_one_simulation
def run_one_simulation(root, net, c_puct=1.5):
    leaf = select_leaf(root, c_puct)

    board = leaf['board']
    to_play = leaf['to_play']

    done, winner = is_terminal(board)

    if done:
        if winner == 0:
            value = 0.0
        elif winner == to_play:
            value = 1.0
        else:
            value = -1.0
    else:
        priors, value = evaluate_with_network(
            net,
            board,
            to_play
        )

        expand_node(leaf, priors)

    backup_value(leaf, value)

# Step 36 - run_mcts
def run_mcts(state, to_play, net, num_simulations, c_puct):
    root = make_mcts_node()

    root['board'] = state.copy()
    root['to_play'] = to_play

    for _ in range(num_simulations):
        run_one_simulation(root, net, c_puct)

    return root

# Step 37 - visit_count_policy (not yet solved)
# TODO: implement

# Step 38 - mcts_choose_action (not yet solved)
# TODO: implement

# Step 39 - record_self_play_step (not yet solved)
# TODO: implement

# Step 40 - play_self_play_game (not yet solved)
# TODO: implement

# Step 41 - assign_value_targets (not yet solved)
# TODO: implement

# Step 42 - generate_self_play_batch (not yet solved)
# TODO: implement

# Step 43 - value_loss_mse (not yet solved)
# TODO: implement

# Step 44 - policy_loss_cross_entropy (not yet solved)
# TODO: implement

# Step 45 - l2_regularization_loss (not yet solved)
# TODO: implement

# Step 46 - combined_loss (not yet solved)
# TODO: implement

# Step 47 - encode_batch_states (not yet solved)
# TODO: implement

# Step 48 - iterate_minibatches (not yet solved)
# TODO: implement

# Step 49 - training_step (not yet solved)
# TODO: implement

# Step 50 - training_epoch (not yet solved)
# TODO: implement

# Step 51 - self_play_iteration (not yet solved)
# TODO: implement

# Step 52 - train_loop (not yet solved)
# TODO: implement

# Step 53 - random_policy_action (not yet solved)
# TODO: implement

# Step 54 - greedy_agent_action (not yet solved)
# TODO: implement

# Step 55 - play_one_match (not yet solved)
# TODO: implement

# Step 56 - match_win_rate (not yet solved)
# TODO: implement

# Step 57 - evaluate_against_random (not yet solved)
# TODO: implement

